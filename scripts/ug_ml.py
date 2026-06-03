"""ML setup-scorer to push the UG-PP2 replica past the pure-rule ceiling (~73% WR) toward
UG's current ~83% — by LEARNING the hidden filter from price (distilling what UG's LLM does).

Pipeline (on 17 months of M5, anti-lookahead throughout):
  1. candidates : every M5 bar that touches the M5 MA89 (value-area pull-back) with a >=2/3
                  higher-TF (M15/M30/H1) trend majority → propose a trade in the trend direction.
  2. features   : per-TF MA34/89 separations + dist-to-value-area (ATR-normalized), stack
                  alignment, ATR, momentum, candle body/wicks, session — ALL from CLOSED bars.
  3. label      : reach50 = price reaches +50pip (+5 price) before SL (entry ∓10px), simulated
                  forward on M5 within 2h.
  4. walk-forward: expanding-window folds (train past → test future, never the reverse); pool
                  out-of-sample predictions; report WR vs signal-count at probability thresholds.
A threshold whose out-of-sample WR ≈ UG's 83% at a usable signal count = ML closed the gap.

Run: python -X utf8 -m scripts.ug_ml
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

PIP = 0.1
SL_PRICE = 10.0
TP1 = 50 * PIP            # +5 price
TOUCH = 0.5
HORIZON = 24             # M5 bars (~2h)
TFS = (("m15", "15min"), ("m30", "30min"), ("h1", "60min"))


def load_m5():
    d = pd.read_csv("data/xau/XAUUSD_M5.csv")
    d["time"] = pd.to_datetime(d["timestamp"], utc=True, format="ISO8601").dt.tz_localize(None)
    return d[["time", "open", "high", "low", "close"]].sort_values("time").reset_index(drop=True)


def atr(df, n=14):
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def add_structure(d):
    """SMC / market-structure features UG's LLMs actually read (all CAUSAL = prior bars only,
    no lookahead): liquidity (recent swing hi/lo), premium/discount (range position), BOS,
    liquidity sweep, fair-value-gap, Fibonacci 0.618. Computed at two lookbacks (≈2h, ≈8h)."""
    h, l, c, o = d["high"], d["low"], d["close"], d["open"]
    a = d["atr"].replace(0, np.nan)
    # FVG (3-bar imbalance), causal: bullish if low[i] > high[i-2]; bearish if high[i] < low[i-2]
    d["fvg_bull"] = (l > h.shift(2)).astype(float)
    d["fvg_bear"] = (h < l.shift(2)).astype(float)
    for L, tag in ((24, "s"), (96, "l")):                    # ~2h and ~8h structure
        swh = h.rolling(L).max().shift(1)                    # prior-range high = buy-side liquidity
        swl = l.rolling(L).min().shift(1)                    # prior-range low  = sell-side liquidity
        rng = (swh - swl).replace(0, np.nan)
        d[f"dist_swh_{tag}"] = (swh - c) / a                 # how far liquidity above (toward it = room up)
        d[f"dist_swl_{tag}"] = (c - swl) / a
        d[f"rangepos_{tag}"] = (c - swl) / rng               # 0=discount(lows) .. 1=premium(highs)
        d[f"bos_{tag}"] = ((c > swh).astype(float) - (c < swl).astype(float))  # +1 break up / -1 down
        # liquidity sweep: wick took prior high/low then closed back inside (stop-hunt reversal)
        d[f"sweep_{tag}"] = (((l < swl) & (c > swl)).astype(float)            # swept sell-side → bullish
                             - ((h > swh) & (c < swh)).astype(float))         # swept buy-side → bearish
        d[f"fib618_{tag}"] = ((c - (swh - 0.618 * rng)).abs() / a)            # dist to 0.618 retr level
        d[f"fvg_near_{tag}"] = (d["fvg_bull"].rolling(L).max()
                                - d["fvg_bear"].rolling(L).max())             # recent FVG bias
    return d


def build(m5):
    d = m5.copy()
    d["ma34"] = d["close"].rolling(34).mean()
    d["ma89"] = d["close"].rolling(89).mean()
    d["atr"] = atr(d)
    # higher-TF MA34/89 from CLOSED bars, merged back anti-lookahead
    for name, rule in TFS:
        r = m5.set_index("time")["close"].resample(rule, label="right", closed="right").last().dropna()
        tf = pd.DataFrame({f"{name}34": r.rolling(34).mean(),
                           f"{name}89": r.rolling(89).mean()}).dropna().reset_index()
        tf = tf.rename(columns={"time": "end"})
        d = pd.merge_asof(d, tf, left_on="time", right_on="end", direction="backward").drop(columns="end")
    d = add_structure(d)
    return d


def label_reach50(d, i, sign, entry, sl):
    hi, lo = d["high"].values, d["low"].values
    tp = entry + sign * TP1
    end = min(i + 1 + HORIZON, len(d))
    for k in range(i + 1, end):
        if (lo[k] <= sl) if sign > 0 else (hi[k] >= sl):     # SL-first (conservative)
            return 0
        if (hi[k] >= tp) if sign > 0 else (lo[k] <= tp):
            return 1
    return 0                                                  # didn't reach +50 in horizon


def make_dataset(d):
    rows, y, ts = [], [], []
    c = d["close"].values
    feats_cols = None
    for i in range(89, len(d) - 1):
        r = d.iloc[i]
        if any(pd.isna(r[x]) for x in ("ma89", "atr", "h189")) or r["atr"] <= 0:
            continue
        if abs(r["close"] - r["ma89"]) > TOUCH:               # value-area touch
            continue
        up = sum(1 for n in ("m15", "m30", "h1") if r[f"{n}34"] > r[f"{n}89"])
        if up >= 2:
            sign = 1
        elif up <= 1:
            sign = -1
        else:
            continue
        a = r["atr"]
        feat = {
            "dir": sign,
            "stack_up": up,
            "m5_sep": (r["ma34"] - r["ma89"]) / a,
            "m5_dist": (r["close"] - r["ma89"]) / a,
            "atr": a,
            "atr_rel": a / r["close"],
            "hour": r["time"].hour,
            "dow": r["time"].dayofweek,
            "body": (r["close"] - r["open"]) / a,
            "uwick": (r["high"] - max(r["close"], r["open"])) / a,
            "lwick": (min(r["close"], r["open"]) - r["low"]) / a,
        }
        for n in ("m15", "m30", "h1"):
            feat[f"{n}_sep"] = (r[f"{n}34"] - r[f"{n}89"]) / a
            feat[f"{n}_dist"] = (r["close"] - r[f"{n}89"]) / a
        for k in (3, 6, 12):
            feat[f"mom{k}"] = (c[i] - c[i - k]) / a
        entry = r["ma89"]
        sl = entry - SL_PRICE if sign > 0 else entry + SL_PRICE
        rows.append(feat)
        y.append(label_reach50(d, i, sign, entry, sl))
        ts.append(r["time"])
    X = pd.DataFrame(rows)
    return X, np.array(y), pd.to_datetime(ts)


def main():
    m5 = load_m5()
    d = build(m5)
    X, y, ts = make_dataset(d)
    base = y.mean()
    print(f"# {len(X)} candidates over {ts.min().date()}→{ts.max().date()} "
          f"({(ts.max()-ts.min()).days}d M5)")
    print(f"# base reach50 (all candidates, = the rule's WR with no scorer): {base*100:.1f}%\n")

    # WALK-FORWARD: 5 expanding folds (train past, test the next chronological slice)
    order = np.argsort(ts.values)
    Xo, yo = X.iloc[order].reset_index(drop=True), y[order]
    n = len(Xo)
    folds = 5
    edges = [int(n * k / (folds + 1)) for k in range(1, folds + 2)]
    oof_p = np.full(n, np.nan)
    for k in range(folds):
        tr_end = edges[k]
        te_end = edges[k + 1]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             max_depth=4, l2_regularization=1.0,
                                             min_samples_leaf=40, random_state=0)
        clf.fit(Xo.iloc[:tr_end], yo[:tr_end])
        oof_p[tr_end:te_end] = clf.predict_proba(Xo.iloc[tr_end:te_end])[:, 1]
    m = ~np.isnan(oof_p)
    p, yt = oof_p[m], yo[m]
    print(f"# walk-forward out-of-sample predictions: {m.sum()}")
    print(f"{'threshold':>10}{'signals':>9}{'/day':>7}{'WR(reach50)':>13}{'vs base':>9}")
    days = (ts.max() - ts.min()).days or 1
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        sel = p >= thr
        if sel.sum() == 0:
            continue
        wr = yt[sel].mean() * 100
        print(f"{thr:>10.2f}{sel.sum():>9}{sel.sum()/days:>7.1f}{wr:>12.1f}%{wr-base*100:>+8.1f}")
    print("\n  Target: a threshold with WR ≈ 83% (UG current) at a usable signal count = ML")
    print("  beat the pure-rule ceiling. (reach50 here ~ TP1-win under our exit.)")


if __name__ == "__main__":
    raise SystemExit(main())
