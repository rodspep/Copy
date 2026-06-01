"""Reproduction test: does UG's decoded MECHANICAL core reproduce its claimed edge?

Hypothesis (from UG's own words "đớp 5 giá", "TP1 95%"): the edge is the tight-TP
fade geometry, not direction accuracy. TP1 = 50 pip = 5.0 price sits at half the
SL distance (10.0), entered as a limit ~1×ATR into a pullback from an extreme —
so mean-reversion makes TP1 hit far more often than SL, REGARDLESS of a "correct"
directional call.

This simulates that clone on MT5 M5 data and measures the TP1-before-SL rate.

Clone rule per bar i (signal at close, no lookahead):
  - direction = fade vs M5 SMA34: close>SMA34 → SHORT (sell rally), else LONG.
  - entry = limit k×ATR(M5) BEYOND current close in the fade direction
            (short: above; long: below) — UG enters deeper into the extreme.
  - SL = entry ∓ 10.0 ; TP1 = entry ± 5.0  (UG scalp geometry).
Resolution: from bar i+1, fill if price touches entry within FILL_BARS; then from
the bar AFTER the fill (we can't know intrabar order on the fill bar itself),
TP1 vs SL within RES_BARS. tie='sl' conservative / 'tp1' optimistic on same-bar
TP+SL. Resolved trades are non-overlapping (pending limits may still overlap).

Run: python -X utf8 -m scripts.ug_reproduce
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.tv_loader import load_tv_csv
from src.indicators import atr, swings

SL_PRICE = 10.0          # UG fixed SL (100 pip)
TP1_PRICE = 5.0          # UG scalp TP1 (50 pip)
FILL_BARS = 12           # 1h to get filled, else cancel
RES_BARS = 48            # 4h to resolve, else timeout


def simulate(df: pd.DataFrame, k_atr: float, tie: str = "sl") -> dict:
    """tie='sl' = conservative (SL wins same-bar ties); 'tp' = optimistic."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    sma34 = df["close"].rolling(34).mean().to_numpy()
    a = atr(df, 14).to_numpy()
    n = len(df)
    out = {"signals": 0, "filled": 0, "tp1": 0, "sl": 0, "timeout": 0, "cancelled": 0}

    i = 100
    while i < n - 1:
        if np.isnan(sma34[i]) or np.isnan(a[i]) or a[i] <= 0:
            i += 1
            continue
        short = close[i] > sma34[i]                 # fade: above mean → sell
        off = k_atr * a[i]
        if short:
            entry = close[i] + off
            sl, tp = entry + SL_PRICE, entry - TP1_PRICE
        else:
            entry = close[i] - off
            sl, tp = entry - SL_PRICE, entry + TP1_PRICE
        out["signals"] += 1

        # --- fill: price must touch the limit within FILL_BARS ---
        fill_j = None
        for j in range(i + 1, min(i + 1 + FILL_BARS, n)):
            if (short and high[j] >= entry) or (not short and low[j] <= entry):
                fill_j = j
                break
        if fill_j is None:
            out["cancelled"] += 1
            i += 1
            continue
        out["filled"] += 1

        # --- resolve TP1 vs SL starting the bar AFTER the fill ---
        # We must NOT score on the fill bar itself: we know price touched `entry`
        # there, but not whether TP/SL printed before or after the fill within
        # that candle (Codex finding — counting fill-bar TP inflates the rate).
        res, end = None, min(fill_j + 1 + RES_BARS, n)
        for j in range(fill_j + 1, end):
            hit_sl = high[j] >= sl if short else low[j] <= sl
            hit_tp = low[j] <= tp if short else high[j] >= tp
            if hit_sl and hit_tp:
                res = tie           # 'sl' pessimistic / 'tp1' optimistic on ties
            elif hit_sl:
                res = "sl"
            elif hit_tp:
                res = "tp1"
            if res:
                end = j
                break
        if res is None:
            out["timeout"] += 1
            i = end                # past the resolution window (non-overlapping)
        else:
            out[res] += 1
            i = end + 1            # continue after resolution
    return out


def simulate_sr(df: pd.DataFrame, tie: str = "sl", tp_price: float = TP1_PRICE,
                regime: np.ndarray | None = None, confirm: bool = False) -> dict:
    """Fade AT a swing level (S/R): SHORT when price retests the last confirmed
    swing HIGH (resistance), LONG when it retests the last confirmed swing LOW
    (support). Entry = the level; SL = level ∓10; TP1 = level ±5. A 'retest' is a
    fresh tag: prior close on the near side, this bar's range reaches the level
    (so we don't fire every bar once price is beyond an old level). Resolve from
    the bar AFTER the tag (no fill-bar lookahead). Non-overlapping."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    high, low, close = (df[c].to_numpy() for c in ("high", "low", "close"))
    sw = swings(df, left=5, right=5)
    R = sw["swing_high_price"].to_numpy()      # carried last confirmed swing high
    S = sw["swing_low_price"].to_numpy()       # carried last confirmed swing low
    n = len(df)
    out = {"signals": 0, "tp1": 0, "sl": 0, "timeout": 0}
    i = 100
    while i < n - 1:
        short = (not np.isnan(R[i]) and close[i - 1] < R[i] <= high[i])   # tag resistance from below
        long_ = (not np.isnan(S[i]) and close[i - 1] > S[i] >= low[i])    # tag support from above
        if confirm:                                  # require a REJECTION close at the level
            short = short and close[i] < R[i]        # wicked up into R but closed back below
            long_ = long_ and close[i] > S[i]        # wicked down into S but closed back above
        if regime is not None:                       # trade pullbacks WITH the HTF trend
            if regime[i] > 0:
                short = False                        # uptrend → only buy dips
            elif regime[i] < 0:
                long_ = False                        # downtrend → only sell rallies
            else:
                short = long_ = False
        if not (short or long_):
            i += 1
            continue
        if short:
            entry = R[i]; sl, tp = entry + SL_PRICE, entry - tp_price
        else:
            entry = S[i]; sl, tp = entry - SL_PRICE, entry + tp_price
        out["signals"] += 1
        res, end = None, min(i + 1 + RES_BARS, n)
        for j in range(i + 1, end):
            hit_sl = high[j] >= sl if short else low[j] <= sl
            hit_tp = low[j] <= tp if short else high[j] >= tp
            if hit_sl and hit_tp:
                res = tie
            elif hit_sl:
                res = "sl"
            elif hit_tp:
                res = "tp1"
            if res:
                end = j
                break
        if res is None:
            out["timeout"] += 1
            i = end
        else:
            out[res] += 1
            i = end + 1
    return out


def main() -> int:
    df = load_tv_csv("data/xau/XAUUSD_M5.csv")
    span = f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}"
    print(f"M5 data: {len(df)} bars · {span}\n")
    print(f"{'k×ATR':>6} {'tie':>5} {'filled':>7} {'TP1':>6} {'SL':>5} "
          f"{'TP1-rate':>9} {'expR(TP1=0.5R)':>15}")
    for k in (0.5, 0.9, 1.3):
        for tie in ("sl", "tp1"):
            r = simulate(df, k, tie=tie)
            resolved = r["tp1"] + r["sl"]
            tp1_rate = r["tp1"] / resolved if resolved else 0
            expR = (r["tp1"] * 0.5 - r["sl"] * 1.0) / resolved if resolved else 0
            lbl = "consv" if tie == "sl" else "optim"
            print(f"{k:>6.1f} {lbl:>5} {r['filled']:>7} {r['tp1']:>6} "
                  f"{r['sl']:>5} {tp1_rate:>8.0%} {expR:>+15.3f}")
    # HTF (H1) regime aligned to M5, no lookahead (use H1 close time = bar+1h).
    h1 = load_tv_csv("data/xau/XAUUSD_H1.csv").sort_values("timestamp").reset_index(drop=True)
    h1["reg"] = np.sign(h1["close"].rolling(34).mean() - h1["close"].rolling(89).mean())
    h1["close_time"] = h1["timestamp"] + pd.Timedelta(hours=1)
    m5s = df.sort_values("timestamp").reset_index(drop=True)
    merged = pd.merge_asof(m5s[["timestamp"]], h1[["close_time", "reg"]].dropna(),
                           left_on="timestamp", right_on="close_time", direction="backward")
    regime = merged["reg"].to_numpy()

    print("\n-- Entry = pull-back to swing level (S/R); TP measured FROM entry --")
    print(f"{'method':>14} {'filter':>10} {'sig':>5} {'WR':>5} {'breakeven':>9} {'expR':>8}")
    for label, tp in [("scalp TP=5", 5.0), ("PRI-GOLD TP=15", 15.0)]:
        rr = tp / SL_PRICE
        be = 1.0 / (1.0 + rr)
        for fname, kw in [("none", {}), ("H1-trend", {"regime": regime}),
                          ("confirm", {"confirm": True}),
                          ("confirm+H1", {"confirm": True, "regime": regime})]:
            r = simulate_sr(df, tie="sl", tp_price=tp, **kw)         # conservative
            resolved = r["tp1"] + r["sl"]
            wr = r["tp1"] / resolved if resolved else 0
            expR = (r["tp1"] * rr - r["sl"]) / resolved if resolved else 0
            print(f"{label:>14} {fname:>10} {r['signals']:>5} {wr:>4.0%} "
                  f"{be:>8.0%} {expR:>+8.3f}")
    print("\n(conservative ties. confirm = rejection close at the level; H1-trend "
          "= pullback with HTF trend. expR in R per resolved trade.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
