"""Reverse-build UG's PP2/Scalp method as OUR OWN rule-based generator (proof-of-concept),
so we could eventually generate signals INDEPENDENTLY of UG instead of only copying.

What we can / cannot replicate (honest):
  - CANNOT: UG's final entry is an LLM ensemble vote (Claude/GPT/Grok/Deepseek) over hidden
    trigger conditions — non-deterministic, not reproducible without their prompt + 4 models.
  - CAN: the decoded skeleton — MA34/MA89 per TF (M5/M15/M30/H1), direction = trend-follow
    when the multi-TF stack is aligned else fade, entry at the M5 "value area" (MA89), SL a
    fixed ~10 price, and OUR unified exit (TP1@50pip + runner@150pip + SL→BE).

This v1 hypothesis for the TRIGGER (UG never published it): a TREND PULL-BACK to the value
area — when the higher-TF stack (M15/M30/H1) agrees on a trend and price pulls back to touch
the M5 MA89, enter in the trend direction. Cooldown to avoid clustering. ANTI-LOOKAHEAD: every
MA is from CLOSED bars only (merge_asof backward); the trade is simulated from the NEXT M1 bar.

Backtest only — generates nothing live. Run: python -X utf8 -m scripts.ug_replica
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.tcu_edge import load_m1, our_exit, PIP

SL_PRICE = 10.0          # UG's fixed "10 gia" stop
COOLDOWN_MIN = 30        # min gap between generated signals (avoid firing every bar)
TOUCH_PX = 0.5           # M5 bar must come within this of MA89 to count as a value-area touch


def tf_ma(m1, rule, s=34, l=89):
    """Resample M1 close → TF, MA34/MA89 on CLOSED bars (shifted so a bar's MA excludes itself
    is NOT needed — we use bar-END time + merge_asof backward, so only closed bars are visible)."""
    r = m1.set_index("time")["close"].resample(rule, label="right", closed="right").last().dropna()
    df = pd.DataFrame({"ma34": r.rolling(s).mean(), "ma89": r.rolling(l).mean()}).dropna()
    return df.reset_index().rename(columns={"time": "end"})


def attach(m1, tf, name):
    return pd.merge_asof(m1.sort_values("time"), tf, left_on="time", right_on="end",
                         direction="backward", suffixes=("", f"_{name}")).rename(
        columns={"ma34": f"{name}34", "ma89": f"{name}89", "end": f"end_{name}"})


def build(m1):
    df = m1.copy()
    for name, rule in (("m5", "5min"), ("m15", "15min"), ("m30", "30min"), ("h1", "60min")):
        df = attach(df, tf_ma(m1, rule), name)
    return df


def generate(df, cfg):
    """Trigger: higher-TF stack agrees on a trend AND price touches the M5 MA89 (value-area
    pull-back) → enter in the trend direction. cfg knobs (each an optional filter):
      m5_align   : also require M5 MA34 on the trend side (full 4/4 stack)
      rejection  : the touch bar must close back on the trend side of MA89 (reject the pull-back)
      min_sep    : require |H1 MA34 − H1 MA89| >= this (real H1 trend, not flat)
      skip_hours : set of UTC hours to skip (e.g. low-liquidity / news windows)
    """
    sigs = []
    last_ts = None
    cols = ["m534", "m589", "m1534", "m1589", "m3034", "m3089", "h134", "h189"]
    for x in df.itertuples():
        vals = [getattr(x, c) for c in cols]
        if any(pd.isna(v) for v in vals):
            continue
        m534, m589, m1534, m1589, m3034, m3089, h134, h189 = vals
        up = sum(1 for a, b in ((m1534, m1589), (m3034, m3089), (h134, h189)) if a > b)
        if up == 3:
            direction = "long"
        elif up == 0:
            direction = "short"
        else:
            continue
        long = direction == "long"
        if cfg.get("m5_align") and ((m534 > m589) != long):
            continue                                    # M5 must agree with the trend
        if cfg.get("min_sep") and abs(h134 - h189) < cfg["min_sep"]:
            continue                                    # H1 trend too flat
        if abs(x.close - m589) > TOUCH_PX:              # value-area touch
            continue
        if cfg.get("rejection") and ((x.close < m589) if long else (x.close > m589)):
            continue                                    # bar must close back on the trend side
        if cfg.get("skip_hours") and pd.Timestamp(x.time).hour in cfg["skip_hours"]:
            continue
        if last_ts is not None and (x.time - last_ts) < pd.Timedelta(minutes=COOLDOWN_MIN):
            continue
        mid = m589
        sl = mid - SL_PRICE if long else mid + SL_PRICE
        sigs.append({"ts": x.time, "direction": direction, "lo": mid - 1.5, "hi": mid + 1.5,
                     "sl": sl, "mid": mid})
        last_ts = x.time
    return sigs


def backtest(sigs, m1, t, hi_a, lo_a, cl):
    n = len(m1)
    filled = net = wins = r50 = 0
    for s in sigs:
        i0 = t.searchsorted(np.datetime64(s["ts"]))
        if not (0 < i0 < n):
            continue
        sign = 1 if s["direction"] == "long" else -1
        entry = s["mid"]
        j = None
        end = s["ts"] + pd.Timedelta(minutes=120)
        k = i0
        while k < n and t[k] <= np.datetime64(end):
            if lo_a[k] <= entry <= hi_a[k]:
                j = k
                break
            k += 1
        if j is None:
            continue
        usd, reach50, _, _ = our_exit(entry, sign, j, s["sl"], m1, hi_a, lo_a, cl)
        filled += 1
        net += usd
        wins += usd > 0
        r50 += reach50
    return len(sigs), filled, net, wins, r50


def main():
    m1 = load_m1()
    df = build(m1)
    t = m1["time"].values
    hi_a, lo_a, cl = m1["high"].values, m1["low"].values, m1["close"].values
    days = (m1["time"].max() - m1["time"].min()).days or 1

    CFGS = {
        "v1 (3/3 trend + touch)": {},
        "+ M5 aligned (4/4)": {"m5_align": True},
        "+ rejection": {"m5_align": True, "rejection": True},
        "+ H1 trend>=3px": {"m5_align": True, "rejection": True, "min_sep": 3.0},
        "+ H1 trend>=6px": {"m5_align": True, "rejection": True, "min_sep": 6.0},
        "+ skip Asia 22-2h": {"m5_align": True, "rejection": True, "min_sep": 3.0,
                              "skip_hours": {22, 23, 0, 1, 2}},
    }
    print(f"# UG-PP2 REPLICA — filter sweep, {m1['time'].min().date()}→{m1['time'].max().date()} "
          f"({days}d). Target = UG current: WR 83%, +$1.4/sig\n")
    print(f"{'config':<26}{'sig':>5}{'/day':>6}{'fill':>6}{'net$':>9}{'$/sig':>8}{'WR':>6}{'reach50':>8}")
    for label, cfg in CFGS.items():
        ns, fl, net, wins, r50 = backtest(generate(df, cfg), m1, t, hi_a, lo_a, cl)
        if fl == 0:
            print(f"{label:<26}{ns:>5}{ns/days:>6.1f}{fl:>6}  (no fills)")
            continue
        print(f"{label:<26}{ns:>5}{ns/days:>6.1f}{fl:>6}{net:>+9.2f}{net/ns:>+8.2f}"
              f"{wins/fl*100:>5.0f}%{r50/fl*100:>7.0f}%")
    print("\n  NOTE: pure-rule ceiling. The gap to UG's 83% is the LLM-ensemble + hidden filters;"
          " closing it needs the ML/walk-forward path (more data + learned setup scorer).")


if __name__ == "__main__":
    raise SystemExit(main())
