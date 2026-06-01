"""UG-style model on M1: limit-at-level + M1 rejection confirm + realistic costs.

The M5 backtest showed the edge dies under MARKET fills but lives under LIMIT
fills at the level. On M5 you can't have both a level-fill AND a close
confirmation in one bar. On M1 you can: the rejection candle is small and prints
right at the level, so you enter near the level (good fill) AND get confirmation.

Mechanic:
  - Levels = M5 swing highs (resistance) / lows (support), carried, aligned to M1
    by M5 CLOSE time (no lookahead).
  - On each M1 bar: LONG if the bar wicks to support (low<=S) and closes back above
    (close>S) having come from above (prev close>S) — an M1 rejection at support.
    Mirror for short at resistance.
  - Entry = that M1 bar's close (≈ at the level). SL = level ∓ sl. TP = entry ± tp.
  - Resolve forward on M1 (conservative SL-first ties). Costs (spread+slippage)
    charged per trade. Non-overlapping.

Run: python -X utf8 -m scripts.ug_m1_model
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.tv_loader import load_tv_csv
from src.indicators import swings

SL_PRICE = 10.0
COST = 0.4          # round-trip spread(0.2)+slippage(0.1 entry+0.1 exit) in price
RES_BARS = 240      # 4h on M1 to resolve, else timeout


def run(m1, S_arr, R_arr, tp_price, confirm=True):
    o, h, l, c = (m1[x].to_numpy() for x in ("open", "high", "low", "close"))
    n = len(m1)
    out = {"signals": 0, "tp": 0, "sl": 0, "timeout": 0}
    net_R = 0.0
    i = 2
    while i < n - 1:
        S, R = S_arr[i], R_arr[i]
        long_ = (not np.isnan(S) and c[i - 1] > S and l[i] <= S)
        short = (not np.isnan(R) and c[i - 1] < R and h[i] >= R)
        if confirm:
            long_ = long_ and c[i] > S
            short = short and c[i] < R
        if long_ and short:
            i += 1; continue
        if not (long_ or short):
            i += 1; continue
        if long_:
            entry = c[i]; sl, tp = S - SL_PRICE, entry + tp_price; side = 1
        else:
            entry = c[i]; sl, tp = R + SL_PRICE, entry - tp_price; side = -1
        out["signals"] += 1
        res, end = None, min(i + 1 + RES_BARS, n)
        for j in range(i + 1, end):
            hit_sl = h[j] >= sl if side < 0 else l[j] <= sl
            hit_tp = l[j] <= tp if side < 0 else h[j] >= tp
            if hit_sl:                          # conservative: SL first on tie
                res = "sl"
            elif hit_tp:
                res = "tp"
            if res:
                end = j
                break
        if res is None:
            out["timeout"] += 1; i = end; continue
        out[res] += 1
        # net R after costs (R = SL_PRICE). win=+tp, loss=-SL_PRICE, minus cost.
        gain = (tp_price - COST) if res == "tp" else -(SL_PRICE + COST)
        net_R += gain / SL_PRICE
        i = end + 1
    resolved = out["tp"] + out["sl"]
    out["WR"] = out["tp"] / resolved if resolved else 0.0
    out["expR"] = net_R / resolved if resolved else 0.0
    return out


def main() -> int:
    m5 = load_tv_csv("data/xau/XAUUSD_M5.csv").sort_values("timestamp").reset_index(drop=True)
    sw = swings(m5, left=5, right=5)
    m5 = m5.assign(R=sw["swing_high_price"], S=sw["swing_low_price"],
                   close_time=m5["timestamp"] + pd.Timedelta(minutes=5))
    m1 = load_tv_csv("data/xau/XAUUSD_M1.csv").sort_values("timestamp").reset_index(drop=True)
    # align M5 levels to M1 by M5 close-time (no lookahead)
    al = pd.merge_asof(m1[["timestamp"]], m5[["close_time", "S", "R"]].dropna(subset=["close_time"]),
                       left_on="timestamp", right_on="close_time", direction="backward")
    S_arr, R_arr = al["S"].to_numpy(), al["R"].to_numpy()
    span = f"{m1['timestamp'].min().date()} → {m1['timestamp'].max().date()}"
    print(f"M1: {len(m1)} bars · {span} · cost={COST}/trade (spread+slip)\n")
    print(f"{'method':>14} {'confirm':>8} {'sig':>5} {'WR':>5} {'breakeven':>9} {'expR(net)':>10}")
    for label, tp in [("scalp tp=5", 5.0), ("PRI-GOLD tp=15", 15.0)]:
        be = 1.0 / (1.0 + tp / SL_PRICE)
        for confirm in (False, True):
            r = run(m1, S_arr, R_arr, tp, confirm=confirm)
            print(f"{label:>14} {str(confirm):>8} {r['signals']:>5} {r['WR']:>4.0%} "
                  f"{be:>8.0%} {r['expR']:>+10.3f}")
    print("\nLIMIT/M1-confirm at level, costs included. expR in R/trade (R=SL=10).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
