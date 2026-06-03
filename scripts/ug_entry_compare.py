"""Where in the UG entry ZONE is best to enter: near edge / middle / deep edge?

For each captured UG signal, simulate placing a LIMIT at the near edge (1st number
shown), the middle, and the deep/favourable edge (2nd number), with the signal's
FIXED SL and TP1 measured FROM that entry. Resolve on M1 (fill within a window,
then TP1-before-SL, conservative, with spread/slippage cost). Report, per method
(TP1 pip group): fill rate, win rate (of filled), and NET price expectancy per
signal attempted (unfilled = 0) — the metric that accounts for missed fills.

Deeper entry = tighter risk to the fixed SL (better R:R) but lower fill rate.
This quantifies that trade-off on real UG signals.

Run: python -X utf8 -m scripts.ug_entry_compare
"""
from __future__ import annotations

import json
import collections

import numpy as np
import pandas as pd

from src.data.tv_loader import load_tv_csv

PIP = 0.1
COST = 0.4              # round-trip spread+slippage in price
FILL_BARS = 180        # M1 bars (3h) to get filled, else no trade
RES_BARS = 360         # M1 bars (6h) to resolve after fill


def main() -> int:
    m1 = load_tv_csv("data/xau/XAUUSD_M1.csv").sort_values("timestamp").reset_index(drop=True)
    ts_arr = m1["timestamp"].values
    hi, lo = m1["high"].to_numpy(), m1["low"].to_numpy()
    sigs = [json.loads(l) for l in open("data/ug/signals.jsonl", encoding="utf-8")]
    # dedup reposts
    seen, uniq = set(), []
    for s in sigs:
        k = (s["direction"], s["entry_low"], s["entry_high"], s["sl"])
        if k in seen:
            continue
        seen.add(k); uniq.append(s)

    def simulate(entry, sl, tp, side, start_idx):
        # fill: price reaches entry within FILL_BARS
        fill = None
        for j in range(start_idx, min(start_idx + FILL_BARS, len(m1))):
            if (side > 0 and lo[j] <= entry) or (side < 0 and hi[j] >= entry):
                fill = j; break
        if fill is None:
            return None        # not filled
        for j in range(fill + 1, min(fill + 1 + RES_BARS, len(m1))):
            hit_sl = lo[j] <= sl if side > 0 else hi[j] >= sl
            hit_tp = hi[j] >= tp if side > 0 else lo[j] <= tp
            if hit_sl:                     # conservative: SL first on same bar
                return -abs(entry - sl) - COST
            if hit_tp:
                return abs(tp - entry) - COST
        return 0.0             # filled, timed out (treat as scratch)

    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in uniq:
        tp1 = (s.get("tps_pip") or {}).get(1) or (s.get("tps_pip") or {}).get("1")
        if tp1 is None:
            continue
        side = 1 if s["direction"] == "long" else -1
        sl = s["sl"]
        lo_e, hi_e = s["entry_low"], s["entry_high"]
        idx = int(np.searchsorted(ts_arr, np.datetime64(pd.Timestamp(s["ts"]))))
        if idx >= len(m1):
            continue
        edges = {"near(1st)": lo_e, "mid": (lo_e + hi_e) / 2, "deep(2nd)": hi_e}
        for name, entry in edges.items():
            tp = entry + side * tp1 * PIP
            groups[tp1][name].append(simulate(entry, sl, tp, side, idx))

    print(f"UG signals (deduped): {len(uniq)} · M1 {m1['timestamp'].min().date()}→"
          f"{m1['timestamp'].max().date()} · cost {COST}/trade\n")
    print(f"{'TP1pip':>7} {'entry':>10} {'n':>4} {'fill%':>6} {'WR(filled)':>11} "
          f"{'net px/signal':>13}")
    for tp1 in sorted(groups):
        for name in ("near(1st)", "mid", "deep(2nd)"):
            res = groups[tp1][name]
            n = len(res)
            filled = [r for r in res if r is not None]
            wins = [r for r in filled if r > 0]
            fillpct = len(filled) / n if n else 0
            wr = len(wins) / len(filled) if filled else 0
            # net price expectancy per SIGNAL ATTEMPTED (unfilled counts as 0)
            net = np.mean([r if r is not None else 0.0 for r in res]) if n else 0
            tag = " <PP2 scalp" if tp1 == 50 else ""
            print(f"{tp1:>7g} {name:>10} {n:>4} {fillpct:>5.0%} {wr:>10.0%} "
                  f"{net:>+13.2f}{tag if name=='near(1st)' else ''}")
        print()
    print("net px/signal = avg price PnL per signal attempted (missed fills = 0). "
          "Higher = better entry choice. SL is fixed; deeper entry = tighter risk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
