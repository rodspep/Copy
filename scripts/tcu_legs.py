"""Test volume-split / leg-structure variants for the copier exit (user idea: price often
touches near then RUNS far, so maybe split the volume to capture more of the move).

Each leg is a full 0.01 lot (broker min), so N legs = 0.0N lot/signal — risk scales with
leg count. Uses tcu_edge.fill for the entry (limit-at-mid) and a generalized exit: legs
book at their pip distances, SL→BE after the FIRST leg books, SL-first per bar. Reports
$/signal + WR + full-straight-SL risk AT THE ACTUAL LOT, so a 3-leg's extra profit is
weighed against its extra risk (key on a small account).
"""
from __future__ import annotations
import pandas as pd
from scripts import tcu_edge as T
from scripts.tcu_edge import PIP, USD_PER_PRICE, COST


def exit_legs(entry, sign, j, sl0, m, hi_a, lo_a, cl, leg_pips):
    """leg_pips: list of TP pip distances; each leg = one 0.01-lot order. SL→BE after the
    first leg books. Returns total $/signal (sum of per-leg $; COST charged per leg)."""
    tps = [entry + sign * p * PIP for p in leg_pips]
    done = [False] * len(leg_pips); pnl_px = [0.0] * len(leg_pips)
    stop = sl0; booked = 0; k = j + 1
    while k < len(m) and not all(done):
        lo_b, hi_b = lo_a[k], hi_a[k]
        if (lo_b <= stop) if sign > 0 else (hi_b >= stop):
            for i in range(len(leg_pips)):
                if not done[i]:
                    pnl_px[i] = (stop - entry) * sign; done[i] = True
            break
        for i, tp in enumerate(tps):
            if done[i]:
                continue
            if (hi_b >= tp) if sign > 0 else (lo_b <= tp):
                pnl_px[i] = (tp - entry) * sign; done[i] = True; booked += 1
                if booked == 1:
                    stop = entry                       # SL → BE after first leg books
        k += 1
    for i in range(len(leg_pips)):
        if not done[i]:
            pnl_px[i] = (cl[len(m) - 1] - entry) * sign
    return sum(px * USD_PER_PRICE - COST for px in pnl_px)   # COST per leg


def main():
    m = T.load_m1(); sigs = T.load_signals()
    t = m["time"].values; hi = m["high"].values; lo = m["low"].values; cl = m["close"].values
    mmax = m["time"].max()
    g150 = [s for s in sigs if s["tp1_pip"] == 150 and s["ts"] <= mmax]

    STRUCTS = [
        ("2L 100+200 (đang chốt)", [100, 200]),
        ("2L 100+250 (runner xa)", [100, 250]),
        ("2L 100+300 (runner xa+)", [100, 300]),
        ("3L 100+150+250", [100, 150, 250]),
        ("3L 100+200+300", [100, 200, 300]),
        ("3L 100+150+300", [100, 150, 300]),
    ]
    for cut, wl in ((None, "FULL"), (mmax - pd.Timedelta(days=28), "28d"), (mmax - pd.Timedelta(days=15), "15d")):
        gg = [s for s in g150 if cut is None or s["ts"] >= cut]
        print(f"=== METHOD 150 — {wl} (n={len(gg)}) ===")
        print(f"  {'structure':24} {'lot':>4} {'$/sig':>6} {'WR':>4} {'fullSL$':>7}")
        for name, legs in STRUCTS:
            net = 0.0; f = 0; wins = 0
            for s in gg:
                fb = T.fill(s, m, t, lo, hi, cl)
                if fb is None:
                    continue
                e, sg, j = fb
                usd = exit_legs(e, sg, j, s["sl"], m, hi, lo, cl, legs)
                net += usd; f += 1; wins += usd > 0
            lot = 0.01 * len(legs)
            # full straight-SL: every leg stops at sl (~150pip) before any books
            full_sl = sum(150 * PIP * USD_PER_PRICE + COST for _ in legs)
            print(f"  {name:24} {lot:>4.2f} {net/len(gg) if gg else 0:>+6.2f} "
                  f"{wins/f*100 if f else 0:>3.0f}% {-full_sl:>7.1f}")


if __name__ == "__main__":
    main()
