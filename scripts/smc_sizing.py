"""Decide the SMC bot's REAL leg structure + per-signal lot + account needed.

NOTE: the $/maxDD/account figures printed here use the IDEALIZED smc_legged model
(look-ahead fill, no concurrency cap) → +$5562 / maxDD -$644. That is NOT the deployable
target. For realizable sizing use scripts/smc_live_sim.py (no-gate cap 4): +$5019 /
maxDD -$915 → account $4500-5000. This script is kept for the leg-structure COMPARISON.


Broker min lot = 0.01 and lot step = 0.01 → you CANNOT split one 0.01 position
50/30/20. Each scale-out leg must be its own >=0.01 order. So the realistic
structures are:
  1-leg  (0.01 total): all-or-nothing single TP, or a runner (BE after +kR -> ride)
  2-leg  (0.02 total): 50/50 -> book leg A @ near TP, leg B runs; SL->BE after A
  3-leg  (0.03 total): 33/33/33 -> 3 TPs; SL->BE after first
Reference: the backtest's 50/30/20 needs 0.10 total (0.05/0.03/0.02).

Each leg = an independent 0.01-lot sub-position (1 "unit") with its own TP and a
shared SL that jumps to entry (BE) once `be_after` legs have booked. P/L per unit
= price move * USD(=1.0 per 0.01 lot). We report net / maxDD / WR / UW at the
structure's ACTUAL total lot, then the account needed to keep maxDD <= 20%/15%.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scripts.optimize import m15, swings, htf_trend, USD, COST
from src.exec.smc_logic import detect as smc_detect


def smc_legged(d, last_sh, last_sl, legs, sl_buf=2.0, sweep_win=8,
               htf=None, htf_align=1, retest=24, horizon=96, be_after=1):
    """legs = [(R_mult, units), ...]; units in multiples of 0.01 lot.
    SL -> entry (BE) once `be_after` legs have booked a TP."""
    o, h, l, c, n = d.open.values, d.high.values, d.low.values, d.close.values, len(d)
    out, used, i = [], 0, 6
    mults = [m for m, _ in legs]
    units = [u for _, u in legs]
    tot_units = sum(units)
    while i < n - 1:
        if i < used:
            i += 1; continue
        # detection via the SHARED core (single source of truth -> backtest<->live parity)
        res = smc_detect(o, h, l, c, last_sh, last_sl, htf, i,
                         sweep_win=sweep_win, sl_buf=sl_buf, htf_align=htf_align)
        if res is None:
            i += 1; continue
        setup, entry, sl, R = res
        j = next((k for k in range(i + 1, min(i + 1 + retest, n)) if l[k] <= entry <= h[k]), None)
        if j is None:
            i += 1; continue
        tps = [entry + setup * m * R for m in mults]
        stop = sl
        leg_done = [False] * len(legs)
        leg_pnl = [None] * len(legs)        # price move booked per leg unit
        booked = 0
        for k in range(j + 1, min(j + 1 + horizon, n)):
            # SL / BE check for still-open legs
            hit_stop = (l[k] <= stop) if setup > 0 else (h[k] >= stop)
            if hit_stop:
                for li in range(len(legs)):
                    if not leg_done[li]:
                        leg_pnl[li] = (stop - entry) * setup
                        leg_done[li] = True
                break
            for li, tp in enumerate(tps):
                if leg_done[li]:
                    continue
                if (h[k] >= tp) if setup > 0 else (l[k] <= tp):
                    leg_pnl[li] = (tp - entry) * setup
                    leg_done[li] = True
                    booked += 1
                    if booked >= be_after:
                        stop = entry
            if all(leg_done):
                break
        # horizon end: close remaining legs at last close
        last_c = c[min(j + horizon, n - 1)]
        for li in range(len(legs)):
            if not leg_done[li]:
                leg_pnl[li] = (last_c - entry) * setup
        move = sum(leg_pnl[li] * units[li] for li in range(len(legs)))
        out.append({"ts": d.time.iloc[j], "usd": move * USD - COST * tot_units})
        used = j + 1; i = j + 1
    return out


def metrics(tr):
    if not tr:
        return dict(n=0, net=0, wr=0, maxdd=0, uw=0)
    tr = sorted(tr, key=lambda x: x["ts"])
    usd = np.array([t["usd"] for t in tr]); ts = pd.to_datetime([t["ts"] for t in tr])
    cum = np.cumsum(usd); peak = np.maximum.accumulate(cum)
    maxdd = (cum - peak).min()
    uw = pd.Timedelta(0); start = None; pk = -1e18
    for t, cm in zip(ts, cum):
        if cm >= pk:
            pk = cm; start = None
        elif start is None:
            start = t
        if start is not None:
            uw = max(uw, t - start)
    return dict(n=len(tr), net=usd.sum(), wr=(usd > 0).mean() * 100,
                maxdd=maxdd, uw=uw.days)


def main():
    d = m15(); sh, sl = swings(d); htf = htf_trend(d)

    STRUCTS = [
        ("1-leg TP=6R",          [(6, 1)],               0.01, 1),
        ("1-leg TP=8R",          [(8, 1)],               0.01, 1),
        ("1-leg TP=10R",         [(10, 1)],              0.01, 1),
        ("2-leg 50/50 4R+10R",   [(4, 1), (10, 1)],      0.02, 1),
        ("2-leg 50/50 4R+8R",    [(4, 1), (8, 1)],       0.02, 1),
        ("2-leg 50/50 6R+12R",   [(6, 1), (12, 1)],      0.02, 1),
        ("3-leg 4R+6R+10R",      [(4, 1), (6, 1), (10, 1)], 0.03, 1),
        ("ref 50/30/20 (x0.10)", [(4, 5), (6, 3), (10, 2)], 0.10, 1),
    ]
    print(f"{'STRUCTURE':24} {'lot':>5} {'#sig':>4} {'net':>7} {'WR':>4} {'maxDD':>7} "
          f"{'UW':>4}  acct@20% acct@15%")
    print("-" * 80)
    for name, legs, lot, be in STRUCTS:
        m = metrics(smc_legged(d, sh, sl, legs, htf=htf, be_after=be))
        a20 = abs(m["maxdd"]) / 0.20
        a15 = abs(m["maxdd"]) / 0.15
        print(f"{name:24} {lot:>5.2f} {m['n']:>4} {m['net']:>+7.0f} {m['wr']:>3.0f}% "
              f"{m['maxdd']:>+7.0f} {m['uw']:>4}  ${a20:>6.0f}  ${a15:>6.0f}")


if __name__ == "__main__":
    main()
