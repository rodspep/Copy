"""Realizable live-state-machine backtest for the SMC bot — validates that the bot's
ACTUAL behaviour (place a pending on a closed-bar signal, ONE in-flight pending at a
time, fill only when price returns, cap concurrent filled setups) reproduces the
idealized smc_legged number (+$5562 / maxDD -$644 / 519 trades, which used look-ahead
fill detection). Parity rule: docs/decisions/smc_backtest_live_parity.md.

Differences from smc_legged this measures:
  - smc_legged scans ahead for the retest and JUMPS to fill+1 (no blocking on no-fill).
  - live blocks NEW detection while any pending is unfilled (up to expiry), then the
    pending cancels if never hit; filled setups coexist up to max_setups.
Post-FILL exit is identical (same SL-first / book-legs / BE / horizon math).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scripts.optimize import m15, swings, htf_trend, USD, COST
from src.exec.smc_logic import detect as smc_detect, RETEST_BARS, HORIZON


def _exit(o, h, l, c, n, sign, entry, slp, R, j, mults, units, be_after, horizon):
    """Identical to smc_sizing.smc_legged's post-fill exit. Returns (total_usd, res_bar)."""
    tps = [entry + sign * m * R for m in mults]
    leg_done = [False] * len(mults); leg_pnl = [0.0] * len(mults)
    stop, booked, res_bar = slp, 0, min(j + horizon, n - 1)
    for k in range(j + 1, min(j + 1 + horizon, n)):
        if (l[k] <= stop) if sign > 0 else (h[k] >= stop):
            for li in range(len(mults)):
                if not leg_done[li]:
                    leg_pnl[li] = (stop - entry) * sign; leg_done[li] = True
            res_bar = k; break
        for li, tp in enumerate(tps):
            if leg_done[li]:
                continue
            if (h[k] >= tp) if sign > 0 else (l[k] <= tp):
                leg_pnl[li] = (tp - entry) * sign; leg_done[li] = True; booked += 1
                if booked >= be_after:
                    stop = entry
        if all(leg_done):
            res_bar = k; break
    last_c = c[min(j + horizon, n - 1)]
    for li in range(len(mults)):
        if not leg_done[li]:
            leg_pnl[li] = (last_c - entry) * sign
    tot_u = sum(units)
    total = sum(leg_pnl[li] * units[li] for li in range(len(mults))) * USD - COST * tot_u
    return total, res_bar


def live_sim(d, sh, sl, htf, mults, units, be_after=1, max_setups=4, gate=True,
             expiry_bars=RETEST_BARS, horizon=HORIZON):
    """gate=True: ONE in-flight pending at a time (the conservative current bot).
    gate=False: multiple pendings allowed, total (pending+open) capped at max_setups —
    the realizable CEILING (captures every retest like smc_legged, at the cost of
    overlapping pendings). Returns (trades, max_concurrent)."""
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    n = len(d)
    trades, pendings, open_until, max_conc = [], [], [], 0
    for i in range(6, n - 1):
        open_until = [b for b in open_until if b > i]      # retire resolved trades
        # 1) fill / expire in-flight pendings
        still = []
        for (sgn, e, s, R, pb) in pendings:
            if i - pb > expiry_bars:
                continue                                    # never retested → cancel
            if l[i] <= e <= h[i]:                           # retest → fill at i
                usd, res = _exit(o, h, l, c, n, sgn, e, s, R, i, mults, units, be_after, horizon)
                trades.append({"ts": d.time.iloc[i], "usd": usd}); open_until.append(res)
            else:
                still.append((sgn, e, s, R, pb))
        pendings = still
        max_conc = max(max_conc, len(pendings) + len(open_until))
        # 2) detect a NEW setup, subject to the gate + the concurrency cap
        gate_ok = (len(pendings) == 0) if gate else True
        if gate_ok and (len(pendings) + len(open_until)) < max_setups:
            res = smc_detect(o, h, l, c, sh, sl, htf, i)
            if res is not None:
                pendings.append((*res, i))
    return trades, max_conc


def metrics(tr):
    if not tr:
        return dict(n=0, net=0, wr=0, maxdd=0)
    usd = np.array([t["usd"] for t in sorted(tr, key=lambda x: x["ts"])])
    cum = np.cumsum(usd); peak = np.maximum.accumulate(cum)
    return dict(n=len(tr), net=usd.sum(), wr=(usd > 0).mean() * 100, maxdd=(cum - peak).min())


def main():
    d = m15(); sh, sl = swings(d); htf = htf_trend(d)
    from scripts.smc_sizing import smc_legged, metrics as m2
    ideal = m2(smc_legged(d, sh, sl, [(4, 1), (10, 1)], htf=htf, be_after=1))
    print(f"{'MODEL':40} {'#tr':>4} {'net':>7} {'WR':>4} {'maxDD':>7}")
    print("-" * 66)
    print(f"{'idealized (smc_legged, look-ahead fill)':40} {ideal['n']:>4} {ideal['net']:>+7.0f} "
          f"{ideal['wr']:>3.0f}% {ideal['maxdd']:>+7.0f}    (not realizable)")
    for cap in (2, 3, 4, 5):
        tr, mc = live_sim(d, sh, sl, htf, [4, 10], [1, 1], max_setups=cap, gate=False)
        m = metrics(tr)
        mar = m["net"] / abs(m["maxdd"]) if m["maxdd"] else 0
        acct20 = abs(m["maxdd"]) / 0.20
        print(f"{'LIVE no-gate cap=' + str(cap):40} {m['n']:>4} {m['net']:>+7.0f} {m['wr']:>3.0f}% "
              f"{m['maxdd']:>+7.0f}  MAR {mar:.1f}  acct@20%% ${acct20:.0f}")


if __name__ == "__main__":
    main()
