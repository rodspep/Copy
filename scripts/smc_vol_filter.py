"""Test a trailing volatility/trend regime filter on SMC.

Hypothesis: SMC bleeds in low-vol sideway regimes (range compresses → far 5R/8R
targets unreachable → runners dry up → death by cuts). Gate entries on a TRAILING
regime measure (no lookahead: 20-day ER + 20-day range%, computed from daily closes
strictly BEFORE the setup bar's day). Sweep thresholds; measure net / WR / maxDD /
max-underwater-days / #trades. Goal: cut the 119-day dead window WITHOUT killing
the runners (which would drop net).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scripts.optimize import m15, swings, htf_trend, USD, COST


def regime_arrays(d, win=20):
    """Return (er[i], rng_pct[i]) aligned to each m15 row i, anti-lookahead.

    Daily ER = |c_last - c_first| / sum|diff| over the last `win` daily closes
    strictly before the bar's calendar day. rng_pct = (max-min)/last over same.
    """
    daily = d.set_index("time")["close"].resample("1D").last().dropna()

    def _er(x):
        s = np.abs(np.diff(x)).sum()
        return abs(x[-1] - x[0]) / s if s else 0.0

    er_roll = daily.rolling(win).apply(_er, raw=True)
    rng_roll = (daily.rolling(win).max() - daily.rolling(win).min()) / daily
    # shift(1): value at day D uses the window ending at D-1 (available start of D)
    reg = pd.DataFrame({"day": daily.index,
                        "er": er_roll.shift(1).values,
                        "rng": (rng_roll.shift(1) * 100).values}).dropna()
    reg = reg.sort_values("day")
    left = d[["time"]].reset_index(drop=True)
    merged = pd.merge_asof(left, reg, left_on="time", right_on="day",
                           direction="backward")
    return merged["er"].values, merged["rng"].values


def smc_trades_gated(d, last_sh, last_sl, tp_mults, sl_buf, sweep_win,
                     htf=None, htf_align=0, retest=24, horizon=96,
                     er=None, rng=None, er_min=0.0, rng_min=0.0):
    """Exact copy of optimize.smc_trades + a regime gate at the setup bar i."""
    o, h, l, c, n = d.open.values, d.high.values, d.low.values, d.close.values, len(d)
    out, used = [], 0
    i = 6
    while i < n - 1:
        if i < used or np.isnan(last_sh[i]) or np.isnan(last_sl[i]):
            i += 1; continue
        swh, swl = last_sh[i], last_sl[i]
        seg = slice(max(0, i - sweep_win), i + 1)
        setup = None
        if l[seg].min() < swl and c[i] > swh and c[i] > o[i]:
            setup = 1
        elif h[seg].max() > swh and c[i] < swl and c[i] < o[i]:
            setup = -1
        if setup is None:
            i += 1; continue
        # ---- REGIME GATE (anti-lookahead: er/rng at i use data before today) ----
        if er is not None and not (np.isnan(er[i])) and er[i] < er_min:
            i += 1; continue
        if rng is not None and not (np.isnan(rng[i])) and rng[i] < rng_min:
            i += 1; continue
        # -------------------------------------------------------------------------
        if htf_align and htf is not None and htf[i] != setup:
            i += 1; continue
        if setup > 0:
            ob = l[seg].min(); entry = ob + (h[seg].max() - ob) * 0.5; sl = ob - sl_buf
        else:
            ob = h[seg].max(); entry = ob - (ob - l[seg].min()) * 0.5; sl = ob + sl_buf
        R = abs(entry - sl)
        if R <= 0 or R > 25:
            i += 1; continue
        j = next((k for k in range(i + 1, min(i + 1 + retest, n)) if l[k] <= entry <= h[k]), None)
        if j is None:
            i += 1; continue
        tps = [entry + setup * m * R for m in tp_mults]
        stop, rem, move, booked = sl, 1.0, 0.0, 0
        for k in range(j + 1, min(j + 1 + horizon, n)):
            if (l[k] <= stop) if setup > 0 else (h[k] >= stop):
                move += (stop - entry) * setup * rem; rem = 0; break
            for ti, (tp, fr) in enumerate(zip(tps, (0.5, 0.3, 0.2))):
                if ti < booked:
                    continue
                if (h[k] >= tp) if setup > 0 else (l[k] <= tp):
                    move += (tp - entry) * setup * fr; rem -= fr; booked = ti + 1
                    if booked == 1:
                        stop = entry
            if rem <= 1e-9:
                break
        if rem > 1e-9:
            move += (c[min(j + horizon, n - 1)] - entry) * setup * rem
        out.append({"ts": d.time.iloc[j], "usd": move * USD - COST})
        used = j + 1; i = j + 1
    return out


def metrics(tr):
    if not tr:
        return dict(n=0, net=0, wr=0, maxdd=0, uw=0)
    tr = sorted(tr, key=lambda x: x["ts"])
    usd = np.array([t["usd"] for t in tr])
    ts = pd.to_datetime([t["ts"] for t in tr])
    cum = np.cumsum(usd)
    peak = np.maximum.accumulate(cum)
    maxdd = (cum - peak).min()
    # longest underwater (days from a peak until equity reclaims it)
    uw = pd.Timedelta(0); start = None; pk = -1e18
    for t, cm in zip(ts, cum):
        if cm >= pk:
            pk = cm; start = None
        else:
            if start is None:
                start = t
            uw = max(uw, t - start)
    return dict(n=len(tr), net=usd.sum(), wr=(usd > 0).mean() * 100,
                maxdd=maxdd, uw=uw.days)


def main():
    d = m15(); sh, sl = swings(d); htf = htf_trend(d)
    er, rng = regime_arrays(d)
    base = dict(tp_mults=(4, 6, 10), sl_buf=2.0, sweep_win=8, htf=htf, htf_align=1)

    def run(er_min=0.0, rng_min=0.0):
        return smc_trades_gated(d, sh, sl, **base, er=er, rng=rng,
                                er_min=er_min, rng_min=rng_min)

    print(f"{'FILTER':28} {'#tr':>4} {'net':>7} {'WR':>4} {'maxDD':>7} {'UW(d)':>5}")
    print("-" * 60)
    m = metrics(run())
    print(f"{'baseline (no filter)':28} {m['n']:>4} {m['net']:>+7.0f} {m['wr']:>3.0f}% {m['maxdd']:>+7.0f} {m['uw']:>5}")
    for thr in (0.08, 0.10, 0.12, 0.15, 0.18, 0.20):
        m = metrics(run(er_min=thr))
        print(f"{'ER >= '+format(thr,'.2f'):28} {m['n']:>4} {m['net']:>+7.0f} {m['wr']:>3.0f}% {m['maxdd']:>+7.0f} {m['uw']:>5}")
    for thr in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        m = metrics(run(rng_min=thr))
        print(f"{'range% >= '+format(thr,'.1f'):28} {m['n']:>4} {m['net']:>+7.0f} {m['wr']:>3.0f}% {m['maxdd']:>+7.0f} {m['uw']:>5}")
    # a couple of combos
    for e, r in ((0.10, 1.5), (0.12, 2.0), (0.10, 2.5)):
        m = metrics(run(er_min=e, rng_min=r))
        print(f"{'ER>='+format(e,'.2f')+' & rng>='+format(r,'.1f'):28} {m['n']:>4} {m['net']:>+7.0f} {m['wr']:>3.0f}% {m['maxdd']:>+7.0f} {m['uw']:>5}")


if __name__ == "__main__":
    main()
