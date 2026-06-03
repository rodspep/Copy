"""Find the SWEET SPOT of each method (SMC + scalp), honestly: sweep parameters on a TRAIN
window, report the best config's OUT-OF-SAMPLE (test) result — so we pick by what generalizes,
not curve-fit. Verdict: which method/config is profitable OOS + robust month-to-month.

Train = M5 history before 2026-01-01 (~12 mo); Test = 2026-01-01 → now (~5 mo, held out).
Run: python -X utf8 -m scripts.optimize
"""
from __future__ import annotations

import collections
import itertools
import numpy as np
import pandas as pd

from scripts.smc_replica import m15, swings
from scripts.scalp_replica import load as load_scalp

USD = 1.0
COST = 0.30
CUT = pd.Timestamp("2026-01-01")


def htf_trend(d):
    """H1 trend direction (EMA50) aligned to each M15 bar, anti-lookahead (last closed H1)."""
    r = d.set_index("time")["close"].resample("60min", label="right", closed="right").last().dropna()
    e = r.ewm(span=50, adjust=False).mean()
    tf = pd.DataFrame({"end": r.index, "htf": np.sign(r.values - e.values)})
    m = pd.merge_asof(d[["time"]], tf, left_on="time", right_on="end", direction="backward")
    return m["htf"].fillna(0).values


# ---------------- SMC backtest (parametrized) ----------------
def smc_trades(d, last_sh, last_sl, tp_mults, sl_buf, sweep_win, htf=None, htf_align=0,
               retest=24, horizon=96):
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
        if htf_align and htf is not None and htf[i] != setup:    # only with the H1 trend
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


# ---------------- scalp backtest (parametrized) ----------------
def scalp_trades(d, tp_px, sl_buf, min_sep_atr, skip_hours, runner, horizon=36):
    e20, e34, e50, e89 = d.e20.values, d.e34.values, d.e50.values, d.e89.values
    h, l, c, atr = d.high.values, d.low.values, d.close.values, d.atr.values
    out, last = [], -999
    for i in range(90, len(d) - 1):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        up = e20[i] > e34[i] > e50[i] > e89[i]
        dn = e20[i] < e34[i] < e50[i] < e89[i]
        if not (up or dn):
            continue
        if abs(e20[i] - e89[i]) / atr[i] < min_sep_atr:    # trend-strength filter
            continue
        if skip_hours and d.time.iloc[i].hour in skip_hours:
            continue
        sign = 1 if up else -1
        touched = (l[i] <= e34[i] and l[i] >= e89[i] - sl_buf) if up else \
                  (h[i] >= e34[i] and h[i] <= e89[i] + sl_buf)
        if not touched or i - last < 6:
            continue
        entry = e50[i]
        sl = (e89[i] - sl_buf) if up else (e89[i] + sl_buf)
        if (entry - sl) * sign <= 0:
            continue
        j = next((k for k in range(i + 1, min(i + 1 + horizon, len(d))) if l[k] <= entry <= h[k]), None)
        if j is None:
            continue
        tp1, tp3 = entry + sign * tp_px, entry + sign * runner
        stop, rem, move, booked = sl, 1.0, 0.0, False
        for k in range(j + 1, min(j + 1 + horizon, len(d))):
            if (l[k] <= stop) if sign > 0 else (h[k] >= stop):
                move += (stop - entry) * sign * rem; rem = 0; break
            if not booked and ((h[k] >= tp1) if sign > 0 else (l[k] <= tp1)):
                move += (tp1 - entry) * sign * 0.5; rem -= 0.5; booked = True; stop = entry
            if booked and rem > 0 and runner and ((h[k] >= tp3) if sign > 0 else (l[k] <= tp3)):
                move += (tp3 - entry) * sign * rem; rem = 0; break
        if rem > 0:
            move += (c[min(j + horizon, len(d) - 1)] - entry) * sign * rem
        out.append({"ts": d.time.iloc[j], "usd": move * USD - COST}); last = i
    return out


def stats(tr, days):
    if not tr:
        return dict(n=0, net=0, ppt=0, wr=0, perday=0, posmo=0, nmo=0)
    net = sum(t["usd"] for t in tr); w = sum(t["usd"] > 0 for t in tr)
    bym = collections.defaultdict(float)
    for t in tr:
        bym[t["ts"].strftime("%Y-%m")] += t["usd"]
    return dict(n=len(tr), net=net, ppt=net / len(tr), wr=w / len(tr) * 100,
                perday=net / days, posmo=sum(v > 0 for v in bym.values()), nmo=len(bym))


def split(tr):
    return [t for t in tr if t["ts"] < CUT], [t for t in tr if t["ts"] >= CUT]


def equity_stats(tr, days):
    """Combined-portfolio stats incl. max drawdown on the cumulative equity (chrono order)."""
    if not tr:
        return dict(n=0, net=0, perday=0, maxdd=0, posmo=0, nmo=0, wr=0)
    tr = sorted(tr, key=lambda t: t["ts"])
    eq, peak, dd = 0.0, 0.0, 0.0
    for t in tr:
        eq += t["usd"]; peak = max(peak, eq); dd = min(dd, eq - peak)
    s = stats(tr, days)
    return dict(n=s["n"], net=s["net"], perday=s["perday"], maxdd=dd,
                posmo=s["posmo"], nmo=s["nmo"], wr=s["wr"])


def sweep(name, grid, fn, days_tr, days_te):
    rows = []
    for params in grid:
        tr = fn(params)
        a, b = split(tr)
        sa, sb = stats(a, days_tr), stats(b, days_te)
        rows.append((params, sa, sb))
    rows = [r for r in rows if r[1]["n"] >= 30]                  # need enough train trades
    rows.sort(key=lambda r: r[1]["net"], reverse=True)           # rank by TRAIN net
    print(f"\n#### {name}: top configs by TRAIN net (then their TEST/OOS result) ####")
    print(f"{'params':<42}{'TRAIN net/$ppt/WR':>22}{'TEST net/$ppt/WR/posmo':>26}")
    for params, sa, sb in rows[:5]:
        print(f"{str(params):<42}"
              f"{sa['net']:>+8.0f}/{sa['ppt']:>+5.2f}/{sa['wr']:>3.0f}%"
              f"{sb['net']:>+9.0f}/{sb['ppt']:>+5.2f}/{sb['wr']:>3.0f}%/{sb['posmo']}of{sb['nmo']}")
    return rows[0] if rows else None


def main():
    # SMC data
    dm = m15(); sh, sl = swings(dm)
    days = (dm.time.max() - dm.time.min()).days
    days_tr = (CUT - dm.time.min()).days; days_te = (dm.time.max() - CUT).days
    print(f"# train < {CUT.date()} ({days_tr}d) | test >= {CUT.date()} ({days_te}d)")

    htf = htf_trend(dm)
    smc_grid = [dict(tp_mults=tm, sl_buf=sb, sweep_win=sw, htf_align=ha)
                for tm in [(3, 5, 8), (4, 6, 10), (2, 4, 6)]
                for sb in [1.0, 2.0] for sw in [6, 8] for ha in [0, 1]]
    best_smc = sweep("SMC", smc_grid,
                     lambda p: smc_trades(dm, sh, sl, p["tp_mults"], p["sl_buf"], p["sweep_win"],
                                          htf=htf, htf_align=p["htf_align"]),
                     days_tr, days_te)

    # scalp data
    ds = load_scalp()
    scalp_grid = [dict(tp_px=tp, sl_buf=sb, min_sep_atr=ms, skip_hours=sk, runner=rn)
                  for tp in [5.0] for sb in [1.0, 2.0]
                  for ms in [0.0, 1.5, 3.0] for sk in [None, {22, 23, 0, 1, 2}]
                  for rn in [0.0, 15.0]]
    best_scalp = sweep("SCALP", scalp_grid,
                       lambda p: scalp_trades(ds, p["tp_px"], p["sl_buf"], p["min_sep_atr"],
                                              p["skip_hours"], p["runner"]),
                       days_tr, days_te)

    print("\n#### VERDICT (OOS) ####")
    for nm, best in (("SMC", best_smc), ("SCALP", best_scalp)):
        if not best:
            print(f"  {nm}: no config with enough trades"); continue
        p, sa, sb = best
        ok = "PROFITABLE" if sb["net"] > 0 and sb["ppt"] > 0 else "NOT profitable"
        print(f"  {nm} {p}\n     TEST: net ${sb['net']:+.0f}, ${sb['ppt']:+.2f}/trade, "
              f"{sb['perday']:+.2f}/day, WR {sb['wr']:.0f}%, {sb['posmo']}/{sb['nmo']} mo+ → {ok}")

    # ---- COMBINED PORTFOLIO: run both best configs, merge TEST trades, real equity + drawdown ----
    if best_smc and best_scalp:
        ps = best_smc[0]; pc = best_scalp[0]
        smc_te = split(smc_trades(dm, sh, sl, ps["tp_mults"], ps["sl_buf"], ps["sweep_win"],
                                  htf=htf, htf_align=ps["htf_align"]))[1]
        sc_te = split(scalp_trades(ds, pc["tp_px"], pc["sl_buf"], pc["min_sep_atr"],
                                   pc["skip_hours"], pc["runner"]))[1]
        eS, eC = equity_stats(smc_te, days_te), equity_stats(sc_te, days_te)
        eB = equity_stats(smc_te + sc_te, days_te)
        print("\n#### COMBINED PORTFOLIO (both, TEST/OOS) — diversification effect ####")
        print(f"{'':<10}{'net$':>8}{'/day':>8}{'maxDD$':>9}{'net/DD':>8}{'mo+':>7}")
        for nm, e in (("SMC", eS), ("scalp", eC), ("BOTH", eB)):
            ndd = (e["net"] / -e["maxdd"]) if e["maxdd"] < 0 else float("inf")
            print(f"  {nm:<8}{e['net']:>+8.0f}{e['perday']:>+8.2f}{e['maxdd']:>+9.0f}"
                  f"{ndd:>8.1f}{e['posmo']:>4}/{e['nmo']}")
        print("  → BOTH should show higher net/day with a BETTER net/maxDD (smoother) than either alone.")


if __name__ == "__main__":
    raise SystemExit(main())
