"""Round 2: why the binary filter failed (re-entry lag) → try fixes that don't
step out of the market.

(A) Faster regime windows (5/10-day) — react quicker to compression/expansion.
(B) Vol-scaled SIZING (continuous): never flat, just trade smaller in low-vol and
    bigger in high-vol. Avoids the re-entry-lag trap that made UW longer.
Compare on net, maxDD, MAR (=net/|maxDD|, risk-adjusted), and UW days.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scripts.optimize import m15, swings, htf_trend
from scripts.smc_vol_filter import regime_arrays, smc_trades_gated, metrics


def main():
    d = m15(); sh, sl = swings(d); htf = htf_trend(d)
    base = dict(tp_mults=(4, 6, 10), sl_buf=2.0, sweep_win=8, htf=htf, htf_align=1)

    # baseline trades (size 1.0) + regime at each entry, several windows
    er20, rng20 = regime_arrays(d, win=20)
    er10, rng10 = regime_arrays(d, win=10)
    er5, rng5 = regime_arrays(d, win=5)
    tr0 = smc_trades_gated(d, sh, sl, **base)  # unfiltered, size 1.0

    def mar(m):
        return m["net"] / abs(m["maxdd"]) if m["maxdd"] else float("inf")

    def row(name, m):
        print(f"{name:30} {m['n']:>4} {m['net']:>+7.0f} {m['wr']:>3.0f}% "
              f"{m['maxdd']:>+7.0f} {mar(m):>5.1f} {m['uw']:>5}")

    print(f"{'VARIANT':30} {'#tr':>4} {'net':>7} {'WR':>4} {'maxDD':>7} {'MAR':>5} {'UW':>5}")
    print("-" * 66)
    row("baseline (size 1.0)", metrics(tr0))

    # ---- (A) faster ER windows, binary gate ----
    for tag, er in (("ER10", er10), ("ER5", er5)):
        for thr in (0.15, 0.20, 0.25):
            tr = smc_trades_gated(d, sh, sl, **base, er=er, er_min=thr)
            row(f"{tag} >= {thr:.2f}", metrics(tr))

    # ---- (A') faster range% windows (compression-sensitive) ----
    for tag, rng in (("rng10", rng10), ("rng5", rng5)):
        for thr in (1.5, 2.0, 2.5):
            tr = smc_trades_gated(d, sh, sl, **base, rng=rng, rng_min=thr)
            row(f"{tag}% >= {thr:.1f}", metrics(tr))

    # ---- (B) vol-scaled SIZING (never flat) ----
    # size factor from trailing ER at each trade's entry bar (lookup by bar time)
    er_ser = pd.Series(er20, index=d.time.values)

    def scaled(lo, hi, ref):
        out = []
        for t in tr0:
            e = er_ser.get(t["ts"], np.nan)
            f = 1.0 if (e != e) else float(np.clip(e / ref, lo, hi))
            out.append({"ts": t["ts"], "usd": t["usd"] * f})
        return out

    for lo, hi, ref in ((0.5, 1.5, 0.12), (0.4, 1.6, 0.12), (0.3, 1.8, 0.15),
                        (0.5, 2.0, 0.12), (0.25, 2.0, 0.12)):
        row(f"size clip[{lo},{hi}] ref{ref}", metrics(scaled(lo, hi, ref)))

    # ---- (B') step sizing: 0.5x low / 1x mid / 1.5x high vol ----
    def stepped(t1, t2, s_lo, s_mid, s_hi):
        out = []
        for t in tr0:
            e = er_ser.get(t["ts"], np.nan)
            f = s_mid if (e != e) else (s_lo if e < t1 else (s_hi if e >= t2 else s_mid))
            out.append({"ts": t["ts"], "usd": t["usd"] * f})
        return out

    for a, b, slo, smid, shi in ((0.10, 0.20, 0.5, 1.0, 1.5),
                                  (0.10, 0.20, 0.5, 1.0, 2.0),
                                  (0.08, 0.18, 0.5, 1.0, 1.5)):
        row(f"step[{a},{b}] {slo}/{smid}/{shi}", metrics(stepped(a, b, slo, smid, shi)))


if __name__ == "__main__":
    main()
