"""Explore levers to raise WR / net on the 50pip edge, under the LIVE deep-limit fill.

Reuses the Codex-reviewed simulate() from ug_method_pnl (deep-limit fill + fill-bar TP
suppression). Tests, on the 50pip bucket:
  1. entry edge (near/mid/deep) under deep-limit (live uses mid),
  2. QUALITY filters from the signal text (risk score, recommendation, MA trend
     alignment) — does dropping 'bad' signals raise WR/expectancy?
  3. expiry sensitivity (does the 240min wait cap fills?).

WARNING: in-sample ~1 week, small n. The more filters tried, the higher the overfit
risk. Treat any improvement as a HYPOTHESIS to forward-test, not a fact.
"""
from __future__ import annotations

import json
from scripts.ug_method_pnl import load_m1, detect_offset, simulate

SPREAD = 3.0


def agg(rows):
    wins = [r for r in rows if r["status"] == "win"]
    traded = wins + [r for r in rows if r["status"] == "loss"]
    usd = sum(r["usd"] for r in traded)
    wr = len(wins) / len(traded) * 100 if traded else 0
    mr = sum(r["R"] for r in traded) / len(traded) if traded else 0
    return len(traded), wr, mr, usd


def ma_aligns(s, n_tf):
    """True if the trade direction agrees with the MA34>MA89 trend on the n_tf
    lowest timeframes among M5,M15,M30,H1 (n_tf=1 -> H1 only ... here we use the
    HIGHER TFs as 'regime')."""
    order = ["H1", "M30", "M15", "M5"]
    want = "up" if s["direction"] == "long" else "down"
    ma = s.get("ma") or {}
    tfs = order[:n_tf]
    got = [(ma.get(tf) or {}).get("dir") for tf in tfs]
    return all(g == want for g in got)


def run(sigs, m1, off, label, expiry=240, mode="mid", chase=False, filt=None):
    rows = []
    for s in sigs:
        if filt and not filt(s):
            continue
        r = simulate(s, m1, off, expiry, SPREAD, mode_override=mode, chase_rule=chase)
        rows.append(r)
    n_sig = sum(1 for s in sigs if not filt or filt(s))
    n, wr, mr, usd = agg(rows)
    print(f"  {label:<34} sig {n_sig:>3} | closed {n:>3} | WR {wr:5.1f}% | meanR {mr:+.3f} | net ${usd:+8.2f}")


def main():
    sigs_all = [json.loads(l) for l in open("data/ug/signals.jsonl", encoding="utf-8") if l.strip()]
    m1 = load_m1()
    off = detect_offset(sigs_all, m1)
    sigs = [s for s in sigs_all if (s.get("tps_pip") or {}).get("1") == 50.0]
    print(f"# 50pip bucket: {len(sigs)} signals | deep-limit fill | in-sample ~1wk\n")

    print("--- 1) ENTRY EDGE (deep-limit) ---")
    for mode in ("near", "mid", "deep"):
        run(sigs, m1, off, f"entry={mode}", mode=mode)

    print("\n--- 2) QUALITY FILTERS (entry=mid, deep-limit) ---")
    run(sigs, m1, off, "baseline (no filter)")
    for thr in (7, 6, 5):
        run(sigs, m1, off, f"risk <= {thr}", filt=lambda s, t=thr: (s.get("risk") or 99) <= t)
    run(sigs, m1, off, "recommendation != CAUTION",
        filt=lambda s: (s.get("recommendation") or "").upper() != "CAUTION")
    run(sigs, m1, off, "H1 MA trend aligns", filt=lambda s: ma_aligns(s, 1))
    run(sigs, m1, off, "H1+M30 align", filt=lambda s: ma_aligns(s, 2))
    run(sigs, m1, off, "all 4 TF align", filt=lambda s: ma_aligns(s, 4))
    run(sigs, m1, off, "risk<=7 AND H1 aligns",
        filt=lambda s: (s.get("risk") or 99) <= 7 and ma_aligns(s, 1))

    print("\n--- 3) EXPIRY sensitivity (entry=mid, deep-limit) ---")
    for e in (60, 120, 240, 480):
        run(sigs, m1, off, f"expiry={e}min", expiry=e)


if __name__ == "__main__":
    raise SystemExit(main())
