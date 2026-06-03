"""(a) Cross-source check: replay BOTH the old 93-signal Telegram export AND the 510-signal
TCU API set through the SAME engine (scripts.tcu_edge: zone-aware fill + our unified exit on
M1). If the "50pip is the edge" conclusion differs between sources under one engine, the
difference is the DATA (period/selection/source), not the execution model. (Codex item 5.)

Read-only. Run: python -X utf8 -m scripts.tcu_replay_compare
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from scripts.tcu_edge import load_m1, load_signals, fill, our_exit, PIP


def detect_offset(sigs, m, t, cl):
    """The Telegram export ts has an uncertain tz; find the hour-shift that best aligns the
    signal's entry-anchor with the M1 close just before it (price-based, like analyze_100_150)."""
    best, be = 0, 1e9
    for off in range(-6, 7):
        e = []
        for s in sigs:
            ts = (pd.Timestamp(s["ts"]).tz_localize(None) + pd.Timedelta(hours=off))
            i = t.searchsorted(np.datetime64(ts))
            if 0 < i < len(cl):
                e.append(abs(cl[i - 1] - (s["entry_low"] + s["entry_high"]) / 2))
        if e:
            md = sorted(e)[len(e) // 2]
            if md < be:
                be, best = md, off
    return best


def load_telegram(m, t, cl):
    raw = [json.loads(l) for l in open("data/ug/signals.jsonl", encoding="utf-8") if l.strip()]
    off = detect_offset(raw, m, t, cl)
    out = []
    for s in raw:
        tp = (s.get("tps_pip") or {}).get(1) or (s.get("tps_pip") or {}).get("1")
        out.append({
            "ts": pd.Timestamp(s["ts"]).tz_localize(None) + pd.Timedelta(hours=off),
            "direction": s["direction"],
            "lo": float(min(s["entry_low"], s["entry_high"])),
            "hi": float(max(s["entry_low"], s["entry_high"])),
            "sl": float(s["sl"]), "tp1_pip": tp, "name": "telegram",
            "entry_api": (s["entry_low"] + s["entry_high"]) / 2,
        })
    return out, off


def run(sigs, m, t, hi_a, lo_a, cl, mmax):
    res = []
    for s in sigs:
        if s["ts"] > mmax:
            continue
        fb = fill(s, m, t, lo_a, hi_a, cl)
        if fb is None:
            res.append({**s, "filled": False})
            continue
        entry, sign, j = fb
        usd, r50, r150, ssl = our_exit(entry, sign, j, s["sl"], m, hi_a, lo_a, cl)
        res.append({**s, "filled": True, "usd": usd, "r50": r50})
    return [r for r in res if r["ts"] <= mmax]


def by_template(res):
    out = {}
    for tp in sorted(set(r["tp1_pip"] for r in res if r["tp1_pip"] is not None)):
        rows = [r for r in res if r["tp1_pip"] == tp]
        f = [r for r in rows if r.get("filled")]
        net = sum(r["usd"] for r in f)
        wr = (sum(1 for r in f if r["usd"] > 0) / len(f) * 100) if f else 0
        out[tp] = (len(rows), len(f), net, net / len(rows) if rows else 0, wr)
    return out


def main():
    m = load_m1()
    t, hi_a, lo_a, cl = m["time"].values, m["high"].values, m["low"].values, m["close"].values
    mmax = m["time"].max()

    api = run(load_signals(), m, t, hi_a, lo_a, cl, mmax)
    tel_sigs, off = load_telegram(m, t, cl)
    tel = run(tel_sigs, m, t, hi_a, lo_a, cl, mmax)

    print(f"# SAME engine, two sources. Telegram tz-offset detected: UTC{off:+d}")
    print(f"# API in-coverage: {len(api)} | Telegram in-coverage: {len(tel)}\n")
    print(f"{'TP1':<8}{'source':<10}{'recv':>5}{'fill':>6}{'net$':>9}{'$/recv':>8}{'WR':>6}")
    a, te = by_template(api), by_template(tel)
    for tp in sorted(set(a) | set(te)):
        for src, d in (("API", a), ("Telegram", te)):
            if tp in d:
                recv, fl, net, pr, wr = d[tp]
                print(f"{str(tp)+'pip':<8}{src:<10}{recv:>5}{fl:>6}{net:>+9.2f}{pr:>+8.2f}{wr:>5.0f}%")
        print()
    # headline: is 50pip the best per-recv in EACH source?
    for src, d in (("API", a), ("Telegram", te)):
        if not d:
            continue
        best = max(d, key=lambda k: d[k][3])
        print(f"  {src}: best $/recv template = {best}pip (${d[best][3]:+.2f}); "
              f"50pip = ${d.get(50,(0,0,0,0,0))[3]:+.2f}/recv")


if __name__ == "__main__":
    raise SystemExit(main())
