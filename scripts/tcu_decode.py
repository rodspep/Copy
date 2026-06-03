"""Decode v2 + edge analysis of UG signals, on the 647-signal labeled dataset
harvested from the Trade Coin Underground public API (data/ug/tcu/signals_full.json).

Each record carries UG's self-disclosed inputs (MA34/89 per TF, Elliott, SMC, risk
in `rawText`) AND the site's labeled outcome (status, maxFavorablePrice, hit times).
We reuse the Telegram parser (scripts.parse_ug_export.parse_signal) on `rawText`, merge
the API outcome fields, then:

  DECODE v2 (reconstruct the generator's rules, measure fidelity on real outcomes):
    - direction : fade-vs-trend hybrid (long when entry<=M5 MA34; trend-follow when the
                  multi-TF MA34/89 stack is >=3/4 aligned). Report % match.
    - anchor    : which MA / price the entry sits on (the only free geometric variable).
    - method    : predict the family (displayName) / template from the disclosed features.

  EDGE (what actually pays — feeds the live copier's method filter):
    - per method/template: WR (their labels) + MFE distribution + reach-rate by pip.

Read-only analysis; no network. Run: python -X utf8 -m scripts.tcu_decode
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from scripts.parse_ug_export import parse_signal

SRC = "data/ug/tcu/signals_full.json"
PIP = 0.1
TFS = ("M5", "M15", "M30", "H1")
WIN = ("TP1_HIT", "TP2_HIT", "TP3_HIT")


def load():
    sigs = json.load(open(SRC, encoding="utf-8"))["signals"]
    rows = []
    for s in sigs:
        raw = s.get("rawText") or ""
        p = parse_signal(s.get("createdAt") or "", raw) if raw else None
        rows.append({"api": s, "p": p, "raw": raw})
    return rows


def stack(p) -> str:
    """Per-TF MA34>MA89 alignment as a 4-char U/D string (M5,M15,M30,H1); '?' if absent."""
    ma = (p or {}).get("ma") or {}
    out = []
    for tf in TFS:
        m = ma.get(tf)
        out.append(("U" if m["dir"] == "up" else "D") if m else "?")
    return "".join(out)


def main():
    rows = load()
    n = len(rows)
    parsed = [r for r in rows if r["p"]]
    print(f"# TCU decode v2 — {n} signals; rawText parsed OK on {len(parsed)}\n")

    # ---- parse success + template by method family ----
    print("== parse success + TP template by displayName ==")
    by_name = defaultdict(list)
    for r in rows:
        by_name[r["api"].get("displayName")].append(r)
    for name, g in sorted(by_name.items(), key=lambda kv: -len(kv[1])):
        ok = [r for r in g if r["p"]]
        tmpl = Counter(str(r["api"].get("tpPips")) for r in g)
        top = ", ".join(f"{k}×{v}" for k, v in tmpl.most_common(3))
        print(f"  {name:<28} n={len(g):<4} parsed={len(ok):<4} tpl: {top}")
    print()

    # ===== DECODE 1: DIRECTION (fade-vs-trend hybrid) =====
    print("== DECODE 1: direction rule (fade default; trend-follow if stack >=3/4 aligned) ==")
    hit = miss = 0
    miss_stacks = Counter()
    for r in parsed:
        p = r["p"]
        ma = p["ma"]; m5 = ma.get("M5")
        if not m5:
            continue
        entryB = p.get("entry")          # anchor (2nd number / single)
        if entryB is None:
            continue
        st = stack(p)
        ups = st.count("U"); downs = st.count("D")
        # fade vs M5 MA34
        fade = "long" if entryB <= m5["ma34"] else "short"
        if ups >= 3:
            pred = "long"
        elif downs >= 3:
            pred = "short"
        else:
            pred = fade
        actual = p["direction"]
        if pred == actual:
            hit += 1
        else:
            miss += 1
            miss_stacks[st] += 1
    tot = hit + miss
    print(f"  match {hit}/{tot} = {hit/tot*100:.1f}%   (miss stacks: {dict(miss_stacks)})\n")

    # ===== DECODE 2: ENTRY ANCHOR (which MA does entry sit on?) =====
    print("== DECODE 2: entry anchor — |entryB - MA| median (price) per MA, all TFs ==")
    import statistics as st_
    diffs = defaultdict(list)
    for r in parsed:
        p = r["p"]; eB = p.get("entry"); ma = p["ma"]
        if eB is None:
            continue
        for tf in TFS:
            m = ma.get(tf)
            if m:
                diffs[f"{tf}.MA34"].append(abs(eB - m["ma34"]))
                diffs[f"{tf}.MA89"].append(abs(eB - m["ma89"]))
    for k, v in sorted(diffs.items(), key=lambda kv: st_.median(kv[1])):
        print(f"  {k:<10} median |Δ| {st_.median(v):.2f}  (n={len(v)})")
    print("  → smallest median = the MA the entry anchors to\n")

    # ===== DECODE 3: METHOD/TEMPLATE selection =====
    print("== DECODE 3: template vs MA-stack alignment (trend→wide TP, chop→tight scalp) ==")
    cell = defaultdict(Counter)
    for r in parsed:
        p = r["p"]; st = stack(p)
        aligned = max(st.count("U"), st.count("D"))
        bucket = "aligned>=3" if aligned >= 3 else "mixed<=2"
        cell[bucket][str(sorted(p.get("tps_pip", {}).values()))] += 1
    for bucket, c in cell.items():
        print(f"  {bucket}: " + ", ".join(f"{k}×{v}" for k, v in c.most_common(4)))
    print()

    # ===== EDGE: WR + MFE per method/template (their labels) =====
    print("== EDGE: outcome by displayName (resolved = TP/SL hit) ==")
    for name, g in sorted(by_name.items(), key=lambda kv: -len(kv[1])):
        res = [r for r in g if r["api"].get("status") in WIN + ("SL_HIT",)]
        if not res:
            continue
        win = sum(1 for r in res if r["api"]["status"] in WIN)
        # MFE in pip from entry (using API maxFavorablePrice vs entry, direction-aware)
        mfes = []
        for r in g:
            a = r["api"]; mfp = a.get("maxFavorablePrice"); e = a.get("entry")
            if mfp is not None and e:
                d = (mfp - e) if a.get("direction") == "BUY" else (e - mfp)
                mfes.append(d / PIP)
        med = st_.median(mfes) if mfes else float("nan")
        print(f"  {name:<28} resolved={len(res):<4} WR(anyTP)={win/len(res)*100:>4.0f}%  "
              f"MFE med={med:>5.0f}pip (n={len(mfes)})")
    print()

    # reach-rate by pip across ALL with MFE (does TP1@50 dominate?)
    print("== EDGE: reach-rate by TP distance (all signals w/ MFE) ==")
    allmfe = []
    for r in rows:
        a = r["api"]; mfp = a.get("maxFavorablePrice"); e = a.get("entry")
        if mfp is not None and e:
            d = ((mfp - e) if a.get("direction") == "BUY" else (e - mfp)) / PIP
            allmfe.append(d)
    for tp in (50, 100, 150, 200, 300):
        rch = sum(1 for m in allmfe if m >= tp)
        print(f"  +{tp:>3}pip reached by {rch}/{len(allmfe)} = {rch/len(allmfe)*100:.0f}%")


if __name__ == "__main__":
    raise SystemExit(main())
