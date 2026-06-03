"""LLM-in-the-loop test: generate scalp candidates (recent window) with a PRE-TRADE structure
snapshot (no outcome leak) + the true outcome stored separately. An LLM then grades take/skip
from the snapshot alone; we measure WR(take) vs the rule baseline — does a top LLM filter the
67%-WR rule scalp up toward UG's 83%? (This is exactly what UG's ensemble does.)

Output: data/ug/llm_scalp_candidates.json (snapshot + outcome) and prints snapshots for grading.
Run: python -X utf8 -m scripts.llm_candidates
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scripts.scalp_replica import load as load_scalp

W = pd.Timestamp("2026-05-18")     # UG current-version window
TP_PX, SL_BUF, MIN_SEP, HORIZON = 5.0, 2.0, 1.5, 36
SKIP_H = {22, 23, 0, 1, 2}


def snap(d, i, sign):
    """Compact pre-trade structure snapshot (what an SMC/scalp trader reads). No outcome."""
    r = d.iloc[i]
    a = r["atr"]
    e = {n: r[f"e{n}"] for n in (20, 34, 50, 89)}
    # recent 6 M5 candles as ATR-relative body/range (price action)
    seg = d.iloc[i - 5:i + 1]
    pa = []
    for _, x in seg.iterrows():
        body = (x.close - x.open) / a
        rng = (x.high - x.low) / a
        pa.append(f"{body:+.1f}/{rng:.1f}")
    # swing hi/lo over last 24 bars (liquidity)
    sw_h = d["high"].iloc[i - 24:i].max()
    sw_l = d["low"].iloc[i - 24:i].min()
    return {
        "dir": "LONG" if sign > 0 else "SHORT",
        "hour_utc": int(r["time"].hour),
        "ema_stack": f"e20={e[20]:.1f} e34={e[34]:.1f} e50={e[50]:.1f} e89={e[89]:.1f}",
        "ema_sep_atr": round(abs(e[20] - e[89]) / a, 1),         # trend strength
        "price": round(r["close"], 1),
        "px_vs_e50_atr": round((r["close"] - e[50]) / a, 1),
        "atr_px": round(a, 1),
        "atr_pct": round(a / r["close"] * 100, 3),                # volatility regime
        "dist_swhi_atr": round((sw_h - r["close"]) / a, 1),       # room to buy-side liquidity
        "dist_swlo_atr": round((r["close"] - sw_l) / a, 1),
        "last6_body/range_atr": " ".join(pa),                     # recent price action
    }


def outcome(d, i, sign, entry, sl):
    h, l = d["high"].values, d["low"].values
    tp = entry + sign * TP_PX
    for k in range(i + 1, min(i + 1 + HORIZON, len(d))):
        if (l[k] <= sl) if sign > 0 else (h[k] >= sl):
            return 0
        if (h[k] >= tp) if sign > 0 else (l[k] <= tp):
            return 1
    return 0


def main():
    d = load_scalp()
    e20, e34, e50, e89 = d.e20.values, d.e34.values, d.e50.values, d.e89.values
    h, l, atr = d.high.values, d.low.values, d.atr.values
    cands, last = [], -999
    for i in range(90, len(d) - 1):
        if d.time.iloc[i] < W or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        up = e20[i] > e34[i] > e50[i] > e89[i]
        dn = e20[i] < e34[i] < e50[i] < e89[i]
        if not (up or dn) or abs(e20[i] - e89[i]) / atr[i] < MIN_SEP:
            continue
        if d.time.iloc[i].hour in SKIP_H or i - last < 6:
            continue
        sign = 1 if up else -1
        touched = (l[i] <= e34[i] and l[i] >= e89[i] - SL_BUF) if up else \
                  (h[i] >= e34[i] and h[i] <= e89[i] + SL_BUF)
        if not touched:
            continue
        entry = e50[i]
        sl = (e89[i] - SL_BUF) if up else (e89[i] + SL_BUF)
        if (entry - sl) * sign <= 0:
            continue
        # require fill within horizon for a real trade
        j = next((k for k in range(i + 1, min(i + 1 + HORIZON, len(d))) if l[k] <= entry <= h[k]), None)
        if j is None:
            continue
        cands.append({"id": len(cands), "ts": d.time.iloc[i].isoformat(),
                      "snap": snap(d, i, sign), "outcome": outcome(d, j, sign, entry, sl)})
        last = i
    json.dump(cands, open("data/ug/llm_scalp_candidates.json", "w"), ensure_ascii=False, indent=1)
    base = np.mean([c["outcome"] for c in cands]) * 100
    print(f"# {len(cands)} scalp candidates on {W.date()}→{d.time.max().date()} | "
          f"baseline WR(reach+50) = {base:.0f}% (rule, no LLM filter)\n")
    print("# SNAPSHOTS for LLM grading (outcome hidden):")
    for c in cands:
        s = c["snap"]
        print(f"#{c['id']:>2} {s['dir']:<5} h{s['hour_utc']:>2} sep{s['ema_sep_atr']} "
              f"pxE50{s['px_vs_e50_atr']:+} atr%{s['atr_pct']} "
              f"swHi{s['dist_swhi_atr']} swLo{s['dist_swlo_atr']} | PA:{s['last6_body/range_atr']}")


if __name__ == "__main__":
    raise SystemExit(main())
