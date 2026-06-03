"""VERIFY: is the 'skip if price already past TP1' rule wrong for the 150pip method?

Hypothesis (user): 50pip (PP2 scalp) entries sit NEAR the market (price between entry
and TP1 → 'don't chase if past TP1' is right). 150pip (PRI-GOLD) entries are LIMITs set
DEEPER — at signal time price is typically already beyond TP1 because the entry is a far
pull-back, so 'skip if past TP1' wrongly rejects valid deep limits.

For each signal, at its M1 close-at-signal price px0, classify px0's position for the
trade direction relative to entry_mid and TP1:
  long : below_entry | in_zone(entry..TP1) | above_TP1
  short: above_entry | in_zone(TP1..entry) | below_TP1
'above_TP1' (long) / 'below_TP1' (short) = what the current rule SKIPs as 'past TP1'.
Also report median |px0 - entry_mid| in price (how far the entry sits from market).
"""
from __future__ import annotations

import json
import pandas as pd

PIP = 0.1


def load_m1():
    df = pd.read_csv("data/xau/XAUUSD_M1.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def detect_offset(sigs, m1):
    t, c = m1["timestamp"].values, m1["close"].values
    best, best_err = 0, 1e9
    for off in range(-6, 7):
        errs = []
        for s in sigs:
            ts = pd.Timestamp(s["ts"]).tz_localize(None) + pd.Timedelta(hours=off)
            i = t.searchsorted(ts.to_datetime64())
            if 0 < i < len(c):
                errs.append(abs(c[i - 1] - (s["entry_low"] + s["entry_high"]) / 2))
        if errs:
            med = sorted(errs)[len(errs) // 2]
            if med < best_err:
                best_err, best = med, off
    return best


def main():
    sigs = [json.loads(l) for l in open("data/ug/signals.jsonl", encoding="utf-8") if l.strip()]
    m1 = load_m1()
    off = detect_offset(sigs, m1)
    t, c = m1["timestamp"].values, m1["close"].values

    buckets = {50.0: [], 100.0: [], 150.0: []}
    for s in sigs:
        tp1 = s.get("tps_pip", {}).get("1")
        if tp1 not in buckets:
            continue
        ts = pd.Timestamp(s["ts"]).tz_localize(None) + pd.Timedelta(hours=off)
        i = t.searchsorted(ts.to_datetime64())
        if not (0 < i < len(c)):
            continue
        px0 = c[i - 1]
        lo, hi = s["entry_low"], s["entry_high"]
        emid = (lo + hi) / 2
        long = s["direction"] == "long"
        sign = 1 if long else -1
        tp1_price = emid + sign * tp1 * PIP
        # position relative to entry and TP1, in the profit direction
        if long:
            if px0 <= emid:
                pos = "below_entry"
            elif px0 >= tp1_price:
                pos = "past_TP1(skip)"
            else:
                pos = "in_zone"
        else:
            if px0 >= emid:
                pos = "above_entry"
            elif px0 <= tp1_price:
                pos = "past_TP1(skip)"
            else:
                pos = "in_zone"
        buckets[tp1].append({"pos": pos, "dist": abs(px0 - emid),
                             "dir": s["direction"], "px0": px0, "emid": emid, "tp1": tp1_price})

    print(f"# tz UTC{off:+d} | position of market price vs entry/TP1 at signal time, per method\n")
    for tp1, rows in buckets.items():
        if not rows:
            print(f"=== TP1 {int(tp1)}pip: 0 signals ===\n"); continue
        import collections
        pc = collections.Counter(r["pos"] for r in rows)
        dists = sorted(r["dist"] for r in rows)
        med = dists[len(dists) // 2]
        print(f"=== TP1 {int(tp1)}pip: {len(rows)} signals ===")
        print(f"  median |price - entry_mid| = {med:.2f} price ({med/PIP:.0f} pip)")
        for k in ("below_entry", "above_entry", "in_zone", "past_TP1(skip)"):
            if pc.get(k):
                print(f"  {k:<16} {pc[k]:>3}  ({pc[k]/len(rows)*100:.0f}%)")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
