"""Compare EXIT strategies (TP1 vs TP2/TP3 vs partial+breakeven) on the UG signals
that actually FILL, simulated bar-by-bar on real M1 XAU.

Baseline today = full exit at TP1 (0.5RR for the 50pip method). UG provides TP1..TP4;
this asks whether holding to TP2/TP3 or scaling out + moving SL to breakeven beats it.

Method (faithful + conservative):
  - Fill like the live copier (DEEP-LIMIT): place if price is on the fillable side of
    entry (all MID), wait for the pull-back fill or `expiry_min`; no TP-first cancel.
    (Same fill set compared across strategies.)
  - From the fill bar, walk M1. Each bar, ADVERSE side first (touch stop => close all
    remaining at stop) THEN favourable (book any TP legs reached, in ascending order).
    Same-bar ambiguity therefore always resolves against us (conservative).
  - Strategies = a ladder of (fraction, tp_pip) legs + optional move-SL-to-breakeven
    once the first (TP1) leg books.
  - Cost: one round-trip spread charged per trade (slightly optimistic for multi-leg).
  - R = price move / |entry - original SL|; $ = price move * 100 * lot (0.01 => $1/price).
  - Unresolved (ran out of M1) => remaining closed at last close, flagged in count.

Reports per TP1 bucket; the 50pip (PP2) bucket is the one with a real edge.
"""
from __future__ import annotations

import json
import math
import sys
import pandas as pd

PIP = 0.1
LOT = 0.01
USD_PER_PRICE = 100 * LOT
MODE = "mid"                       # copier trades all methods at MID now


def load_m1():
    df = pd.read_csv("data/xau/XAUUSD_M1.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


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


def fill_index(s, m1, off, expiry_min):
    """Return (entry, sl, sign, tps_pip_dict, fill_bar_index) or None if not filled
    (skip/cancel/expire) — mirrors the copier's placement rules."""
    tp1 = s.get("tps_pip", {}).get("1")
    if tp1 is None or not math.isfinite(tp1):
        return None
    lo, hi, sl = s["entry_low"], s["entry_high"], s["sl"]
    long = s["direction"] == "long"
    entry = (lo + hi) / 2 if MODE == "mid" else (lo if MODE == "near" else hi)
    sign = 1 if long else -1
    tp1_price = entry + sign * tp1 * PIP

    ts = pd.Timestamp(s["ts"]).tz_localize(None) + pd.Timedelta(hours=off)
    t = m1["timestamp"].values
    i0 = t.searchsorted(ts.to_datetime64())
    if i0 <= 0 or i0 >= len(m1):
        return None
    px0 = m1["close"].values[i0 - 1]
    # DEEP-LIMIT fill (matches the live copier): place if price is on the fillable side
    # of entry (buy-limit below market), then WAIT for the pull-back to entry. No
    # 'skip/cancel if TP1 touched first' — we are placing a limit, not chasing.
    ok = (px0 > entry) if long else (px0 < entry)
    if not ok:
        return None

    hi_a, lo_a, tm = m1["high"].values, m1["low"].values, m1["timestamp"].values
    end = ts + pd.Timedelta(minutes=expiry_min)
    j = i0
    while j < len(m1) and tm[j] <= end.to_datetime64():
        if (lo_a[j] <= entry) if long else (hi_a[j] >= entry):
            return (entry, sl, sign, s["tps_pip"], j)     # filled on pull-back
        j += 1
    return None                                  # expired unfilled


def resolve(m1, fill, legs, be_after_tp1, spread_pip, be_to="entry"):
    """Walk M1 from the fill bar; return (usd, R, resolved_bool). legs = list of
    (fraction, tp_pip). Adverse-first each bar (conservative)."""
    entry, sl0, sign, tps, j = fill
    risk = abs(entry - sl0)
    hi_a, lo_a, cl = m1["high"].values, m1["low"].values, m1["close"].values
    # build absolute TP prices, ascending in the direction of profit
    ladder = [(frac, entry + sign * float(tps[str(k)]) * PIP, k) for frac, k in legs]
    stop = sl0
    remaining = sum(f for f, _ in legs)
    move = 0.0                                   # accumulated signed price*fraction
    idx = 0
    k = j
    n = len(m1)
    booked_tp1 = False
    while k < n and remaining > 1e-9:
        lo_b, hi_b = lo_a[k], hi_a[k]
        # 1) adverse: stop hit -> close all remaining at stop
        adverse = (lo_b <= stop) if sign > 0 else (hi_b >= stop)
        if adverse:
            move += (stop - entry) * sign * remaining
            remaining = 0.0
            return (move * USD_PER_PRICE - spread_pip * PIP * USD_PER_PRICE, move / risk, True)
        # 2) favourable: book any TP legs reached this bar, ascending.
        #    SUPPRESSED on the fill bar (k==j): the bar's high may have occurred BEFORE
        #    the limit filled, so a same-bar TP would be optimistic (Codex). TP from j+1.
        while k > j and idx < len(ladder):
            frac, tp_price, klevel = ladder[idx]
            reached = (hi_b >= tp_price) if sign > 0 else (lo_b <= tp_price)
            if not reached:
                break
            move += (tp_price - entry) * sign * frac
            remaining -= frac
            idx += 1
            if klevel == 1:
                booked_tp1 = True
                if be_after_tp1 and be_to == "tp1":
                    # LOCK TP1: move the runner's stop to the TP1 level (guarantees ~TP1
                    # profit on the runner). It sits AT the just-touched price, so we do
                    # NOT same-bar check it (that would insta-stop); it applies from the
                    # next bar's adverse check.
                    stop = tp_price
                elif be_after_tp1:               # be_to == "entry": break-even
                    stop = entry
                    # CONSERVATIVE: if THIS bar's range also reaches breakeven, assume
                    # the pull-back to BE happened before any further extension —
                    # stop the runner at BE now, do NOT let TP2/TP3 book this same bar.
                    be_hit = (lo_b <= stop) if sign > 0 else (hi_b >= stop)
                    if be_hit and remaining > 1e-9:
                        move += (stop - entry) * sign * remaining   # = 0 at BE
                        remaining = 0.0
                        return (move * USD_PER_PRICE - spread_pip * PIP * USD_PER_PRICE,
                                move / risk, True)
        k += 1
    if remaining > 1e-9:
        # ran out of data: mark remaining to last close (flag unresolved)
        move += (cl[n - 1] - entry) * sign * remaining
        return (move * USD_PER_PRICE - spread_pip * PIP * USD_PER_PRICE, move / risk, False)
    return (move * USD_PER_PRICE - spread_pip * PIP * USD_PER_PRICE, move / risk, True)


def resolve_trail(m1, fill, tp1_frac, trail_pip, spread_pip):
    """Trailing-stop strategies. tp1_frac (0 or 0.5) = portion banked at TP1 first;
    the rest TRAILS by `trail_pip` behind the high-water mark, with a break-even floor
    once TP1 is banked. tp1_frac=0 => trail the whole position from fill (no partial).

    Conservative bar model: each bar, check the stop AS IT STOOD AT THE PRIOR BAR CLOSE
    (adverse-first), THEN raise the trail from this bar's extreme (never loosen). The
    partial's same-bar break-even stop IS re-checked (matches resolve())."""
    entry, sl0, sign, tps, j = fill
    risk = abs(entry - sl0)
    hi_a, lo_a, cl = m1["high"].values, m1["low"].values, m1["close"].values
    tp1_price = entry + sign * float(tps["1"]) * PIP
    long = sign > 0
    stop = sl0
    hwm = entry                                  # favourable extreme so far
    remaining = 1.0
    move = 0.0
    booked_tp1 = (tp1_frac <= 0)
    trail_on = (tp1_frac <= 0)                   # full-trail: active from fill
    k, n = j, len(m1)
    while k < n and remaining > 1e-9:
        lo_b, hi_b = lo_a[k], hi_a[k]
        # 1) adverse: stop (as of prior close) hit -> close remaining at stop
        if (lo_b <= stop) if long else (hi_b >= stop):
            move += (stop - entry) * sign * remaining
            return (move * USD_PER_PRICE - spread_pip * PIP * USD_PER_PRICE, move / risk, True)
        # Favourable actions SUPPRESSED on the fill bar (k==j): its extreme may predate
        # the limit fill, so a same-bar TP/trail would be optimistic (Codex). From j+1.
        # 2) bank the TP1 partial the first bar it is reached
        if k > j and not booked_tp1:
            if (hi_b >= tp1_price) if long else (lo_b <= tp1_price):
                move += (tp1_price - entry) * sign * tp1_frac
                remaining -= tp1_frac
                booked_tp1 = True
                trail_on = True
                stop = max(stop, entry) if long else min(stop, entry)   # BE floor
                if (lo_b <= stop) if long else (hi_b >= stop):           # same-bar BE
                    move += (stop - entry) * sign * remaining            # = 0 at BE
                    return (move * USD_PER_PRICE - spread_pip * PIP * USD_PER_PRICE,
                            move / risk, True)
        # 3) raise the trail from THIS bar's extreme (ratchet up only), for next bar
        if k > j and trail_on:
            hwm = max(hwm, hi_b) if long else min(hwm, lo_b)
            newstop = hwm - trail_pip * PIP if long else hwm + trail_pip * PIP
            if booked_tp1 and tp1_frac > 0:                              # BE floor on runner
                newstop = max(newstop, entry) if long else min(newstop, entry)
            stop = max(stop, newstop) if long else min(stop, newstop)
        k += 1
    if remaining > 1e-9:
        move += (cl[n - 1] - entry) * sign * remaining
        return (move * USD_PER_PRICE - spread_pip * PIP * USD_PER_PRICE, move / risk, False)
    return (move * USD_PER_PRICE - spread_pip * PIP * USD_PER_PRICE, move / risk, True)


TRAIL_STRATEGIES = {
    "full trail 50pip":        (0.0, 50.0),
    "full trail 100pip":       (0.0, 100.0),
    "50% TP1 + 50% trail50":   (0.5, 50.0),
    "50% TP1 + 50% trail100":  (0.5, 100.0),
}


STRATEGIES = {
    "TP1 full (now)":      (lambda: [(1.0, 1)], False),
    "TP2 full":            (lambda: [(1.0, 2)], False),
    "TP3 full":            (lambda: [(1.0, 3)], False),
    "TP4 full":            (lambda: [(1.0, 4)], False),
    "50% TP1 / 50% TP2 +BE": (lambda: [(0.5, 1), (0.5, 2)], True),
    "50% TP1 / 50% TP3 +BE (now)": (lambda: [(0.5, 1), (0.5, 3)], True),
    "50% TP1 / 50% TP4 +BE": (lambda: [(0.5, 1), (0.5, 4)], True),
    "1/3 TP1/TP2/TP3 +BE": (lambda: [(1/3, 1), (1/3, 2), (1/3, 3)], True),
    "1/4 TP1/2/3/4 +BE":   (lambda: [(0.25, 1), (0.25, 2), (0.25, 3), (0.25, 4)], True),
}


def main():
    expiry_min, spread_pip = 240, 3.0
    sigs = [json.loads(l) for l in open("data/ug/signals.jsonl", encoding="utf-8") if l.strip()]
    m1 = load_m1()
    off = detect_offset(sigs, m1)
    print(f"# tz offset UTC{off:+d} | cost {spread_pip}pip | {len(sigs)} signals | exit-strategy compare\n")

    for tp1_bucket in (50.0, 100.0, 150.0):
        bucket = [s for s in sigs if s.get("tps_pip", {}).get("1") == tp1_bucket]
        fills = [f for f in (fill_index(s, m1, off, expiry_min) for s in bucket) if f]
        print(f"=== TP1 {int(tp1_bucket)}pip bucket: {len(bucket)} signals, {len(fills)} filled ===")
        if not fills:
            print("  (no fills)\n"); continue
        print(f"  {'strategy':<24}{'net $':>10}{'meanR':>9}{'avg$/trade':>12}{'unresolved':>12}")
        def row(name, outs):
            usd = sum(o[0] for o in outs)
            meanR = sum(o[1] for o in outs) / len(outs)
            unres = sum(1 for o in outs if not o[2])
            print(f"  {name:<24}{usd:>+10.2f}{meanR:>+9.3f}{usd/len(outs):>+12.2f}{unres:>12}")
        for name, (legf, be) in STRATEGIES.items():
            row(name, [resolve(m1, f, legf(), be, spread_pip) for f in fills])
        for name, (tp1f, trail) in TRAIL_STRATEGIES.items():
            row(name, [resolve_trail(m1, f, tp1f, trail, spread_pip) for f in fills])
        # runner stop: break-even (entry) vs lock-at-TP1
        for tgt in (3, 4):
            legs = [(0.5, 1), (0.5, tgt)]
            row(f"50%TP1/50%TP{tgt} SL->entry",
                [resolve(m1, f, list(legs), True, spread_pip, be_to="entry") for f in fills])
            row(f"50%TP1/50%TP{tgt} SL->TP1",
                [resolve(m1, f, list(legs), True, spread_pip, be_to="tp1") for f in fills])
        print()


if __name__ == "__main__":
    sys.exit(main())
