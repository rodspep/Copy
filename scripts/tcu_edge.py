"""EDGE analysis of UG signals UNDER OUR COPIER'S EXIT, on real M1.

Codex's #1 priority: don't try to recover UG's (LLM-vote) entry formula — instead measure,
PER METHOD/family and PER MONTH, what OUR copier exit actually earns on UG's signals. We
replay each signal from the Trade Coin Underground public API (entry zone + SL + direction +
displayName + tpPips, with labeled outcomes) through our exact exit:
    TP1 leg @ +50pip + runner leg @ +150pip, SL→BE after the TP1 leg books,
on real XAUUSD M1 (conservative bar model: SL-first each bar, TP from the bar AFTER fill).
Zone-aware fill mirrors the live copier (market at/better than anchor, else limit-at-anchor).

This validates the claimed ~85-89% TP1-touch out-of-sample and shows whether the scalp edge
is stable or decaying month-to-month. Read-only. Run: python -X utf8 -m scripts.tcu_edge
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

PIP = 0.1
USD_PER_PRICE = 1.0          # 0.01 lot XAU
EXPIRY_MIN = 120
COST = 3 * PIP * USD_PER_PRICE   # ~3pip round-trip per leg


def load_m1():
    a = pd.read_csv("data/xau/XAUUSD_M1.csv").rename(columns={"timestamp": "time"})
    b = pd.read_csv("data/xau/XAUUSD_M1_recent.csv")
    m = pd.concat([a[["time", "open", "high", "low", "close"]],
                   b[["time", "open", "high", "low", "close"]]])
    m["time"] = pd.to_datetime(m["time"], utc=True, format="ISO8601").dt.tz_localize(None)
    return m.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def load_signals():
    sigs = json.load(open("data/ug/tcu/signals_full.json", encoding="utf-8"))["signals"]
    out = []
    for s in sigs:
        er, sl, ct = s.get("entryRange"), s.get("stopLoss"), s.get("createdAt")
        if not er or sl is None or not ct or len(er) != 2:
            continue
        d = s.get("direction", "").upper()
        direction = "long" if d in ("LONG", "BUY") else "short" if d in ("SHORT", "SELL") else None
        if direction is None:
            continue
        tp = (s.get("tpPips") or [None])[0]
        out.append({
            "ts": pd.Timestamp(ct).tz_convert("UTC").tz_localize(None),
            "direction": direction, "lo": float(min(er)), "hi": float(max(er)),
            "sl": float(sl), "tp1_pip": tp, "name": s.get("displayName") or "?",
            "status": s.get("status"), "mfp": s.get("maxFavorablePrice"),
            "entry_api": s.get("entry"),
        })
    return out


def fill(s, m, t, lo_a, hi_a, cl):
    """Zone-aware fill like the live copier → (entry, sign, fill_idx) or None."""
    zlo, zhi = s["lo"], s["hi"]
    mid = (zlo + zhi) / 2.0
    long = s["direction"] == "long"
    sign = 1 if long else -1
    i0 = t.searchsorted(np.datetime64(s["ts"]))
    if not (0 < i0 < len(m)):
        return None
    px0 = cl[i0 - 1]
    if long:
        if px0 < zlo:
            return None                       # voided (below buy zone)
        if px0 <= mid:
            return (px0, sign, i0)            # market
        entry = mid
    else:
        if px0 > zhi:
            return None
        if px0 >= mid:
            return (px0, sign, i0)
        entry = mid
    end = s["ts"] + pd.Timedelta(minutes=EXPIRY_MIN)
    j = i0
    while j < len(m) and t[j] <= np.datetime64(end):
        if (lo_a[j] <= entry) if long else (hi_a[j] >= entry):
            return (entry, sign, j)
        j += 1
    return None


def our_exit(entry, sign, j, sl0, m, hi_a, lo_a, cl, tp1_pip=50, tp3_pip=150):
    """Replay the live exit: 50% TP1@+tp1, 50% runner@+tp3, SL→BE after TP1. SL-first per bar,
    TP from the bar AFTER fill, same-bar BE re-checked. Returns (usd, reach50, reach150, straightSL)."""
    tp1 = entry + sign * tp1_pip * PIP
    tp3 = entry + sign * tp3_pip * PIP
    stop = sl0
    remaining = 1.0
    move = 0.0
    booked = False
    mfe = 0.0
    k = j + 1
    while k < len(m) and remaining > 1e-9:
        lo_b, hi_b = lo_a[k], hi_a[k]
        fav = (hi_b - entry) if sign > 0 else (entry - lo_b)
        if fav > mfe:
            mfe = fav
        adverse = (lo_b <= stop) if sign > 0 else (hi_b >= stop)
        if adverse:
            move += (stop - entry) * sign * remaining
            remaining = 0.0
            break
        if not booked and ((hi_b >= tp1) if sign > 0 else (lo_b <= tp1)):
            move += (tp1 - entry) * sign * 0.5
            remaining -= 0.5
            booked = True
            stop = entry                       # SL → BE
            if (lo_b <= stop) if sign > 0 else (hi_b >= stop):   # same-bar BE
                move += (stop - entry) * sign * remaining
                remaining = 0.0
                break
        if booked and remaining > 1e-9 and ((hi_b >= tp3) if sign > 0 else (lo_b <= tp3)):
            move += (tp3 - entry) * sign * remaining
            remaining = 0.0
            break
        k += 1
    if remaining > 1e-9:
        move += (cl[len(m) - 1] - entry) * sign * remaining     # mark out at last close
    usd = move * USD_PER_PRICE - COST
    reach50 = mfe >= 50 * PIP
    reach150 = mfe >= 150 * PIP
    return usd, reach50, reach150, (mfe < 50 * PIP)


def main():
    m = load_m1()
    sigs = load_signals()
    t = m["time"].values
    hi_a, lo_a, cl = m["high"].values, m["low"].values, m["close"].values
    mmax = m["time"].max()

    # price-alignment sanity: |entry_api - M1 close at ts| should be small if ts/prices align
    diffs = []
    for s in sigs:
        i = t.searchsorted(np.datetime64(s["ts"]))
        if 0 < i < len(cl) and s["entry_api"]:
            diffs.append(abs(cl[i - 1] - s["entry_api"]))
    print(f"# {len(sigs)} signals w/ zone+SL | M1 ends {mmax}")
    print(f"# price-alignment |entry-M1close| median {np.median(diffs):.2f} price "
          f"(small = ts/prices aligned, no offset)\n")

    results = []
    for s in sigs:
        if s["ts"] > mmax:                      # outside M1 coverage (newest signals)
            continue
        fb = fill(s, m, t, lo_a, hi_a, cl)
        if fb is None:
            results.append({**s, "filled": False})
            continue
        entry, sign, j = fb
        usd, r50, r150, straightSL = our_exit(entry, sign, j, s["sl"], m, hi_a, lo_a, cl)
        results.append({**s, "filled": True, "usd": usd, "r50": r50, "r150": r150,
                        "straightSL": straightSL})

    def agg(rows, label):
        # Per Codex: report per-RECEIVED (unfilled = $0) AND per-filled, plus fill rate, so a
        # family that merely fills more selectively isn't mistaken for a better edge.
        recv = len(rows)
        f = [r for r in rows if r.get("filled")]
        if recv == 0:
            print(f"  {label:<30} (none)")
            return
        net = sum(r["usd"] for r in f)
        wins = sum(1 for r in f if r["usd"] > 0)
        fillpct = len(f) / recv * 100
        per_recv = net / recv
        per_fill = (net / len(f)) if f else 0.0
        wr = (wins / len(f) * 100) if f else 0.0
        r50 = (sum(1 for r in f if r["r50"]) / len(f) * 100) if f else 0.0
        print(f"  {label:<26} recv={recv:<4} fill={len(f):<4}({fillpct:>3.0f}%) "
              f"net ${net:>+8.2f}  $/recv {per_recv:>+5.2f}  $/fill {per_fill:>+5.2f}  "
              f"WR {wr:>3.0f}%  reach50 {r50:>3.0f}%")

    incov = [r for r in results if r["ts"] <= mmax]
    print(f"== OUR EXIT (TP1@50+runner@150+BE) on {len(incov)} signals in M1 coverage ==")
    agg(incov, "ALL")
    print("\n  -- by method family (displayName) --")
    for name in sorted(set(r["name"] for r in incov)):
        agg([r for r in incov if r["name"] == name], name)
    print("\n  -- by TP1 template (method id) --")
    for tp in sorted(set(r["tp1_pip"] for r in incov if r["tp1_pip"] is not None)):
        agg([r for r in incov if r["tp1_pip"] == tp], f"TP1={tp}pip")
    print("\n  -- ROLLING by month (is the edge stable?) --")
    for mo in sorted(set(r["ts"].strftime("%Y-%m") for r in incov)):
        agg([r for r in incov if r["ts"].strftime("%Y-%m") == mo], mo)
    print("\n  -- 50pip scalp by month --")
    scalp = [r for r in incov if r["tp1_pip"] == 50]
    for mo in sorted(set(r["ts"].strftime("%Y-%m") for r in scalp)):
        agg([r for r in scalp if r["ts"].strftime("%Y-%m") == mo], f"50pip {mo}")

    # Entry-adjustment + SL-distance audit per family: is a 'better family' just getting a
    # more favourable shifted entry (artifact), not a better signal? (Codex disambiguation.)
    print("\n  -- entry-adjustment / SL-distance audit by family --")
    for name in sorted(set(r["name"] for r in incov)):
        g = [r for r in incov if r["name"] == name]
        adj = []
        for r in g:
            i = t.searchsorted(np.datetime64(r["ts"]))
            if 0 < i < len(cl) and r["entry_api"]:
                mid = (r["lo"] + r["hi"]) / 2
                # signed adjustment in pip: how far the entry zone sits from market at post,
                # in the FILL (pullback) direction (+ = needs a deeper pull-back to fill)
                sign = 1 if r["direction"] == "long" else -1
                adj.append(((cl[i - 1] - mid) * sign) / PIP)
        sld = [abs(r["lo"] - r["sl"]) for r in g]      # SL distance (price) ~ zone edge to SL
        import statistics as st_
        amed = st_.median(adj) if adj else float("nan")
        smed = st_.median(sld) if sld else float("nan")
        print(f"  {name:<26} entry-adj med {amed:>+5.0f}pip  SL-dist med {smed:>4.1f} price  (n={len(g)})")


if __name__ == "__main__":
    raise SystemExit(main())
