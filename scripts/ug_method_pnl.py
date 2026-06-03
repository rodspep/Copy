"""Per-method P/L of ALL collected UG signals, simulated on real M1 XAU.

Models the copier's actual behaviour as faithfully as the data allows:
  - entry = the copier's mode for that TP1 bucket (50->mid, 100->near, 150->deep)
  - SKIP if at signal time price is already at/through TP1 (copier rule)
  - place LIMIT at entry; fill only if price trades into it within `expiry_min`
  - cancel if price reaches TP1 before filling ("don't chase")
  - after fill, resolve TP1-vs-SL on M1; SAME-bar ambiguity = SL first (conservative)
  - cost: `spread_pip` charged against the exit (XAU spread eats tight TPs)
  - 0.01 lot => $1 per 1.0 price; pnl_$ = signed price move * 1.0 - cost

Buckets by TP1 pip (the differentiator the copier filters on) and cross-tabs by
UG's PP label (method==2 vs not). In-sample, ~1 week — directional, not gospel.
"""
from __future__ import annotations

import json
import sys
import pandas as pd

PIP = 0.1
LOT = 0.01
USD_PER_PRICE = 100 * LOT           # XAU: 1 lot=100oz -> $100/price; 0.01 lot -> $1/price
MODE_BY_TP1 = {50.0: "mid", 100.0: "mid", 150.0: "mid"}   # matches live copier (all MID)


def load_m1() -> pd.DataFrame:
    df = pd.read_csv("data/xau/XAUUSD_M1.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def detect_offset(sigs, m1) -> int:
    """M1 csv is in broker time; signals in UTC. Pick the hour offset that makes the
    M1 close at signal-time sit closest to the signal's entry zone (signals fade the
    live price, so entry ~ current price)."""
    t = m1["timestamp"].values
    c = m1["close"].values
    best, best_err = 0, 1e9
    for off in range(-6, 7):
        errs = []
        for s in sigs:
            ts = pd.Timestamp(s["ts"]).tz_localize(None) + pd.Timedelta(hours=off)
            i = t.searchsorted(ts.to_datetime64())
            if 0 < i < len(c):
                mid = (s["entry_low"] + s["entry_high"]) / 2
                errs.append(abs(c[i - 1] - mid))
        if errs:
            med = sorted(errs)[len(errs) // 2]
            if med < best_err:
                best_err, best = med, off
    print(f"# tz offset detected: broker = UTC{best:+d}h (median |close-entry| {best_err:.2f})")
    return best


def simulate(s, m1, off, expiry_min, spread_pip, mode_override=None, chase_rule=True):
    tp1 = s.get("tps_pip", {}).get("1")
    if tp1 not in MODE_BY_TP1:
        return {"status": "filtered_tp1"}
    lo, hi, sl = s["entry_low"], s["entry_high"], s["sl"]
    long = s["direction"] == "long"
    mode = mode_override or MODE_BY_TP1[tp1]
    entry = {"near": lo, "mid": (lo + hi) / 2, "deep": hi}[mode]
    sign = 1 if long else -1
    tp = entry + sign * tp1 * PIP
    cost = spread_pip * PIP

    ts = pd.Timestamp(s["ts"]).tz_localize(None) + pd.Timedelta(hours=off)
    t = m1["timestamp"].values
    i0 = t.searchsorted(ts.to_datetime64())
    if i0 <= 0 or i0 >= len(m1):
        return {"status": "no_data"}
    px0 = m1["close"].values[i0 - 1]                  # price at signal time

    # Placement rule:
    #  chase_rule=True  (current copier): require price BETWEEN entry and TP1 (skip if
    #                   past TP1 — 'don't chase'), and cancel if TP1 hit before fill.
    #  chase_rule=False (deep-limit): only require price on the fillable side of entry
    #                   (price>entry for a buy-limit); place even if past TP1 and wait
    #                   for the pull-back to entry. No cancel-on-TP-first.
    if chase_rule:
        ok = (entry < px0 < tp) if long else (tp < px0 < entry)
    else:
        ok = (px0 > entry) if long else (px0 < entry)
    if not ok:
        return {"status": "skip_no_room"}

    hi_a, lo_a, tm = m1["high"].values, m1["low"].values, m1["timestamp"].values
    end = ts + pd.Timedelta(minutes=expiry_min)
    filled = False
    j = i0
    # 1) wait for fill (or cancel: TP1 reached first [chase only], or expiry)
    while j < len(m1) and tm[j] <= end.to_datetime64():
        if long:
            if chase_rule and hi_a[j] >= tp:
                return {"status": "cancel_tp_first", "tp1": tp1, "mode": mode}
            if lo_a[j] <= entry:
                filled = True; break
        else:
            if chase_rule and lo_a[j] <= tp:
                return {"status": "cancel_tp_first", "tp1": tp1, "mode": mode}
            if hi_a[j] >= entry:
                filled = True; break
        j += 1
    if not filled:
        return {"status": "expired_nofill", "tp1": tp1, "mode": mode}

    # 2) resolve TP1 vs SL. On the FILL bar, only SL is allowed (conservative): the
    #    bar's favourable extreme may have occurred BEFORE the limit filled, so a
    #    same-bar TP would be optimistic (Codex). TP becomes eligible from j+1.
    def _loss():
        move = (sl - entry) * sign
        return {"status": "loss", "tp1": tp1, "mode": mode, "R": move / abs(entry - sl),
                "usd": move * USD_PER_PRICE - cost * USD_PER_PRICE}

    def _win():
        move = (tp - entry) * sign
        return {"status": "win", "tp1": tp1, "mode": mode, "R": move / abs(entry - sl),
                "usd": move * USD_PER_PRICE - cost * USD_PER_PRICE}

    if (lo_a[j] <= sl) if long else (hi_a[j] >= sl):     # SL on the fill bar
        return _loss()
    k = j + 1
    while k < len(m1):
        hit_sl = lo_a[k] <= sl if long else hi_a[k] >= sl
        hit_tp = hi_a[k] >= tp if long else lo_a[k] <= tp
        if hit_sl:
            return _loss()
        if hit_tp:
            return _win()
        k += 1
    return {"status": "unresolved", "tp1": tp1, "mode": mode}    # ran out of data


def main():
    expiry_min = 240
    spread_pip = 3.0          # ~2pip spread + ~1pip slippage on XAU
    sigs = [json.loads(l) for l in open("data/ug/signals.jsonl", encoding="utf-8") if l.strip()]
    m1 = load_m1()
    off = detect_offset(sigs, m1)

    res = []
    for s in sigs:
        r = simulate(s, m1, off, expiry_min, spread_pip)
        r["pp2"] = (s.get("method") == 2)
        res.append(r)

    def agg(rows, label):
        wins = [r for r in rows if r["status"] == "win"]
        loss = [r for r in rows if r["status"] == "loss"]
        traded = wins + loss
        usd = sum(r["usd"] for r in traded)
        wr = len(wins) / len(traded) * 100 if traded else 0
        meanR = sum(r["R"] for r in traded) / len(traded) if traded else 0
        print(f"{label:<14} signals {len(rows):>3} | filled&closed {len(traded):>3} "
              f"| win {len(wins):>3} loss {len(loss):>3} | WR {wr:5.1f}% "
              f"| meanR {meanR:+.3f} | net ${usd:+8.2f}")

    print(f"\n=== UG signals backtest (M1, limit+confirm-room, expiry {expiry_min}m, "
          f"cost {spread_pip}pip, 0.01 lot) ===")
    print(f"total signals: {len(sigs)}")
    # status breakdown
    import collections
    st = collections.Counter(r["status"] for r in res)
    print("status:", dict(st))
    print("\n--- by TP1 bucket (copier's method key) ---")
    for tp1 in (50.0, 100.0, 150.0):
        agg([r for r in res if r.get("tp1") == tp1], f"TP1 {int(tp1)}pip")
    print("\n--- by UG PP label ---")
    agg([r for r in res if r["pp2"]], "PP2 (method=2)")
    agg([r for r in res if not r["pp2"]], "PP1/other")
    print("\n--- ALL traded ---")
    agg(res, "ALL")

    # ENTRY-MODE SWEEP: for each TP1 bucket try near/mid/deep, report fill + net.
    print("\n=== ENTRY-MODE SWEEP per TP1 bucket (which entry fills + profits best) ===")
    for tp1 in (50.0, 100.0, 150.0):
        bucket = [s for s in sigs if s.get("tps_pip", {}).get("1") == tp1]
        print(f"\nTP1 {int(tp1)}pip  (n={len(bucket)} signals)")
        print(f"  {'mode':<6}{'closed':>8}{'fill%':>8}{'WR':>8}{'meanR':>9}{'net $':>10}")
        stats = {}
        for mode in ("near", "mid", "deep"):
            rows = [simulate(s, m1, off, expiry_min, spread_pip, mode_override=mode) for s in bucket]
            wins = [r for r in rows if r["status"] == "win"]
            traded = wins + [r for r in rows if r["status"] == "loss"]
            usd = sum(r["usd"] for r in traded)
            stats[mode] = (len(traded), len(traded) / len(bucket) * 100 if bucket else 0,
                           len(wins) / len(traded) * 100 if traded else 0,
                           sum(r["R"] for r in traded) / len(traded) if traded else 0, usd)
        best = max(stats, key=lambda m: stats[m][4]) if bucket else None
        for mode in ("near", "mid", "deep"):
            n, fp, wr, mr, usd = stats[mode]
            tag = "  <= best $" if mode == best and n else ""
            print(f"  {mode:<6}{n:>8}{fp:>7.0f}%{wr:>7.1f}%{mr:>+9.3f}{usd:>+10.2f}{tag}")

    # CHASE-RULE vs DEEP-LIMIT: does dropping 'skip if past TP1' (treat as a deep
    # pull-back limit) help any bucket? (entry = MID, the live copier's choice)
    print("\n=== CHASE-RULE vs DEEP-LIMIT per bucket (entry=mid) ===")
    for tp1 in (50.0, 100.0, 150.0):
        bucket = [s for s in sigs if s.get("tps_pip", {}).get("1") == tp1]
        print(f"\nTP1 {int(tp1)}pip  (n={len(bucket)})")
        print(f"  {'rule':<12}{'closed':>8}{'fill%':>8}{'WR':>8}{'meanR':>9}{'net $':>10}")
        for label, chase in (("chase (now)", True), ("deep-limit", False)):
            rows = [simulate(s, m1, off, expiry_min, spread_pip, mode_override="mid",
                             chase_rule=chase) for s in bucket]
            wins = [r for r in rows if r["status"] == "win"]
            traded = wins + [r for r in rows if r["status"] == "loss"]
            usd = sum(r["usd"] for r in traded)
            wr = len(wins) / len(traded) * 100 if traded else 0
            mr = sum(r["R"] for r in traded) / len(traded) if traded else 0
            fp = len(traded) / len(bucket) * 100 if bucket else 0
            print(f"  {label:<12}{len(traded):>8}{fp:>7.0f}%{wr:>7.1f}%{mr:>+9.3f}{usd:>+10.2f}")


if __name__ == "__main__":
    sys.exit(main())
