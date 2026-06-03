"""WHY do the 100 (PRI GOLD) and 150 (Ai Signals) methods lose? Data-grounded, not guesses.

For each such signal, fill it like the live copier (zone-aware: market at/better than the
anchor, else limit-at-anchor waiting for the pull-back), then on real M1 measure:
  - outcome at the method's OWN TP1, and the MAX FAVOURABLE EXCURSION (MFE) before close
    (how far price moved in our favour before TP/SL) — tells us if the entry/direction
    was right but the TP was too far;
  - the win-rate the SAME entries would get at SHORTER TP distances (50/100/150/200/300
    pip) — if short TPs win where the far TP loses, the entry is good and the TP is the
    problem (not a bad-zone entry);
  - the 'immediate adverse' rate: trades whose MFE is ~0 (price went straight to SL) =
    genuinely bad entry/direction.
Conservative bar model: SL checked before TP each bar; TP from the bar AFTER fill.
"""
from __future__ import annotations

import json
import pandas as pd

PIP = 0.1
USD_PER_PRICE = 1.0          # 0.01 lot XAU
EXPIRY = 120


def load_m1():
    df = pd.read_csv("data/xau/XAUUSD_M1.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def detect_offset(sigs, m1):
    t, c = m1["timestamp"].values, m1["close"].values
    best, be = 0, 1e9
    for off in range(-6, 7):
        e = []
        for s in sigs:
            ts = pd.Timestamp(s["ts"]).tz_localize(None) + pd.Timedelta(hours=off)
            i = t.searchsorted(ts.to_datetime64())
            if 0 < i < len(c):
                e.append(abs(c[i - 1] - (s["entry_low"] + s["entry_high"]) / 2))
        if e:
            m = sorted(e)[len(e) // 2]
            if m < be:
                be, best = m, off
    return best


def fill_bar(s, m1, off):
    """Zone-aware fill → (entry, sign, fill_index) or None (skip/voided/no-fill)."""
    lo, hi, sl = s["entry_low"], s["entry_high"], s["sl"]
    zlo, zhi = min(lo, hi), max(lo, hi)
    mid = (lo + hi) / 2
    long = s["direction"] == "long"
    sign = 1 if long else -1
    t = m1["timestamp"].values
    ts = pd.Timestamp(s["ts"]).tz_localize(None) + pd.Timedelta(hours=off)
    i0 = t.searchsorted(ts.to_datetime64())
    if not (0 < i0 < len(m1)):
        return None
    px0 = m1["close"].values[i0 - 1]
    if long:
        if px0 < zlo:
            return None                       # voided
        if px0 <= mid:
            return (px0, sign, i0)            # market
        entry = mid                            # limit, wait for pull-back down
    else:
        if px0 > zhi:
            return None
        if px0 >= mid:
            return (px0, sign, i0)
        entry = mid
    lo_a, hi_a, tm = m1["low"].values, m1["high"].values, t
    end = ts + pd.Timedelta(minutes=EXPIRY)
    j = i0
    while j < len(m1) and tm[j] <= end.to_datetime64():
        if (lo_a[j] <= entry) if long else (hi_a[j] >= entry):
            return (entry, sign, j)
        j += 1
    return None


def analyze(rows, m1, off, label):
    hi_a, lo_a = m1["high"].values, m1["low"].values
    tps = [50, 100, 150, 200, 300]
    COST = 3 * PIP * USD_PER_PRICE
    win_at = {tp: 0 for tp in tps}
    net_at = {tp: 0.0 for tp in tps}
    n_fill = 0
    mfes = []
    immediate_adverse = 0
    for s in rows:
        fb = fill_bar(s, m1, off)
        if fb is None:
            continue
        entry, sign, j = fb
        sl = s["sl"]
        risk = abs(entry - sl)                 # price distance mid→SL ($ at 0.01 lot)
        n_fill += 1
        mfe = 0.0
        reached = set()
        k = j + 1
        hit_sl = False
        while k < len(m1):
            adverse = (lo_a[k] <= sl) if sign > 0 else (hi_a[k] >= sl)
            fav = (hi_a[k] - entry) if sign > 0 else (entry - lo_a[k])
            if adverse:
                hit_sl = True
                break
            if fav > mfe:
                mfe = fav
            for tp in tps:
                if fav >= tp * PIP:
                    reached.add(tp)
            k += 1
        mfes.append(mfe)
        if mfe < 1.0:
            immediate_adverse += 1
        for tp in tps:
            if tp in reached:                  # TP reached before SL → win
                win_at[tp] += 1
                net_at[tp] += tp * PIP * USD_PER_PRICE - COST
            elif hit_sl:                       # SL hit before this TP → loss at real SL
                net_at[tp] += -risk * USD_PER_PRICE - COST
            # else unresolved (ran out of data) → not counted
    print(f"=== {label}: {len(rows)} signals, {n_fill} filled ===")
    if n_fill:
        mm = sorted(mfes)
        print(f"  MFE before close: median {mm[len(mm)//2]:.1f} price ({mm[len(mm)//2]/PIP:.0f}pip), "
              f"max {max(mfes):.1f} price")
        print(f"  'immediate adverse' (MFE<10pip → straight to SL = bad entry): "
              f"{immediate_adverse}/{n_fill} = {immediate_adverse/n_fill*100:.0f}%")
        print(f"  {'TP exit':<16}{'WR':>7}{'net $':>10}")
        for tp in tps:
            print(f"     {tp:>3}pip ({tp*PIP:>4.0f} price){win_at[tp]/n_fill*100:>6.0f}%{net_at[tp]:>+10.2f}")
    print()


def bracket_net(rows, m1, off, tp1_pip=50, tp3_pip=150, label=""):
    """FIXED bracket (method-independent): 50% at +tp1_pip, 50% runner at +tp3_pip with
    SL→BE after the TP1 leg books. Conservative: SL-first each bar; TP from the bar AFTER
    fill; same-bar BE re-checked. Returns (n, wr_signal, net$)."""
    hi_a, lo_a, cl = m1["high"].values, m1["low"].values, m1["close"].values
    COST = 3 * PIP * USD_PER_PRICE
    n = 0; net = 0.0; wins = 0
    for s in rows:
        fb = fill_bar(s, m1, off)
        if fb is None:
            continue
        entry, sign, j = fb
        sl0 = s["sl"]
        tp1 = entry + sign * tp1_pip * PIP
        tp3 = entry + sign * tp3_pip * PIP
        n += 1
        stop = sl0
        remaining = 1.0
        move = 0.0
        booked = False
        k = j + 1
        while k < len(m1) and remaining > 1e-9:
            lo_b, hi_b = lo_a[k], hi_a[k]
            adverse = (lo_b <= stop) if sign > 0 else (hi_b >= stop)
            if adverse:
                move += (stop - entry) * sign * remaining
                remaining = 0.0
                break
            if not booked:                      # book the 50% TP1 leg
                if (hi_b >= tp1) if sign > 0 else (lo_b <= tp1):
                    move += (tp1 - entry) * sign * 0.5
                    remaining -= 0.5
                    booked = True
                    stop = entry                # BE
                    if (lo_b <= stop) if sign > 0 else (hi_b >= stop):   # same-bar BE
                        move += (stop - entry) * sign * remaining
                        remaining = 0.0
                        break
            if booked and remaining > 1e-9:     # runner to TP3
                if (hi_b >= tp3) if sign > 0 else (lo_b <= tp3):
                    move += (tp3 - entry) * sign * remaining
                    remaining = 0.0
                    break
            k += 1
        if remaining > 1e-9:
            move += (cl[len(m1) - 1] - entry) * sign * remaining   # mark out at last close
        usd = move * USD_PER_PRICE - COST
        net += usd
        if usd > 0:
            wins += 1
    wr = round(wins / n * 100, 1) if n else 0
    print(f"  {label:<26} n={n:<3} WR {wr:>5}% net ${net:+.2f}")
    return n, wr, net


def main():
    rows = [json.loads(l) for l in open("data/ug/signals.jsonl", encoding="utf-8") if l.strip()]
    m1 = load_m1()
    off = detect_offset(rows, m1)
    print(f"# tz UTC{off:+d} | zone-aware fill | MFE = max favourable move before SL\n")
    groups = {150.0: "Ai Signals (TP1=150)", 100.0: "PRI GOLD (TP1=100)",
              50.0: "Scalp (TP1=50)"}
    for tp1, lbl in groups.items():
        analyze([s for s in rows if (s.get("tps_pip") or {}).get("1") == tp1], m1, off, lbl)

    print("=== PROPOSED unified exit: 50% TP1@50pip + 50% runner@150pip + SL→BE ===")
    print("    (FIXED distances, ignoring each method's published far TP)\n")
    for tp1, lbl in groups.items():
        bracket_net([s for s in rows if (s.get("tps_pip") or {}).get("1") == tp1],
                    m1, off, 50, 150, lbl)
    print()
    print("  current live (scalp uses signal TP1=50 + TP3=150; 100/150 observe-only)")


if __name__ == "__main__":
    raise SystemExit(main())
