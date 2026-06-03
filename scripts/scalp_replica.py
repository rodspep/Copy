"""Faithful replica of UG's SCALP (PP2) method per admin #677 — to compare TOTAL profit +
durability against the SMC method (smc_replica.py) on the SAME 17 months, FAIRLY.

Decoded spec (#677): EMA 20/34/50/89. "Trend → pullback chạm EMA20, vùng EMA34-50, DCA 50-89;
ăn 5 giá xong té." So in a trend, on a pull-back into the EMA cloud (EMA34..EMA89), enter in
the trend direction; the DCA-to-EMA89 means the effective entry sits in the EMA50-89 zone and
the natural stop is just beyond EMA89; take +5 price (50pip). Best in sideway/normal vol.

Compares two exits: (a) UG's "ăn 5 giá té" = full +50pip; (b) our live unified TP1@50+runner@150+BE.
Backtest on 17 months M5, conservative bar model, anti-lookahead (EMAs are causal).
Run: python -X utf8 -m scripts.scalp_replica
"""
from __future__ import annotations

import collections
import numpy as np
import pandas as pd

PIP = 0.1
USD = 1.0
COST = 0.30
TP_PX = 5.0              # +50 pip = "ăn 5 giá"
RUNNER_PX = 15.0         # +150 pip runner (our unified exit)
SL_BUF = 1.0            # stop just beyond EMA89 (cloud far edge)
HORIZON = 36            # M5 bars (~3h) to resolve


def load():
    d = pd.read_csv("data/xau/XAUUSD_M5.csv")
    d["time"] = pd.to_datetime(d["timestamp"], utc=True, format="ISO8601").dt.tz_localize(None)
    d = d[["time", "open", "high", "low", "close"]].sort_values("time").reset_index(drop=True)
    for n in (20, 34, 50, 89):
        d[f"e{n}"] = d["close"].ewm(span=n, adjust=False).mean()
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(),
                    (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    return d


def signals(d):
    """Trend-pullback into the EMA cloud. Trend = EMA20/34/50/89 stacked; pullback = price
    trades back into [EMA34, EMA89]; enter trend direction at the cloud, SL beyond EMA89."""
    out = []
    last = -999
    e20, e34, e50, e89 = d.e20.values, d.e34.values, d.e50.values, d.e89.values
    h, l, c, atr = d.high.values, d.low.values, d.close.values, d.atr.values
    for i in range(90, len(d) - 1):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        up = e20[i] > e34[i] > e50[i] > e89[i]
        dn = e20[i] < e34[i] < e50[i] < e89[i]
        if not (up or dn):
            continue                                   # need a clean trend stack
        sign = 1 if up else -1
        lo_cloud, hi_cloud = (e89[i], e34[i]) if up else (e34[i], e89[i])
        # pullback INTO the cloud this bar (trend up: price dipped to between e89..e34)
        if up:
            touched = l[i] <= e34[i] and l[i] >= e89[i] - SL_BUF
        else:
            touched = h[i] >= e34[i] and h[i] <= e89[i] + SL_BUF
        if not touched:
            continue
        if i - last < 6:                               # cooldown
            continue
        entry = e50[i]                                 # DCA-averaged entry ~ EMA50 (mid of 50-89 zone)
        sl = (e89[i] - SL_BUF) if up else (e89[i] + SL_BUF)
        if (entry - sl) * sign <= 0:
            continue
        out.append({"i": i, "ts": d.time.iloc[i], "sign": sign, "entry": entry, "sl": sl})
        last = i
    return out


def run(d, sigs, exit_mode):
    h, l, c, n = d.high.values, d.low.values, d.close.values, len(d)
    net = wins = 0
    res = []
    for s in sigs:
        i, sign, entry, sl = s["i"], s["sign"], s["entry"], s["sl"]
        # fill: wait for price to trade to entry within HORIZON
        j = None
        for k in range(i + 1, min(i + 1 + HORIZON, n)):
            if l[k] <= entry <= h[k]:
                j = k
                break
        if j is None:
            continue
        if exit_mode == "scalp5":
            tp = entry + sign * TP_PX
            move = None
            for k in range(j + 1, min(j + 1 + HORIZON, n)):
                if (l[k] <= sl) if sign > 0 else (h[k] >= sl):
                    move = (sl - entry) * sign; break
                if (h[k] >= tp) if sign > 0 else (l[k] <= tp):
                    move = (tp - entry) * sign; break
            if move is None:
                move = (c[min(j + HORIZON, n - 1)] - entry) * sign
            usd = move * USD - COST
        else:                                          # unified TP1@50 + runner@150 + BE
            tp1, tp3 = entry + sign * TP_PX, entry + sign * RUNNER_PX
            stop = sl; rem = 1.0; move = 0.0; booked = False
            for k in range(j + 1, min(j + 1 + HORIZON, n)):
                if (l[k] <= stop) if sign > 0 else (h[k] >= stop):
                    move += (stop - entry) * sign * rem; rem = 0; break
                if not booked and ((h[k] >= tp1) if sign > 0 else (l[k] <= tp1)):
                    move += (tp1 - entry) * sign * 0.5; rem -= 0.5; booked = True; stop = entry
                if booked and rem > 0 and ((h[k] >= tp3) if sign > 0 else (l[k] <= tp3)):
                    move += (tp3 - entry) * sign * rem; rem = 0; break
            if rem > 0:
                move += (c[min(j + HORIZON, n - 1)] - entry) * sign * rem
            usd = move * USD - COST
        net += usd; wins += usd > 0
        res.append({"ts": s["ts"], "usd": usd})
    return net, wins, res


def main():
    d = load()
    sigs = signals(d)
    days = (d.time.max() - d.time.min()).days or 1
    print(f"# SCALP replica (faithful EMA cloud, #677) on M5 {d.time.min().date()}→{d.time.max().date()} "
          f"({days}d) — {len(sigs)} setups ({len(sigs)/days:.1f}/day)\n")
    for mode, label in (("scalp5", "UG 'ăn 5 giá té' (full +50pip)"),
                        ("unified", "our unified (TP1@50 + runner@150 + BE)")):
        net, wins, res = run(d, sigs, mode)
        if not res:
            print(f"  {label}: no fills"); continue
        wr = wins / len(res) * 100
        print(f"  {label:<42} fills={len(res):<4} net ${net:>+8.2f} $/trade {net/len(res):>+5.2f} "
              f"WR {wr:>3.0f}%  ${net/days:>+5.2f}/day")
        bym = collections.defaultdict(float); cm = collections.Counter()
        for r in res:
            bym[r["ts"].strftime("%Y-%m")] += r["usd"]; cm[r["ts"].strftime("%Y-%m")] += 1
        posm = sum(1 for v in bym.values() if v > 0)
        print(f"      durability: {posm}/{len(bym)} months positive")
    print("\n  COMPARE vs SMC replica: SMC ~$2.24/trade, ~1/day (~$2/day), WR 30%, 13/18 mo +.")
    print("  Decide on TOTAL $/day AND durability — not WR alone.")


if __name__ == "__main__":
    raise SystemExit(main())
