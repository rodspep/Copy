"""Replicate UG's SMC method ("Thợ Săn SMC", decoded from admin-shared signals #654/#687/#688
+ infographic #655) as our own rule-based generator — this is the method with the REAL edge
(R:R 3:1+ on tight OB stops + structure), unlike the scalp's bad 0.5R geometry.

Decoded spec:
  setup (bullish; bearish mirrors): in a down-leg, price SWEEPS sell-side liquidity (wicks
  below the prior swing low, closes back above) then a CHOCH/BOS (closes above the prior
  swing high). The Order Block = the last down-candle of that leg; entry = retest of the OB
  zone; SL = just beyond the OB; TP1/2/3 at R:R 3/5/8; exit 50%/30%/20%, SL→BE after TP1.

Backtest on M15 (resampled from 17 months of M5), conservative bar model, anti-lookahead
(swings confirmed only on closed bars). Run: python -X utf8 -m scripts.smc_replica
"""
from __future__ import annotations

import numpy as np
import pandas as pd

USD_PER_PRICE = 1.0          # 0.01 lot XAU
COST = 0.30                  # ~3pip round-trip
W = 2                        # fractal half-window (swing confirmed W bars later)
RETEST_BARS = 24             # bars to wait for an OB retest after the CHOCH
HORIZON = 96                 # bars to resolve the trade (~24h on M15)
SL_BUFFER = 0.5              # price beyond the OB edge


def m15():
    d = pd.read_csv("data/xau/XAUUSD_M5.csv")
    d["time"] = pd.to_datetime(d["timestamp"], utc=True, format="ISO8601").dt.tz_localize(None)
    d = d.set_index("time")
    o = d["open"].resample("15min").first()
    h = d["high"].resample("15min").max()
    lo = d["low"].resample("15min").min()
    c = d["close"].resample("15min").last()
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}).dropna().reset_index()


def swings(d):
    """Causal fractal swings: bar i is a swing high if its high is the max of [i-W, i+W];
    it is only CONFIRMED (usable) at i+W. Returns arrays of confirmed swing hi/lo prices
    'as known at each bar' (forward-filled from confirmation time)."""
    h, l, n = d["high"].values, d["low"].values, len(d)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for i in range(W, n - W):
        win_h, win_l = h[i - W:i + W + 1], l[i - W:i + W + 1]
        if h[i] == win_h.max():
            sh[i + W] = h[i]            # known only at i+W (anti-lookahead)
        if l[i] == win_l.min():
            sl[i + W] = l[i]
    # last known swing hi/lo price + its bar index, forward-filled
    last_sh = pd.Series(sh).ffill().values
    last_sl = pd.Series(sl).ffill().values
    return last_sh, last_sl


def backtest():
    d = m15()
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    n = len(d)
    last_sh, last_sl = swings(d)
    trades = []
    i = 2 * W
    used_until = 0
    while i < n - 1:
        if i < used_until or np.isnan(last_sh[i]) or np.isnan(last_sl[i]):
            i += 1
            continue
        swh, swl = last_sh[i], last_sl[i]
        # BULLISH: swept sell-side (recent low < swl) then close back above, AND CHOCH up
        swept_dn = l[max(0, i - 6):i + 1].min() < swl
        swept_up = h[max(0, i - 6):i + 1].max() > swh
        setup = None
        if swept_dn and c[i] > swh and (c[i] > o[i]):
            setup = "long"
        elif swept_up and c[i] < swl and (c[i] < o[i]):
            setup = "short"
        if setup is None:
            i += 1
            continue
        sign = 1 if setup == "long" else -1
        # Order block = extreme of the last opposite candle in the impulse leg (last W+3 bars)
        seg = slice(max(0, i - 6), i + 1)
        if setup == "long":
            ob_edge = l[seg].min()                 # OB low (demand)
            entry = ob_edge + (h[seg].max() - ob_edge) * 0.5   # retest into OB mid
            sl = ob_edge - SL_BUFFER
        else:
            ob_edge = h[seg].max()
            entry = ob_edge - (ob_edge - l[seg].min()) * 0.5
            sl = ob_edge + SL_BUFFER
        R = abs(entry - sl)
        if R <= 0 or R > 25:                       # sane stop distance
            i += 1
            continue
        # wait for retest of entry within RETEST_BARS
        j = None
        for k in range(i + 1, min(i + 1 + RETEST_BARS, n)):
            if l[k] <= entry <= h[k]:
                j = k
                break
        if j is None:
            i += 1
            continue
        # simulate exit: TP 3/5/8 R, 50/30/20%, SL->BE after TP1; SL-first per bar
        tps = [entry + sign * m * R for m in (3, 5, 8)]
        stop = sl
        rem = 1.0
        move = 0.0
        booked = 0
        for k in range(j + 1, min(j + 1 + HORIZON, n)):
            if (l[k] <= stop) if sign > 0 else (h[k] >= stop):
                move += (stop - entry) * sign * rem
                rem = 0.0
                break
            for ti, (tp, frac) in enumerate(zip(tps, (0.5, 0.3, 0.2))):
                if ti < booked:
                    continue
                if (h[k] >= tp) if sign > 0 else (l[k] <= tp):
                    move += (tp - entry) * sign * frac
                    rem -= frac
                    booked = ti + 1
                    if booked == 1:
                        stop = entry               # BE after TP1
            if rem <= 1e-9:
                break
        if rem > 1e-9:
            move += (c[min(j + HORIZON, n - 1)] - entry) * sign * rem
        usd = move * USD_PER_PRICE - COST
        trades.append({"ts": d["time"].iloc[j], "dir": setup, "R": R, "usd": usd,
                       "win": usd > 0, "booked": booked})
        used_until = j + 1                          # no overlapping setups
        i = j + 1
    return d, trades


def main():
    d, tr = backtest()
    days = (d["time"].max() - d["time"].min()).days or 1
    print(f"# SMC replica (Method B) on M15 {d['time'].min().date()}→{d['time'].max().date()} "
          f"({days}d, {len(d)} bars)")
    if not tr:
        print("  no setups"); return
    net = sum(t["usd"] for t in tr)
    wins = sum(t["win"] for t in tr)
    print(f"  setups {len(tr)} ({len(tr)/days*7:.1f}/wk) | net ${net:+.2f} | $/trade {net/len(tr):+.2f} "
          f"| WR {wins/len(tr)*100:.0f}% | medR {np.median([t['R'] for t in tr]):.1f}px")
    import collections
    bk = collections.Counter(t["booked"] for t in tr)
    print(f"  exit reached: TP1+ {bk[1]+bk[2]+bk[3]} / TP2+ {bk[2]+bk[3]} / TP3 {bk[3]} / none(SL/BE) {bk[0]}")
    # monthly
    bym = collections.defaultdict(list)
    for t in tr:
        bym[t["ts"].strftime("%Y-%m")].append(t)
    print("  -- by month --")
    for mo in sorted(bym):
        g = bym[mo]; nt = sum(x["usd"] for x in g); w = sum(x["win"] for x in g)
        print(f"    {mo}: n={len(g):<3} net ${nt:>+7.2f} WR {w/len(g)*100:>3.0f}%")
    print("\n  NOTE: v1 structure detector (approx CHOCH/sweep/OB). R:R 3R+ geometry means even"
          " ~40% WR is profitable — the opposite of the scalp. Refine detector to lift further.")


if __name__ == "__main__":
    raise SystemExit(main())
