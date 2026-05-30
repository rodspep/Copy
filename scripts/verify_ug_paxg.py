"""Verify UG signal outcomes on clean fresh PAXG (Binance) data.

PAXG 5m is a trustworthy gold proxy (0.91 5m-return corr with an independent
GC=F feed) covering up to "now" — unlike HistData XAU which has noisy intrabar
OHLC. We measure, for each signal, whether TP1 is reached before SL.

Two measurements per signal:
  1. Random baseline (control): random entries with the same SL/TP geometry,
     telling us the WR achievable with ZERO directional skill.
  2. Signal outcome: enter at PAXG price at the signal's post time (market),
     measure TP1/TP2/.. before SL. Basis-free (uses PAXG's own price as entry).

Edge = signal WR minus random-baseline WR for the same R:R.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.paxg_loader import load as load_paxg

PIP = 0.10  # gold: 1 pip = $0.10 → 50 pip = $5


@dataclass
class Sig:
    name: str
    post_utc: str          # ISO UTC
    side: int              # +1 buy, -1 sell
    sl_pip: float
    tp_pips: list[float]   # [TP1, TP2, ...]
    channel: str = ""
    entry_lo: float = 0.0  # entry zone (spot $) — used only for limit-entry model
    entry_hi: float = 0.0

    @property
    def zone_width_pip(self) -> float:
        return abs(self.entry_hi - self.entry_lo) / PIP if self.entry_hi else 0.0


def _simulate(c_arr, h_arr, l_arr, ts_arr, n, entry_i, side, sl_pip, tp_pips, max_bars):
    """Walk forward from entry_i. Clean chronological ordering:
       - tp1_before_sl: True iff TP1 is touched on a bar strictly BEFORE SL.
       - tp_reached: highest TP level index reached before SL hit (0 = none).
       - sl_first: SL touched before TP1 ever touched.
       Ties within a bar resolve PESSIMISTICALLY (SL wins).
    """
    entry = c_arr[entry_i]
    sl_d = sl_pip * PIP
    tp_d = [t * PIP for t in tp_pips]
    sl = entry - side * sl_d
    tps = [entry + side * d for d in tp_d]

    tp1_before_sl = False
    sl_first = False
    tp_reached = 0
    for j in range(entry_i + 1, min(entry_i + 1 + max_bars, n)):
        if side == 1:
            sl_touch = l_arr[j] <= sl
            tp_touch = [h_arr[j] >= tp for tp in tps]
        else:
            sl_touch = h_arr[j] >= sl
            tp_touch = [l_arr[j] <= tp for tp in tps]

        # Pessimistic tie: if SL touched this bar and TP1 not yet recorded as
        # before-SL, the SL ends the trade (loss), even if TP1 also prints now.
        if sl_touch and not tp1_before_sl:
            sl_first = True
            break
        # Record TPs reached this bar (only matters once we know TP1 came first).
        for k in range(len(tps)):
            if tp_touch[k]:
                if k == 0 and not tp1_before_sl:
                    tp1_before_sl = True
                if tp1_before_sl:
                    tp_reached = max(tp_reached, k + 1)
        if sl_touch:   # SL after TP1 already booked → stop (runner stopped out)
            break
        if tp_reached >= len(tps):
            break
    return tp1_before_sl, sl_first, tp_reached


def random_baseline(df, sl_pip, tp1_pip, n_samples=4000, max_bars=288, seed=1):
    c = df["close"].to_numpy(); h = df["high"].to_numpy(); l = df["low"].to_numpy()
    ts = df["timestamp"].to_numpy(); n = len(c)
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(50, n - max_bars - 1), size=min(n_samples, n - max_bars - 100), replace=False)
    wins = 0; tot = 0
    for side in (1, -1):
        for i in idx:
            tp1b, slf, _ = _simulate(c, h, l, ts, n, int(i), side, sl_pip, [tp1_pip], max_bars)
            if tp1b:
                wins += 1; tot += 1
            elif slf:
                tot += 1
    return wins / tot if tot else 0.0, tot


def verify_signals(df, signals, max_bars=288):
    c = df["close"].to_numpy(); h = df["high"].to_numpy(); l = df["low"].to_numpy()
    ts = df["timestamp"]; n = len(c)
    out = []
    for s in signals:
        post = pd.Timestamp(s.post_utc)
        # nearest bar at/after post time
        future = ts[ts >= post]
        if len(future) == 0:
            out.append({"name": s.name, "status": "NO_DATA"}); continue
        ei = ts.searchsorted(future.iloc[0])
        tp1b, slf, tp_reached = _simulate(c, h, l, ts.to_numpy(), n, int(ei), s.side,
                                          s.sl_pip, s.tp_pips, max_bars)
        out.append({
            "name": s.name, "channel": s.channel,
            "side": "BUY" if s.side == 1 else "SELL",
            "rr1": round(s.tp_pips[0] / s.sl_pip, 2),
            "entry_px": round(c[ei], 1),
            "tp1_before_sl": tp1b, "tp_reached": tp_reached,
            "result": "WIN_TP1" if tp1b else ("LOSS_SL" if slf else "UNRESOLVED"),
        })
    return out


# Signals from screenshots. Times shown are Vietnam local (UTC+7) → UTC = local-7.
# Each tuple: (label, post_utc, side, sl_pip, [TP pips], channel)
SIGNALS = [
    # name, post_utc, side, sl_pip, tp_pips, channel, entry_lo, entry_hi
    # ---- May 26 2026 (DOWN -0.5%) ----
    Sig("0526_0736_SELL", "2026-05-26T00:36:00+00:00", -1, 100, [50, 100, 150, 200], "PP2",     4552, 4555),
    Sig("0526_1135_BUY",  "2026-05-26T04:35:00+00:00",  1, 100, [50, 100, 150, 200], "PP2",     4528, 4531),
    Sig("0526_1235_SELL", "2026-05-26T05:35:00+00:00", -1, 100, [50, 100, 150, 200], "PP2",     4533, 4536),
    Sig("0526_1307_SELL", "2026-05-26T06:07:00+00:00", -1, 100, [50, 100, 150, 200], "Scalp",   4534, 4537),
    Sig("0526_1448_SELL", "2026-05-26T07:48:00+00:00", -1, 100, [50, 100, 150, 200], "PP2",     4533, 4536),
    Sig("0526_1459_SELL", "2026-05-26T07:59:00+00:00", -1, 100, [50, 100, 150, 200], "PP2",     4537, 4540),
    Sig("0526_1848_BUY",  "2026-05-26T11:48:00+00:00",  1, 100, [50, 100, 150, 200], "Scalp",   4506, 4509),
    Sig("0526_1859_BUY",  "2026-05-26T11:59:00+00:00",  1, 100, [150, 200, 300, 400], "Signals", 4488, 4498),
    Sig("0526_2059_BUY",  "2026-05-26T13:59:00+00:00",  1, 100, [50, 100, 150, 200], "PP2",     4511, 4514),
    # ---- May 27 2026 (DOWN -1.63%) ----
    Sig("0527_0052_SELL", "2026-05-26T17:52:00+00:00", -1, 100, [50, 100, 150, 200], "Scalp",   4501, 4505),
    Sig("0527_0841_BUY",  "2026-05-27T01:41:00+00:00",  1, 100, [50, 100, 150, 200], "PP2",     4509, 4512),
    Sig("0527_1245_BUY",  "2026-05-27T05:45:00+00:00",  1, 100, [150, 200, 300, 400], "Signals", 4478, 4488),
    Sig("0527_1349_SELL", "2026-05-27T06:49:00+00:00", -1, 100, [50, 100, 150, 200], "Scalp",   4491, 4494),
    Sig("0527_1533_SELL", "2026-05-27T08:33:00+00:00", -1, 100, [50, 100, 150, 200], "PP2",     4487, 4490),
    Sig("0527_1741_BUY",  "2026-05-27T10:41:00+00:00",  1, 100, [150, 200, 300, 400], "Signals", 4453, 4463),
    Sig("0527_1836_BUY",  "2026-05-27T11:36:00+00:00",  1, 100, [150, 200, 300, 400], "Signals", 4423, 4433),
    Sig("0527_1942_SELL", "2026-05-27T12:42:00+00:00", -1, 100, [100, 200, 300],      "PRI_RR1.0", 4460, 4470),
    Sig("0527_1953_SELL", "2026-05-27T12:53:00+00:00", -1, 100, [50, 100, 150, 200], "PP2",     4444, 4447),
    Sig("0527_2011_BUY",  "2026-05-27T13:11:00+00:00",  1, 100, [150, 200, 300, 400], "Signals", 4403, 4413),
    # ---- May 28 2026 ----
    Sig("0528_0905_BUY",  "2026-05-28T02:05:00+00:00",  1, 100, [150, 200, 300, 400], "Signals", 4393, 4403),
    Sig("0528_1121_BUY",  "2026-05-28T04:21:00+00:00",  1, 100, [150, 200, 300, 400], "Signals", 4358, 4368),
    Sig("0528_1231_BUY",  "2026-05-28T05:31:00+00:00",  1, 100, [100, 200, 300],      "PRI_RR1.0", 4370, 4380),
    Sig("0528_2003_SELL", "2026-05-28T13:03:00+00:00", -1, 100, [50, 100, 150, 200], "PP2",     4434, 4437),
    Sig("0528_2009_SELL", "2026-05-28T13:09:00+00:00", -1, 100, [50, 100, 150, 200], "PP2",     4431, 4434),
    Sig("0528_2113_SELL", "2026-05-28T14:13:00+00:00", -1, 100, [150, 200, 300, 400], "Signals", 4442, 4452),
    Sig("0528_2114_SELL", "2026-05-28T14:14:00+00:00", -1, 100, [150, 200, 300, 400], "Signals", 4462, 4472),
    Sig("0528_2120_SELL", "2026-05-28T14:20:00+00:00", -1, 100, [50, 100, 150, 200], "PP2",     4466, 4469),
    Sig("0528_2322_SELL", "2026-05-28T16:22:00+00:00", -1, 100, [50, 100, 150, 200], "PP2",     4499, 4502),
    # ---- May 29 2026 (UP day — counter-trend SELLs = key test) ----
    Sig("0529_0947_SELL", "2026-05-29T02:47:00+00:00", -1, 100, [100, 200, 300],      "PRI_RR1.0", 4507, 4517),
    Sig("0529_1128_BUY",  "2026-05-29T04:28:00+00:00",  1, 100, [50, 100, 150, 200], "PP2",     4506, 4509),
    Sig("0529_1241_BUY",  "2026-05-29T05:41:00+00:00",  1, 100, [50, 100, 150, 200], "Scalp",   4499, 4502),
    Sig("0529_1316_SELL", "2026-05-29T06:16:00+00:00", -1, 100, [50, 100, 150, 200], "Scalp",   4516, 4519),
    Sig("0529_1403_SELL", "2026-05-29T07:03:00+00:00", -1, 100, [150, 200, 300, 400], "Signals", 4525, 4535),
    Sig("0529_1659_BUY",  "2026-05-29T09:59:00+00:00",  1, 100, [50, 100, 150, 200], "PP2",     4525, 4528),
    Sig("0529_1753_SELL", "2026-05-29T10:53:00+00:00", -1, 100, [50, 100, 150, 200], "Scalp",   4539, 4542),
    Sig("0529_2016_SELL", "2026-05-29T13:16:00+00:00", -1, 100, [100, 200, 300],      "PRI_RR1.0", 4525, 4535),
    Sig("0529_2022_SELL", "2026-05-29T13:22:00+00:00", -1, 100, [150, 200, 300, 400], "Signals", 4530, 4540),
    Sig("0529_2033_SELL", "2026-05-29T13:33:00+00:00", -1, 100, [50, 100, 150, 200], "PP2",     4531, 4534),
    Sig("0529_2053_SELL", "2026-05-29T13:53:00+00:00", -1, 100, [50, 100, 150, 200], "Scalp",   4537, 4540),
    Sig("0529_2106_SELL", "2026-05-29T14:06:00+00:00", -1, 100, [50, 100, 150, 200], "Scalp",   4548, 4551),
]


def verify_limit(df, signals, expire_bars=36, max_bars=288):
    """Limit-entry model (basis-free, at-market assumption).

    UG posts an entry ZONE and the trader fills at the FAVORABLE edge as price
    retraces into it. We assume the near-market edge of the zone == post-time
    price (most UG entries are at-market), so in PAXG space:
       BUY  fills at P0 - W  (price must dip the zone width W)
       SELL fills at P0 + W  (price must rally W)
    If price never reaches the favorable edge within `expire_bars`, the limit is
    NOT filled (mirrors UG's 'skip if already ran' selection) and the signal is
    excluded — neither win nor loss.
    SL/TP are measured (in pips) from the fill price.
    """
    c = df["close"].to_numpy(); h = df["high"].to_numpy(); l = df["low"].to_numpy()
    ts = df["timestamp"]; n = len(c)
    out = []
    for s in signals:
        post = pd.Timestamp(s.post_utc)
        fut = ts[ts >= post]
        if len(fut) == 0:
            out.append({"name": s.name, "status": "NO_DATA"}); continue
        i0 = int(ts.searchsorted(fut.iloc[0]))
        P0 = c[i0]
        W = s.zone_width_pip * PIP
        fill_level = P0 - s.side * W   # BUY: P0-W ; SELL: P0+W
        fill_i = None
        for j in range(i0, min(i0 + expire_bars + 1, n)):
            if s.side == 1 and l[j] <= fill_level:
                fill_i = j; break
            if s.side == -1 and h[j] >= fill_level:
                fill_i = j; break
        rr1 = round(s.tp_pips[0] / s.sl_pip, 2)
        sd = "BUY" if s.side == 1 else "SELL"
        if fill_i is None:
            out.append({"name": s.name, "channel": s.channel, "side": sd, "rr1": rr1,
                        "filled": False, "result": "NO_FILL", "tp1_before_sl": None})
            continue
        tp1b, slf, reached = _simulate(c, h, l, ts.to_numpy(), n, fill_i, s.side,
                                       s.sl_pip, s.tp_pips, max_bars)
        out.append({"name": s.name, "channel": s.channel, "side": sd, "rr1": rr1,
                    "filled": True, "tp1_before_sl": tp1b, "tp_reached": reached,
                    "result": "WIN_TP1" if tp1b else ("LOSS_SL" if slf else "UNRESOLVED")})
    return out


def main() -> int:
    df = load_paxg("PAXGUSDT", "5m")
    print(f"PAXG 5m: {len(df)} bars, {df.timestamp.iloc[0]} -> {df.timestamp.iloc[-1]}\n")

    print("=== RANDOM BASELINE on PAXG (zero skill, 24h horizon) ===")
    for label, sl, tp1 in [("R:R 0.5 (SL100/TP50)", 100, 50),
                            ("R:R 1.0 (SL100/TP100)", 100, 100),
                            ("R:R 1.5 (SL100/TP150)", 100, 150)]:
        wr, tot = random_baseline(df, sl, tp1)
        rr = tp1 / sl
        exp = wr * rr - (1 - wr)
        print(f"  {label:24s}: WR={wr:.1%}  exp={exp:+.3f}R  (n={tot})")

    print("\n=== UG SIGNAL OUTCOMES on PAXG (market entry at post time) ===")
    res = verify_signals(df, SIGNALS)
    for r in res:
        if r.get("status") == "NO_DATA":
            print(f"  {r['name']}: NO DATA at that time"); continue
        print(f"  {r['name']:16s} {r['channel']:10s} {r['side']:4s} R:R{r['rr1']:<4} "
              f"entry~{r['entry_px']} | TP1<SL={'WIN ' if r['tp1_before_sl'] else 'loss'} "
              f"maxTP={r['tp_reached']}")
    done = [r for r in res if r.get("result")]
    if not done:
        return 0

    def wr(rows):
        w = sum(1 for r in rows if r["tp1_before_sl"])
        return w, len(rows), (w / len(rows) if rows else 0)

    w, n, p = wr(done)
    print(f"\n  OVERALL TP1-before-SL: {w}/{n} = {p:.0%}")
    print("\n  By direction:")
    for sd in ("SELL", "BUY"):
        sub = [r for r in done if r["side"] == sd]
        if sub:
            w, n, p = wr(sub); print(f"    {sd}: {w}/{n} = {p:.0%}")
    print("\n  By R:R (vs random baseline):")
    base = {0.5: 0.665, 1.0: 0.50, 1.5: 0.397, 3.0: 0.25}
    for rr in sorted(set(r["rr1"] for r in done)):
        sub = [r for r in done if r["rr1"] == rr]
        w, n, p = wr(sub)
        b = base.get(rr, None)
        edge = f"(random {b:.0%}, edge {p-b:+.0%})" if b else ""
        print(f"    R:R {rr}: {w}/{n} = {p:.0%} {edge}")
    print("\n  By day (+ that day's gold direction):")
    daydir = {}
    for day in ("0526", "0527", "0528", "0529"):
        d = pd.Timestamp(f"2026-05-{day[2:]}", tz="UTC")
        sub_bars = df[df.timestamp.dt.date == d.date()]
        if len(sub_bars):
            mv = sub_bars.close.iloc[-1] - sub_bars.close.iloc[0]
            daydir[day] = "UP" if mv > 0 else "DOWN"
    for day in ("0526", "0527", "0528", "0529"):
        sub = [r for r in done if r["name"].startswith(day)]
        if sub:
            w, n, p = wr(sub)
            ws = sum(1 for r in sub if r["side"] == "SELL" and r["tp1_before_sl"])
            ns = sum(1 for r in sub if r["side"] == "SELL")
            wb = sum(1 for r in sub if r["side"] == "BUY" and r["tp1_before_sl"])
            nb = sum(1 for r in sub if r["side"] == "BUY")
            print(f"    May {day[2:]} [{daydir.get(day,'?'):4}]: {w}/{n}={p:.0%}  "
                  f"(SELL {ws}/{ns}, BUY {wb}/{nb})")

    # ---- LIMIT-ENTRY model: per-channel comparison ----
    print("\n" + "=" * 64)
    print("LIMIT-ENTRY MODEL (fill at favorable zone edge, no-fill if missed)")
    print("=" * 64)
    lim = verify_limit(df, SIGNALS)
    lim_ok = [r for r in lim if r.get("filled")]
    n_nofill = sum(1 for r in lim if r.get("result") == "NO_FILL")
    print(f"  filled {len(lim_ok)}/{len(lim)}  (no-fill/expired: {n_nofill})")
    base = {0.5: 0.665, 1.0: 0.50, 1.5: 0.397}
    print(f"\n  {'channel':10s} {'R:R':>4s} | {'MARKET entry':>16s} | {'LIMIT entry':>16s} | random")
    chans = ["PP2", "Scalp", "PRI_RR1.0", "Signals"]
    rr_of = {"PP2": 0.5, "Scalp": 0.5, "PRI_RR1.0": 1.0, "Signals": 1.5}
    for ch in chans:
        m = [r for r in done if r["channel"] == ch]
        lq = [r for r in lim_ok if r["channel"] == ch]
        rr = rr_of[ch]
        mw = sum(1 for r in m if r["tp1_before_sl"])
        lw = sum(1 for r in lq if r["tp1_before_sl"])
        ms = f"{mw}/{len(m)} = {mw/len(m):.0%}" if m else "-"
        ls = f"{lw}/{len(lq)} = {lw/len(lq):.0%}" if lq else "-"
        print(f"  {ch:10s} {rr:>4} | {ms:>16s} | {ls:>16s} | {base[rr]:.0%}")
    # overall limit
    lw = sum(1 for r in lim_ok if r["tp1_before_sl"])
    print(f"\n  LIMIT overall TP1-before-SL: {lw}/{len(lim_ok)} = {lw/len(lim_ok):.0%}"
          if lim_ok else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
