"""Verify UG Trading signals against actual XAU M5 data.

Takes a list of UG signals (entry range, SL, TP1-4, direction, post_time UTC)
and simulates fills + outcomes on real M5 data. Reports WR per TP level.

Each signal is simulated as a 2-leg PENDING LIMIT order (UG-style range):
  - Long: limit_leg1 at entry_top (closer to current price), limit_leg2 at entry_bot (further away)
  - Short: mirror

For each filled leg:
  - Watch subsequent bars
  - Track if SL hit FIRST (loss) or TP1/TP2/TP3/TP4 hit FIRST (partial wins)
  - Record outcome

Usage:
  python -m scripts.verify_ug_signals
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from datetime import timedelta

from src.data.histdata_loader import load


@dataclass
class UgSignal:
    name: str
    post_time_utc: str           # ISO format
    direction: str                # 'BUY' or 'SELL'
    entry_low: float              # lower bound of entry range
    entry_high: float             # upper bound of entry range
    sl: float
    tp1_pip: int                  # TP1 in pip (XAU 1 pip = 0.1 USD)
    tp2_pip: int
    tp3_pip: int
    tp4_pip: int
    channel: str                  # 'PhuongPhap2' / 'Scalp' / 'Signals'
    tag: str                      # 'FOLLOW' / 'CAUTION'
    expire_hours: int = 24        # how long to wait for limit fill


# UG signals extracted from screenshots (May 26 2026, Vietnam time → UTC by -7h)
# XAU 1 pip = 0.10 USD (so TP1=50 pip = $5.0)
PIP = 0.10

UG_SIGNALS_MAY26 = [
    UgSignal("S1_11:35_SELL", "2026-05-26T04:35:00+00:00", "SELL",
             entry_low=4552, entry_high=4555, sl=4565,
             tp1_pip=50, tp2_pip=100, tp3_pip=150, tp4_pip=200,
             channel="PhuongPhap2", tag="CAUTION"),
    UgSignal("S2_12:34_SELL", "2026-05-26T05:34:00+00:00", "SELL",
             entry_low=4533, entry_high=4536, sl=4546,
             tp1_pip=50, tp2_pip=100, tp3_pip=150, tp4_pip=200,
             channel="PhuongPhap2", tag="FOLLOW"),
    UgSignal("S3_13:07_SELL", "2026-05-26T06:07:00+00:00", "SELL",
             entry_low=4533, entry_high=4536, sl=4546,
             tp1_pip=50, tp2_pip=100, tp3_pip=150, tp4_pip=200,
             channel="PhuongPhap2", tag="CAUTION"),
    UgSignal("S4_14:48_SELL", "2026-05-26T07:48:00+00:00", "SELL",
             entry_low=4537, entry_high=4540, sl=4550,
             tp1_pip=50, tp2_pip=100, tp3_pip=150, tp4_pip=200,
             channel="PhuongPhap2", tag="CAUTION"),
    UgSignal("S5_14:59_BUY", "2026-05-26T07:59:00+00:00", "BUY",
             entry_low=4506, entry_high=4509, sl=4496,
             tp1_pip=50, tp2_pip=100, tp3_pip=150, tp4_pip=200,
             channel="Scalp", tag="CAUTION"),
    UgSignal("S6_18:48_BUY", "2026-05-26T11:48:00+00:00", "BUY",
             entry_low=4506, entry_high=4509, sl=4496,
             tp1_pip=50, tp2_pip=100, tp3_pip=150, tp4_pip=200,
             channel="Scalp", tag="CAUTION"),
    UgSignal("S7_18:59_BUY_PREMIUM", "2026-05-26T11:59:00+00:00", "BUY",
             entry_low=4488, entry_high=4498, sl=4478,
             tp1_pip=150, tp2_pip=200, tp3_pip=300, tp4_pip=400,
             channel="Signals", tag="CAUTION"),
    UgSignal("S8_19:59_BUY", "2026-05-26T12:59:00+00:00", "BUY",
             entry_low=4511, entry_high=4514, sl=4501,
             tp1_pip=50, tp2_pip=100, tp3_pip=150, tp4_pip=200,
             channel="PhuongPhap2", tag="CAUTION"),
]


def simulate_signal(m5: pd.DataFrame, sig: UgSignal) -> dict:
    """Simulate one UG signal against M5 data. Return dict with outcomes."""
    post_t = pd.Timestamp(sig.post_time_utc)
    # Find bars >= post_t
    future = m5[m5["timestamp"] >= post_t]
    if future.empty:
        return {"name": sig.name, "status": "NO_DATA", "reason": "post_time after data end"}

    # Limit fill window
    expire_t = post_t + timedelta(hours=sig.expire_hours)
    window = future[future["timestamp"] <= expire_t].reset_index(drop=True)
    if window.empty:
        return {"name": sig.name, "status": "NO_DATA", "reason": "no bars in fill window"}

    # Identify limit fill price (UG range = ladder; we model simplest: fill when
    # price first reaches the FAR edge of entry range)
    if sig.direction == "BUY":
        # Limit BUY: price must drop INTO the range. Fill when low <= entry_high
        fill_price = sig.entry_high  # first leg fills here (closer to current)
        fill_mask = window["low"] <= fill_price
    else:
        # Limit SELL: price must rise INTO the range. Fill when high >= entry_low
        fill_price = sig.entry_low
        fill_mask = window["high"] >= fill_price

    if not fill_mask.any():
        return {"name": sig.name, "status": "NOT_FILLED",
                "reason": "limit never reached in 24h window",
                "channel": sig.channel, "tag": sig.tag}

    fill_idx = fill_mask.idxmax()
    fill_bar = window.iloc[fill_idx]

    # Post-fill: scan subsequent bars, track which level hits first
    # We don't include fill_bar itself in SL/TP race (entry fill happens within it)
    post_fill = window.iloc[fill_idx:].reset_index(drop=True)

    # Compute TP levels in price space
    pip = PIP
    if sig.direction == "BUY":
        tp1 = fill_price + sig.tp1_pip * pip
        tp2 = fill_price + sig.tp2_pip * pip
        tp3 = fill_price + sig.tp3_pip * pip
        tp4 = fill_price + sig.tp4_pip * pip
        sl_price = sig.sl  # already absolute
    else:
        tp1 = fill_price - sig.tp1_pip * pip
        tp2 = fill_price - sig.tp2_pip * pip
        tp3 = fill_price - sig.tp3_pip * pip
        tp4 = fill_price - sig.tp4_pip * pip
        sl_price = sig.sl

    # Track first-hit per level
    outcome = {
        "name": sig.name,
        "channel": sig.channel,
        "tag": sig.tag,
        "direction": sig.direction,
        "fill_time": str(fill_bar["timestamp"]),
        "fill_price": fill_price,
        "sl_price": sl_price,
        "tp1_price": tp1, "tp2_price": tp2, "tp3_price": tp3, "tp4_price": tp4,
        "hit_tp1": False, "hit_tp2": False, "hit_tp3": False, "hit_tp4": False,
        "hit_sl": False,
        "sl_before_tp1": False,  # critical: did SL hit before even TP1?
    }

    for _, bar in post_fill.iterrows():
        high = bar["high"]; low = bar["low"]
        if sig.direction == "BUY":
            sl_touched = low <= sl_price
            tp1_touched = high >= tp1
            tp2_touched = high >= tp2
            tp3_touched = high >= tp3
            tp4_touched = high >= tp4
        else:
            sl_touched = high >= sl_price
            tp1_touched = low <= tp1
            tp2_touched = low <= tp2
            tp3_touched = low <= tp3
            tp4_touched = low <= tp4

        # Conservative tie-break: if BOTH SL and TP touched same bar, count SL hit first
        # (pessimistic — gap risk)
        if sl_touched and not outcome["hit_tp1"]:
            outcome["hit_sl"] = True
            outcome["sl_before_tp1"] = True
            break
        if sl_touched:
            outcome["hit_sl"] = True
            break

        if tp1_touched: outcome["hit_tp1"] = True
        if tp2_touched: outcome["hit_tp2"] = True
        if tp3_touched: outcome["hit_tp3"] = True
        if tp4_touched: outcome["hit_tp4"] = True

        if outcome["hit_tp4"]:
            break  # all TPs hit, stop

    outcome["status"] = "DONE"
    return outcome


def main() -> int:
    print("Loading XAU M5 data...")
    m5 = load("XAUUSD", "M5")
    print(f"M5: {len(m5)} bars, range {m5['timestamp'].iloc[0]} → {m5['timestamp'].iloc[-1]}")
    print()

    results = []
    for sig in UG_SIGNALS_MAY26:
        out = simulate_signal(m5, sig)
        results.append(out)

    # Print per-signal
    print(f"\n{'='*100}")
    print("PER-SIGNAL RESULTS")
    print("="*100)
    for r in results:
        if r.get("status") == "DONE":
            tps = [f"TP{i}={'✓' if r[f'hit_tp{i}'] else '✗'}" for i in range(1,5)]
            sl = "SL=✓" if r["hit_sl"] else "SL=✗"
            print(f"  {r['name']:30s} {r['direction']:5s} {r['channel']:12s} {r['tag']:8s} "
                  f"fill={r['fill_price']:.2f} | {' '.join(tps)} | {sl}")
        else:
            print(f"  {r['name']:30s} {r.get('status','?')}: {r.get('reason','')}")

    # Aggregate WR
    done = [r for r in results if r.get("status") == "DONE"]
    not_filled = [r for r in results if r.get("status") == "NOT_FILLED"]
    no_data = [r for r in results if r.get("status") == "NO_DATA"]

    print(f"\n{'='*100}")
    print(f"AGGREGATE: total={len(results)}, filled={len(done)}, not_filled={len(not_filled)}, no_data={len(no_data)}")
    print("="*100)

    if not done:
        print("No filled signals — cannot compute WR.")
        return 0

    for level in ["tp1", "tp2", "tp3", "tp4"]:
        n_hit = sum(1 for r in done if r[f"hit_{level}"])
        print(f"  {level.upper()} hit rate: {n_hit}/{len(done)} = {n_hit/len(done):.1%}")

    n_sl = sum(1 for r in done if r["hit_sl"])
    n_sl_before_tp1 = sum(1 for r in done if r["sl_before_tp1"])
    print(f"  SL hit (ever):           {n_sl}/{len(done)} = {n_sl/len(done):.1%}")
    print(f"  SL hit BEFORE TP1:       {n_sl_before_tp1}/{len(done)} = {n_sl_before_tp1/len(done):.1%}")

    # UG-style WR claim breakdown
    print(f"\n{'='*100}")
    print("UG CLAIM CHECK")
    print(f"  UG claims TP1 ~85-95%: We see {sum(1 for r in done if r['hit_tp1'])/len(done):.1%}")
    print(f"  TP1-hit-before-SL:      {(len(done)-n_sl_before_tp1)/len(done):.1%}")
    print("="*100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
