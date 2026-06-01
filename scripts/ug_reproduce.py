"""Reproduction test: does UG's decoded MECHANICAL core reproduce its claimed edge?

Hypothesis (from UG's own words "đớp 5 giá", "TP1 95%"): the edge is the tight-TP
fade geometry, not direction accuracy. TP1 = 50 pip = 5.0 price sits at half the
SL distance (10.0), entered as a limit ~1×ATR into a pullback from an extreme —
so mean-reversion makes TP1 hit far more often than SL, REGARDLESS of a "correct"
directional call.

This simulates that clone on MT5 M5 data and measures the TP1-before-SL rate.

Clone rule per bar i (signal at close, no lookahead):
  - direction = fade vs M5 SMA34: close>SMA34 → SHORT (sell rally), else LONG.
  - entry = limit k×ATR(M5) BEYOND current close in the fade direction
            (short: above; long: below) — UG enters deeper into the extreme.
  - SL = entry ∓ 10.0 ; TP1 = entry ± 5.0  (UG scalp geometry).
Resolution: from bar i+1, fill if price touches entry within FILL_BARS; then from
the bar AFTER the fill (we can't know intrabar order on the fill bar itself),
TP1 vs SL within RES_BARS. tie='sl' conservative / 'tp1' optimistic on same-bar
TP+SL. Resolved trades are non-overlapping (pending limits may still overlap).

Run: python -X utf8 -m scripts.ug_reproduce
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.tv_loader import load_tv_csv
from src.indicators import atr

SL_PRICE = 10.0          # UG fixed SL (100 pip)
TP1_PRICE = 5.0          # UG scalp TP1 (50 pip)
FILL_BARS = 12           # 1h to get filled, else cancel
RES_BARS = 48            # 4h to resolve, else timeout


def simulate(df: pd.DataFrame, k_atr: float, tie: str = "sl") -> dict:
    """tie='sl' = conservative (SL wins same-bar ties); 'tp' = optimistic."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    sma34 = df["close"].rolling(34).mean().to_numpy()
    a = atr(df, 14).to_numpy()
    n = len(df)
    out = {"signals": 0, "filled": 0, "tp1": 0, "sl": 0, "timeout": 0, "cancelled": 0}

    i = 100
    while i < n - 1:
        if np.isnan(sma34[i]) or np.isnan(a[i]) or a[i] <= 0:
            i += 1
            continue
        short = close[i] > sma34[i]                 # fade: above mean → sell
        off = k_atr * a[i]
        if short:
            entry = close[i] + off
            sl, tp = entry + SL_PRICE, entry - TP1_PRICE
        else:
            entry = close[i] - off
            sl, tp = entry - SL_PRICE, entry + TP1_PRICE
        out["signals"] += 1

        # --- fill: price must touch the limit within FILL_BARS ---
        fill_j = None
        for j in range(i + 1, min(i + 1 + FILL_BARS, n)):
            if (short and high[j] >= entry) or (not short and low[j] <= entry):
                fill_j = j
                break
        if fill_j is None:
            out["cancelled"] += 1
            i += 1
            continue
        out["filled"] += 1

        # --- resolve TP1 vs SL starting the bar AFTER the fill ---
        # We must NOT score on the fill bar itself: we know price touched `entry`
        # there, but not whether TP/SL printed before or after the fill within
        # that candle (Codex finding — counting fill-bar TP inflates the rate).
        res, end = None, min(fill_j + 1 + RES_BARS, n)
        for j in range(fill_j + 1, end):
            hit_sl = high[j] >= sl if short else low[j] <= sl
            hit_tp = low[j] <= tp if short else high[j] >= tp
            if hit_sl and hit_tp:
                res = tie           # 'sl' pessimistic / 'tp1' optimistic on ties
            elif hit_sl:
                res = "sl"
            elif hit_tp:
                res = "tp1"
            if res:
                end = j
                break
        if res is None:
            out["timeout"] += 1
            i = end                # past the resolution window (non-overlapping)
        else:
            out[res] += 1
            i = end + 1            # continue after resolution
    return out


def main() -> int:
    df = load_tv_csv("data/xau/XAUUSD_M5.csv")
    span = f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}"
    print(f"M5 data: {len(df)} bars · {span}\n")
    print(f"{'k×ATR':>6} {'tie':>5} {'filled':>7} {'TP1':>6} {'SL':>5} "
          f"{'TP1-rate':>9} {'expR(TP1=0.5R)':>15}")
    for k in (0.5, 0.9, 1.3):
        for tie in ("sl", "tp1"):
            r = simulate(df, k, tie=tie)
            resolved = r["tp1"] + r["sl"]
            tp1_rate = r["tp1"] / resolved if resolved else 0
            expR = (r["tp1"] * 0.5 - r["sl"] * 1.0) / resolved if resolved else 0
            lbl = "consv" if tie == "sl" else "optim"
            print(f"{k:>6.1f} {lbl:>5} {r['filled']:>7} {r['tp1']:>6} "
                  f"{r['sl']:>5} {tp1_rate:>8.0%} {expR:>+15.3f}")
    print("\nUG claims TP1 ~95% (scalp). True rate lies between consv (SL-first "
          "ties) and optim (TP-first ties). Non-overlapping samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
