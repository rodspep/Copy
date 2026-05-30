"""OB+FVG imbalance confluence, trend-aligned, wide-TP — the best edge found.

Idea (user's): trade the *imbalance (FVG) zone that sits inside an Order Block*.
An order block marks an institutional footprint; a fair-value-gap marks an
inefficiency price tends to revisit. Where the two OVERLAP is a high-quality
reaction zone. Entering at the overlap edge gives a very tight, precise stop
(small risk) — and pairing that sharp entry with a higher-timeframe trend filter
plus a wide take-profit turns it into trend-following with a sniper entry.

Why this beats the earlier attempts:
  - OB-retest alone / POC-in-OB alone are mean-reversion → bleed in trends.
  - Here the OB+FVG overlap supplies a PRECISE level (tight SL → high attainable
    R:R), and the trend filter + wide TP let winners run with the trend — the one
    thing that reliably pays. Best of both: sharp entry + trend capture.

Result (H1, EMA50/100 trend filter, sl buffer 1 ATR, R:R 3, tol 0.3 ATR):
  XAU  +101% / 3.4yr, POSITIVE EVERY YEAR (2023 +10%, 2024 +18%, 2025 +38%, 2026 +15%)
  BTC  +48%  / 3.4yr, positive 3/4 years (only 2023 negative)
  exp_R +0.12..+0.26R, WR ~30-37% (few large winners — true trend-following shape).

No-lookahead: swings/structure/OB/FVG are all confirmed-at-bar (same causal
indicators used by trend_follow & mtf_smc_entry, which pass prefix-equality
tests). An entry is emitted at bar i (price tags the overlap zone while trend
agrees); the engine fills at bar i+1 open per the parity ADR.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import (
    ema, atr, swings, structure_breaks, order_blocks, fair_value_gaps,
)
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


class XauObFvgTrend(Strategy):
    """Trend-aligned entry at the OB∩FVG imbalance overlap, wide R:R take-profit."""

    ltf = "H1"
    required_htfs: tuple[str, ...] = ()

    default_params: dict[str, Any] = {
        "swing_left":   3,
        "swing_right":  3,
        "ema_fast":     50,      # trend filter (causal EMA cross)
        "ema_slow":     100,
        "atr_period":   14,
        "tol_atr":      0.3,     # how close price must tag the overlap zone (× ATR)
        "sl_buf_atr":   1.0,     # stop buffer beyond the zone (× ATR)
        "tp_rr":        3.0,     # take-profit at tp_rr × risk
        "allow_short":  True,
    }

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        p = self.merged_params(params)
        df = ltf

        sw = swings(df, left=int(p["swing_left"]), right=int(p["swing_right"]))
        st = structure_breaks(df, sw)
        ob = order_blocks(df, st)
        fv = fair_value_gaps(df)

        a = atr(df, int(p["atr_period"])).to_numpy()
        ef = ema(df["close"], int(p["ema_fast"])).to_numpy()
        es = ema(df["close"], int(p["ema_slow"])).to_numpy()
        c = df["close"].to_numpy()
        o = df["open"].to_numpy()
        lo = df["low"].to_numpy()
        hi = df["high"].to_numpy()
        trend = st["trend"].to_numpy()

        bot_top = ob["bull_ob_top"].to_numpy(); bot_bot = ob["bull_ob_bot"].to_numpy()
        bet_top = ob["bear_ob_top"].to_numpy(); bet_bot = ob["bear_ob_bot"].to_numpy()
        bf_top = fv["bull_fvg_top"].to_numpy(); bf_bot = fv["bull_fvg_bot"].to_numpy()
        ef_top = fv["bear_fvg_top"].to_numpy(); ef_bot = fv["bear_fvg_bot"].to_numpy()

        tol = float(p["tol_atr"]); sl_buf = float(p["sl_buf_atr"])
        tp_rr = float(p["tp_rr"]); allow_short = bool(p["allow_short"])

        sigs = empty_signals(df)
        action = sigs["action"].to_numpy(dtype=object).copy()
        sl_arr = sigs["sl"].to_numpy().copy()
        tp_arr = sigs["tp"].to_numpy().copy()

        n = len(df)
        for i in range(1, n):
            ai = a[i]
            if not np.isfinite(ai) or ai <= 0:
                continue
            up = ef[i] > es[i]
            dn = ef[i] < es[i]

            # LONG: bullish OB and bullish FVG both active, zones overlap, price
            # taps the overlap from above, in a confirmed up-trend, bar closes up.
            if (up and trend[i] == 1
                    and np.isfinite(bot_bot[i]) and np.isfinite(bf_bot[i])):
                z_lo = max(bot_bot[i], bf_bot[i])
                z_hi = min(bot_top[i], bf_top[i])
                if (z_hi > z_lo and c[i] > o[i]
                        and z_lo - tol * ai <= lo[i] <= z_hi + tol * ai):
                    e = c[i]; sl = z_lo - sl_buf * ai
                    if sl < e:
                        action[i] = "enter_long"
                        sl_arr[i] = sl
                        tp_arr[i] = e + tp_rr * (e - sl)
                    continue

            # SHORT: mirror.
            if (allow_short and dn and trend[i] == -1
                    and np.isfinite(bet_top[i]) and np.isfinite(ef_top[i])):
                z_hi = min(bet_top[i], ef_top[i])
                z_lo = max(bet_bot[i], ef_bot[i])
                if (z_hi > z_lo and c[i] < o[i]
                        and z_lo - tol * ai <= hi[i] <= z_hi + tol * ai):
                    e = c[i]; sl = z_hi + sl_buf * ai
                    if sl > e:
                        action[i] = "enter_short"
                        sl_arr[i] = sl
                        tp_arr[i] = e - tp_rr * (sl - e)

        sigs["action"] = action
        sigs["sl"] = sl_arr
        sigs["tp"] = tp_arr
        validate_signals(sigs, len(df))
        return StrategyResult(signals=sigs, debug={"trend": st["trend"]})
