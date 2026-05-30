"""XAU EMA-pullback strategy (HTF trend filter + M5 pullback + ATR SL/TP).

Idea:
- HTF (H1) trend filter: long bias if H1 EMA_fast > H1 EMA_slow AND H1 ADX > threshold.
                         short bias if mirrored.
- LTF (M5) entry: on a pullback to the M5 EMA_pullback level, with a confirmation
  candle in the trend direction. ATR-based SL beyond the pullback, ATR-based TP
  at a configurable RR multiple.

Symmetric on both sides (long-and-short, per [[feedback-backtest-scope]]).

Notes:
- This is one of several XAU candidates; the optimizer will tune parameters per
  asset and the shortlist task picks the top 1-3.
- Strict no-lookahead: every condition for bar `i` reads only `ltf[0..i]` or HTF
  values aligned via `align_htf_to_ltf` (which is availability-time correct).
- Signal at bar `i` close → engine fills at bar `i+1` open per parity ADR §2.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import ema, adx, atr, align_htf_to_ltf
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


class XauEmaPullback(Strategy):
    """HTF EMA-trend + M5 pullback + ATR SL/TP. Long-and-short symmetric."""

    ltf = "M5"
    required_htfs = ("H1",)

    default_params: dict[str, Any] = {
        # HTF trend filter
        "htf_ema_fast": 50,
        "htf_ema_slow": 200,
        "htf_adx_period": 14,
        "htf_adx_min": 20.0,         # require minimum trend strength

        # LTF pullback / confirmation
        "ltf_ema_pullback": 20,
        "ltf_pullback_atr_mult": 0.5,  # how close to EMA we accept ("pullback proximity")
        "ltf_atr_period": 14,

        # SL/TP
        "sl_atr_mult": 1.2,
        "tp_atr_mult": 1.8,
    }

    def generate_signals(
        self,
        ltf: pd.DataFrame,
        htfs: dict[str, pd.DataFrame],
        params: dict[str, Any] | None = None,
    ) -> StrategyResult:
        if "H1" not in htfs:
            raise ValueError("XauEmaPullback requires HTF 'H1' in htfs")
        p = self.merged_params(params)

        # ---- HTF features ----
        h1 = htfs["H1"].copy()
        h1["htf_ema_fast"] = ema(h1["close"], int(p["htf_ema_fast"]))
        h1["htf_ema_slow"] = ema(h1["close"], int(p["htf_ema_slow"]))
        h1_adx = adx(h1, int(p["htf_adx_period"]))
        h1["htf_adx"] = h1_adx["adx"]
        h1_features = h1[["timestamp", "htf_ema_fast", "htf_ema_slow", "htf_adx"]]

        # Align HTF onto LTF via availability-time merge (ADR §5, no lookahead).
        aligned = align_htf_to_ltf(
            ltf=ltf,
            htf=h1_features,
            ltf_tf=self.ltf,
            htf_tf="H1",
            htf_cols=["htf_ema_fast", "htf_ema_slow", "htf_adx"],
            suffix="",
        )

        htf_ema_fast = aligned["htf_ema_fast"]
        htf_ema_slow = aligned["htf_ema_slow"]
        htf_adx_v = aligned["htf_adx"]

        # ---- LTF features ----
        ltf_ema_pull = ema(ltf["close"], int(p["ltf_ema_pullback"]))
        ltf_atr_v = atr(ltf, int(p["ltf_atr_period"]))

        # ---- Trend regime (HTF) ----
        long_trend = (
            htf_ema_fast.notna() & htf_ema_slow.notna()
            & (htf_ema_fast > htf_ema_slow)
            & (htf_adx_v >= float(p["htf_adx_min"]))
        )
        short_trend = (
            htf_ema_fast.notna() & htf_ema_slow.notna()
            & (htf_ema_fast < htf_ema_slow)
            & (htf_adx_v >= float(p["htf_adx_min"]))
        )

        # ---- Pullback proximity to LTF EMA (within `proximity_atr_mult` * ATR) ----
        prox = float(p["ltf_pullback_atr_mult"]) * ltf_atr_v
        # Long pullback: low touched/got within `prox` of EMA from above; close above EMA.
        pullback_long = (
            (ltf["low"] <= ltf_ema_pull + prox)
            & (ltf["close"] > ltf_ema_pull)
        )
        # Short pullback: high touched/got within prox of EMA from below; close below EMA.
        pullback_short = (
            (ltf["high"] >= ltf_ema_pull - prox)
            & (ltf["close"] < ltf_ema_pull)
        )

        # ---- Confirmation candle (close in trend direction relative to open) ----
        bullish_candle = ltf["close"] > ltf["open"]
        bearish_candle = ltf["close"] < ltf["open"]

        long_signal = long_trend & pullback_long & bullish_candle & ltf_atr_v.notna()
        short_signal = short_trend & pullback_short & bearish_candle & ltf_atr_v.notna()

        # ---- Build SL/TP from ATR ----
        sl_dist = float(p["sl_atr_mult"]) * ltf_atr_v
        tp_dist = float(p["tp_atr_mult"]) * ltf_atr_v

        sigs = empty_signals(ltf)
        # Long entries
        if long_signal.any():
            sigs.loc[long_signal, "action"] = "enter_long"
            sigs.loc[long_signal, "sl"] = (ltf["close"] - sl_dist).loc[long_signal].values
            sigs.loc[long_signal, "tp"] = (ltf["close"] + tp_dist).loc[long_signal].values
        # Short entries (mutually exclusive with long since HTF trend is exclusive)
        if short_signal.any():
            sigs.loc[short_signal, "action"] = "enter_short"
            sigs.loc[short_signal, "sl"] = (ltf["close"] + sl_dist).loc[short_signal].values
            sigs.loc[short_signal, "tp"] = (ltf["close"] - tp_dist).loc[short_signal].values

        validate_signals(sigs, len(ltf))

        return StrategyResult(signals=sigs, debug={
            "long_trend": long_trend,
            "short_trend": short_trend,
            "ltf_ema_pull": ltf_ema_pull,
            "ltf_atr": ltf_atr_v,
        })
