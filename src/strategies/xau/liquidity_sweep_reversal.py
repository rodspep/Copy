"""XAU liquidity-sweep reversal scalp.

Idea: Recent swing high/low marks resting liquidity. A bar that wicks past the
swing extreme but closes back inside has "grabbed" that liquidity — often
followed by a snap reversal. We enter at the close of the sweep bar, SL beyond
the wick extreme, TP at a fast RR (this is the small-target / high-WR variant).

Symmetric long-and-short.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import atr, swings, liquidity_sweeps, vwap
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


class XauLiquiditySweepReversal(Strategy):
    ltf = "M5"
    required_htfs: tuple[str, ...] = ()

    default_params: dict[str, Any] = {
        "swing_left": 5,
        "swing_right": 5,
        "atr_period": 14,
        "sl_atr_mult": 0.5,
        "tp_atr_mult": 1.0,
        "trade_start_hour": 7,
        "trade_end_hour": 20,
        "vwap_filter": True,  # only fade in the direction of session VWAP
    }

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        p = self.merged_params(params)
        sw = swings(ltf, left=int(p["swing_left"]), right=int(p["swing_right"]))
        ls = liquidity_sweeps(ltf, sw)
        a = atr(ltf, int(p["atr_period"]))

        hour = ltf["timestamp"].dt.hour
        in_window = (hour >= int(p["trade_start_hour"])) & (hour < int(p["trade_end_hour"]))

        if p["vwap_filter"]:
            v = vwap(ltf, anchor="daily_2200")
            above_vwap = ltf["close"] > v["vwap"]
            below_vwap = ltf["close"] < v["vwap"]
        else:
            above_vwap = pd.Series(True, index=ltf.index)
            below_vwap = pd.Series(True, index=ltf.index)

        long_signal = in_window & ls["bull_sweep"] & above_vwap & a.notna()
        short_signal = in_window & ls["bear_sweep"] & below_vwap & a.notna()

        sigs = empty_signals(ltf)
        sl_dist = float(p["sl_atr_mult"]) * a
        tp_dist = float(p["tp_atr_mult"]) * a

        if long_signal.any():
            sigs.loc[long_signal, "action"] = "enter_long"
            sigs.loc[long_signal, "sl"] = (ltf["low"] - sl_dist).loc[long_signal].values
            sigs.loc[long_signal, "tp"] = (ltf["close"] + tp_dist).loc[long_signal].values
        if short_signal.any():
            sigs.loc[short_signal, "action"] = "enter_short"
            sigs.loc[short_signal, "sl"] = (ltf["high"] + sl_dist).loc[short_signal].values
            sigs.loc[short_signal, "tp"] = (ltf["close"] - tp_dist).loc[short_signal].values

        validate_signals(sigs, len(ltf))
        return StrategyResult(signals=sigs)
