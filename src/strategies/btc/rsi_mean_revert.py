"""BTC RSI mean-revert in range regime.

Idea: When HTF ADX is LOW (sideways regime), price tends to revert from RSI
extremes. Long when RSI < 30 in a flat HTF; short when RSI > 70. Tight SL,
target return to the mean.

This is the "small-target / very-high-WR" candidate for BTC. Designed to be
gated by regime so it doesn't trade in trends (which would chew it up).

Long-and-short symmetric.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import rsi, adx, atr, align_htf_to_ltf, ema
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


class BtcRsiMeanRevert(Strategy):
    ltf = "M5"
    required_htfs = ("H1",)

    default_params: dict[str, Any] = {
        "rsi_period": 14,
        "rsi_oversold": 25.0,
        "rsi_overbought": 75.0,
        "htf_adx_period": 14,
        "htf_adx_max": 20.0,  # only trade in low-ADX (range) regime
        "atr_period": 14,
        "ema_target": 20,     # TP = return to LTF EMA20 (mean)
        "sl_atr_mult": 1.0,
        "min_tp_atr": 0.3,    # require some minimum TP distance to avoid noise
    }

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        if "H1" not in htfs:
            raise ValueError("BtcRsiMeanRevert requires HTF 'H1'")
        p = self.merged_params(params)

        h1 = htfs["H1"].copy()
        h1["htf_adx"] = adx(h1, int(p["htf_adx_period"]))["adx"]
        h1_feat = h1[["timestamp", "htf_adx"]]
        aligned = align_htf_to_ltf(ltf=ltf, htf=h1_feat, ltf_tf=self.ltf, htf_tf="H1",
                                   htf_cols=["htf_adx"], suffix="")

        flat_regime = aligned["htf_adx"].notna() & (aligned["htf_adx"] <= float(p["htf_adx_max"]))

        r = rsi(ltf["close"], int(p["rsi_period"]))
        a = atr(ltf, int(p["atr_period"]))
        mean_target = ema(ltf["close"], int(p["ema_target"]))

        # Cross from oversold up / overbought down — signal AFTER reversal print
        oversold_prev = r.shift(1) < float(p["rsi_oversold"])
        overbought_prev = r.shift(1) > float(p["rsi_overbought"])

        long_signal = (
            flat_regime
            & oversold_prev
            & (r >= float(p["rsi_oversold"]))       # cross back above threshold
            & (ltf["close"] > ltf["open"])
            & a.notna()
            & mean_target.notna()
            & (mean_target - ltf["close"] > float(p["min_tp_atr"]) * a)
        )
        short_signal = (
            flat_regime
            & overbought_prev
            & (r <= float(p["rsi_overbought"]))
            & (ltf["close"] < ltf["open"])
            & a.notna()
            & mean_target.notna()
            & (ltf["close"] - mean_target > float(p["min_tp_atr"]) * a)
        )

        sigs = empty_signals(ltf)
        sl_dist = float(p["sl_atr_mult"]) * a

        if long_signal.any():
            sigs.loc[long_signal, "action"] = "enter_long"
            sigs.loc[long_signal, "sl"] = (ltf["close"] - sl_dist).loc[long_signal].values
            sigs.loc[long_signal, "tp"] = mean_target.loc[long_signal].values  # target = mean
        if short_signal.any():
            sigs.loc[short_signal, "action"] = "enter_short"
            sigs.loc[short_signal, "sl"] = (ltf["close"] + sl_dist).loc[short_signal].values
            sigs.loc[short_signal, "tp"] = mean_target.loc[short_signal].values

        validate_signals(sigs, len(ltf))
        return StrategyResult(signals=sigs)
