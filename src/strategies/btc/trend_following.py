"""BTC trend-following with regime gate.

Idea: BTC has long persistent trends. Trade them with HTF (H1) EMA200 direction
filter + ADX strength gate; entry on M5 pullback to EMA20. ATR SL/TP. The
regime gate stops us trading in choppy sideways regimes where this method bleeds.

Long-and-short symmetric.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import ema, adx, atr, align_htf_to_ltf
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


class BtcTrendFollowing(Strategy):
    ltf = "M5"
    required_htfs = ("H1",)

    default_params: dict[str, Any] = {
        "htf_ema": 200,
        "htf_adx_period": 14,
        "htf_adx_min": 25.0,
        "ltf_ema_pullback": 20,
        "ltf_atr_period": 14,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 3.0,
        "pullback_proximity_atr": 0.3,
    }

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        if "H1" not in htfs:
            raise ValueError("BtcTrendFollowing requires HTF 'H1'")
        p = self.merged_params(params)

        h1 = htfs["H1"].copy()
        h1["htf_ema"] = ema(h1["close"], int(p["htf_ema"]))
        h1_adx = adx(h1, int(p["htf_adx_period"]))
        h1["htf_adx"] = h1_adx["adx"]
        h1_feat = h1[["timestamp", "htf_ema", "htf_adx", "close"]].rename(columns={"close": "htf_close"})

        aligned = align_htf_to_ltf(
            ltf=ltf, htf=h1_feat, ltf_tf=self.ltf, htf_tf="H1",
            htf_cols=["htf_ema", "htf_adx", "htf_close"], suffix="",
        )

        long_regime = (
            aligned["htf_ema"].notna()
            & (aligned["htf_close"] > aligned["htf_ema"])
            & (aligned["htf_adx"] >= float(p["htf_adx_min"]))
        )
        short_regime = (
            aligned["htf_ema"].notna()
            & (aligned["htf_close"] < aligned["htf_ema"])
            & (aligned["htf_adx"] >= float(p["htf_adx_min"]))
        )

        ltf_ema_v = ema(ltf["close"], int(p["ltf_ema_pullback"]))
        ltf_atr_v = atr(ltf, int(p["ltf_atr_period"]))
        prox = float(p["pullback_proximity_atr"]) * ltf_atr_v

        long_signal = (
            long_regime
            & ltf_atr_v.notna()
            & (ltf["low"] <= ltf_ema_v + prox)
            & (ltf["close"] > ltf_ema_v)
            & (ltf["close"] > ltf["open"])
        )
        short_signal = (
            short_regime
            & ltf_atr_v.notna()
            & (ltf["high"] >= ltf_ema_v - prox)
            & (ltf["close"] < ltf_ema_v)
            & (ltf["close"] < ltf["open"])
        )

        sigs = empty_signals(ltf)
        sl_dist = float(p["sl_atr_mult"]) * ltf_atr_v
        tp_dist = float(p["tp_atr_mult"]) * ltf_atr_v

        if long_signal.any():
            sigs.loc[long_signal, "action"] = "enter_long"
            sigs.loc[long_signal, "sl"] = (ltf["close"] - sl_dist).loc[long_signal].values
            sigs.loc[long_signal, "tp"] = (ltf["close"] + tp_dist).loc[long_signal].values
        if short_signal.any():
            sigs.loc[short_signal, "action"] = "enter_short"
            sigs.loc[short_signal, "sl"] = (ltf["close"] + sl_dist).loc[short_signal].values
            sigs.loc[short_signal, "tp"] = (ltf["close"] - tp_dist).loc[short_signal].values

        validate_signals(sigs, len(ltf))
        return StrategyResult(signals=sigs)
