"""BTC SMC order-block retest (mirrors XAU variant, tuned for crypto microstructure).

The SMC concepts work on any liquid market. On BTC we typically use a higher
HTF (H4 vs M15 for XAU) because BTC trades 24/7 and the M15 OB count gets noisy.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import atr, swings, structure_breaks, order_blocks, align_htf_to_ltf
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


class BtcSmcOrderBlock(Strategy):
    ltf = "M5"
    required_htfs = ("H1",)  # H1 for BTC (vs M15 for XAU)

    default_params: dict[str, Any] = {
        "htf_swing_left": 3,
        "htf_swing_right": 3,
        "ltf_atr_period": 14,
        "sl_atr_mult": 1.0,
        "tp_atr_mult": 3.0,        # BTC moves bigger — RR can be higher
        "ob_proximity_pct": 0.002, # 0.2% of OB (BTC has bigger absolute moves)
    }

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        if "H1" not in htfs:
            raise ValueError("BtcSmcOrderBlock requires HTF 'H1'")
        p = self.merged_params(params)

        h1 = htfs["H1"].copy()
        sw = swings(h1, left=int(p["htf_swing_left"]), right=int(p["htf_swing_right"]))
        st = structure_breaks(h1, sw)
        ob = order_blocks(h1, st)

        h1_feat = h1[["timestamp"]].copy()
        h1_feat["bull_ob_top"] = ob["bull_ob_top"].values
        h1_feat["bull_ob_bot"] = ob["bull_ob_bot"].values
        h1_feat["bear_ob_top"] = ob["bear_ob_top"].values
        h1_feat["bear_ob_bot"] = ob["bear_ob_bot"].values
        h1_feat["smc_trend"] = st["trend"].values

        aligned = align_htf_to_ltf(
            ltf=ltf, htf=h1_feat, ltf_tf=self.ltf, htf_tf="H1",
            htf_cols=["bull_ob_top", "bull_ob_bot", "bear_ob_top", "bear_ob_bot", "smc_trend"],
            suffix="",
        )

        ltf_atr_v = atr(ltf, int(p["ltf_atr_period"]))
        prox = float(p["ob_proximity_pct"])

        bull_top = aligned["bull_ob_top"]
        bull_bot = aligned["bull_ob_bot"]
        long_retest = (
            (aligned["smc_trend"] == 1)
            & bull_top.notna()
            & ltf_atr_v.notna()
            & (ltf["low"] <= bull_top * (1 + prox))
            & (ltf["low"] >= bull_bot * (1 - prox))
            & (ltf["close"] > ltf["open"])
        )

        bear_top = aligned["bear_ob_top"]
        bear_bot = aligned["bear_ob_bot"]
        short_retest = (
            (aligned["smc_trend"] == -1)
            & bear_top.notna()
            & ltf_atr_v.notna()
            & (ltf["high"] >= bear_bot * (1 - prox))
            & (ltf["high"] <= bear_top * (1 + prox))
            & (ltf["close"] < ltf["open"])
        )

        sigs = empty_signals(ltf)
        sl_dist = float(p["sl_atr_mult"]) * ltf_atr_v
        tp_dist = float(p["tp_atr_mult"]) * ltf_atr_v

        if long_retest.any():
            sigs.loc[long_retest, "action"] = "enter_long"
            sigs.loc[long_retest, "sl"] = (bull_bot - sl_dist).loc[long_retest].values
            sigs.loc[long_retest, "tp"] = (ltf["close"] + tp_dist).loc[long_retest].values
        if short_retest.any():
            sigs.loc[short_retest, "action"] = "enter_short"
            sigs.loc[short_retest, "sl"] = (bear_top + sl_dist).loc[short_retest].values
            sigs.loc[short_retest, "tp"] = (ltf["close"] - tp_dist).loc[short_retest].values

        validate_signals(sigs, len(ltf))
        return StrategyResult(signals=sigs)
