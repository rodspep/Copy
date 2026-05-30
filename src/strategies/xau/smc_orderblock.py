"""XAU SMC order-block retest entry.

Idea: After a BOS-up on M15 (HTF), enter long on M5 when price retests the most
recent bullish order block (the last down candle before the BOS). Mirror for
short. SL beyond the OB; TP at ATR multiple or to the next opposing structure.

Long-and-short symmetric.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import atr, swings, structure_breaks, order_blocks, align_htf_to_ltf
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


class XauSmcOrderBlock(Strategy):
    ltf = "M5"
    required_htfs = ("M15",)

    default_params: dict[str, Any] = {
        "htf_swing_left": 3,
        "htf_swing_right": 3,
        "ltf_atr_period": 14,
        "sl_atr_mult": 0.8,        # SL just beyond the OB
        "tp_atr_mult": 2.4,        # OB strategies typically run for higher RR
        "ob_proximity_pct": 0.001, # within 0.1% of OB to count as a retest
    }

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        if "M15" not in htfs:
            raise ValueError("XauSmcOrderBlock requires HTF 'M15'")
        p = self.merged_params(params)

        m15 = htfs["M15"].copy()
        sw = swings(m15, left=int(p["htf_swing_left"]), right=int(p["htf_swing_right"]))
        st = structure_breaks(m15, sw)
        ob = order_blocks(m15, st)

        # HTF features to attach onto LTF: current OB zones and current SMC trend.
        m15_feat = m15[["timestamp"]].copy()
        m15_feat["bull_ob_top"] = ob["bull_ob_top"].values
        m15_feat["bull_ob_bot"] = ob["bull_ob_bot"].values
        m15_feat["bear_ob_top"] = ob["bear_ob_top"].values
        m15_feat["bear_ob_bot"] = ob["bear_ob_bot"].values
        m15_feat["smc_trend"] = st["trend"].values

        aligned = align_htf_to_ltf(
            ltf=ltf, htf=m15_feat, ltf_tf=self.ltf, htf_tf="M15",
            htf_cols=["bull_ob_top", "bull_ob_bot", "bear_ob_top", "bear_ob_bot", "smc_trend"],
            suffix="",
        )

        ltf_atr_v = atr(ltf, int(p["ltf_atr_period"]))
        prox = float(p["ob_proximity_pct"])

        # Long retest: HTF trend up, price retests bullish OB zone (low touches OB top from above)
        bull_top = aligned["bull_ob_top"]
        bull_bot = aligned["bull_ob_bot"]
        long_retest = (
            (aligned["smc_trend"] == 1)
            & bull_top.notna()
            & ltf_atr_v.notna()
            & (ltf["low"] <= bull_top * (1 + prox))
            & (ltf["low"] >= bull_bot * (1 - prox))
            & (ltf["close"] > ltf["open"])  # bullish confirmation
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
