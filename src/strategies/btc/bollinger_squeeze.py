"""BTC Bollinger-squeeze + breakout.

Idea: When BB bandwidth contracts below a percentile threshold, volatility is
compressed. A breakout out of the bands often kicks off a meaningful move.
Enter at break direction; SL inside the bands; TP at ATR multiple.

Long-and-short symmetric.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import bollinger, atr
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


class BtcBollingerSqueeze(Strategy):
    ltf = "M5"
    required_htfs: tuple[str, ...] = ()

    default_params: dict[str, Any] = {
        "bb_period": 20,
        "bb_std": 2.0,
        "squeeze_lookback": 120,  # bars used for the bandwidth percentile
        "squeeze_pctile": 0.20,   # bandwidth must be in lowest 20% over lookback
        "atr_period": 14,
        "sl_atr_mult": 1.0,
        "tp_atr_mult": 2.5,
    }

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        p = self.merged_params(params)
        b = bollinger(ltf["close"], period=int(p["bb_period"]), n_std=float(p["bb_std"]))
        a = atr(ltf, int(p["atr_period"]))

        bw = b["bandwidth"]
        # Rolling quantile threshold (causal — uses past `squeeze_lookback` bars).
        lookback = int(p["squeeze_lookback"])
        bw_thresh = bw.rolling(window=lookback, min_periods=lookback).quantile(float(p["squeeze_pctile"]))
        in_squeeze = bw <= bw_thresh

        # The squeeze must have been present on the PRIOR bar; we look for a breakout NOW.
        prior_squeeze = in_squeeze.shift(1).fillna(False)

        long_break = prior_squeeze & (ltf["close"] > b["upper"]) & a.notna()
        short_break = prior_squeeze & (ltf["close"] < b["lower"]) & a.notna()

        sigs = empty_signals(ltf)
        sl_dist = float(p["sl_atr_mult"]) * a
        tp_dist = float(p["tp_atr_mult"]) * a

        if long_break.any():
            sigs.loc[long_break, "action"] = "enter_long"
            sigs.loc[long_break, "sl"] = (ltf["close"] - sl_dist).loc[long_break].values
            sigs.loc[long_break, "tp"] = (ltf["close"] + tp_dist).loc[long_break].values
        if short_break.any():
            sigs.loc[short_break, "action"] = "enter_short"
            sigs.loc[short_break, "sl"] = (ltf["close"] + sl_dist).loc[short_break].values
            sigs.loc[short_break, "tp"] = (ltf["close"] - tp_dist).loc[short_break].values

        validate_signals(sigs, len(ltf))
        return StrategyResult(signals=sigs)
