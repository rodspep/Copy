"""Two strategies replicating UG Trading's two distinct methods.

Discovered by verifying UG signals on clean PAXG data (May 26-29 2026):

  METHOD A — "PP2 / Scalp" (tight TP, R:R ~0.5)
      Mechanical edge = mean-reversion fade + limit entry at a pullback extreme
      + very tight TP (TP touched easily) + skip the runaway moves. High WR
      (~86% in-sample) but small per-trade R. This file's `XauScalpFade`.

  METHOD B — "PRI GOLD / Signals" (wide TP, R:R ~1.0-1.5)
      Deep limit entry at a swing extreme (buy a deep pullback / sell a deep
      rally) then ride a larger move. Lower WR (~50% at R:R 1.5) but big winners.
      This file's `XauDeepPullback`.

Both use R:R-relative TP geometry (tp = entry ± tp_rr × |entry - sl|), which is
the only TP geometry that produced positive expectancy in 3-year walk-forward.

Both are no-lookahead (signal at bar i reads only bars ≤ i + aligned HTF).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import ema, atr, rsi, align_htf_to_ltf, swings
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


# =============================================================================
# METHOD A — Scalp fade (mean-reversion, tight TP)
# =============================================================================

class XauScalpFade(Strategy):
    """Fade short-term extremes, tight TP (R:R ~0.5). UG 'PP2/Scalp' analog.

    Long when price is stretched BELOW a reference (oversold) and prints a
    bullish reversal candle — i.e. buy the dip for a quick bounce. Mirror short.
    SL beyond the dip, TP at a tight R:R.
    """

    ltf = "M5"
    required_htfs: tuple[str, ...] = ()

    default_params: dict[str, Any] = {
        "ref_ema":          34,        # reference mean
        "stretch_atr":      0.8,       # how far below ref to be 'oversold'
        "rsi_period":       14,
        "rsi_os":           35.0,
        "rsi_ob":           65.0,
        "use_rsi":          True,
        "atr_period":       14,
        "session_filter":   True,
        "trade_start_hour": 7,
        "trade_end_hour":   18,
        "sl_atr_mult":      1.0,       # fixed SL
        "tp_rr":            0.5,       # tight TP (Method A signature)
    }

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        p = self.merged_params(params)
        a = atr(ltf, int(p["atr_period"]))
        ref = ema(ltf["close"], int(p["ref_ema"]))
        stretch = float(p["stretch_atr"]) * a

        # Oversold/overbought by distance from reference
        oversold = ltf["low"] <= (ref - stretch)
        overbought = ltf["high"] >= (ref + stretch)

        if bool(p["use_rsi"]):
            r = rsi(ltf["close"], int(p["rsi_period"]))
            oversold = oversold & (r <= float(p["rsi_os"]))
            overbought = overbought & (r >= float(p["rsi_ob"]))

        bull_rev = ltf["close"] > ltf["open"]      # reversal confirmation
        bear_rev = ltf["close"] < ltf["open"]

        if bool(p["session_filter"]):
            hr = ltf["timestamp"].dt.hour
            insess = (hr >= int(p["trade_start_hour"])) & (hr < int(p["trade_end_hour"]))
        else:
            insess = pd.Series(True, index=ltf.index)

        long_mask = oversold & bull_rev & insess & a.notna()
        short_mask = overbought & bear_rev & insess & a.notna()

        sl_d = float(p["sl_atr_mult"]) * a
        rr = float(p["tp_rr"])
        sigs = empty_signals(ltf)
        if long_mask.any():
            entry = ltf["close"]
            sl = entry - sl_d
            tp = entry + rr * (entry - sl)
            sigs.loc[long_mask, "action"] = "enter_long"
            sigs.loc[long_mask, "sl"] = sl.loc[long_mask].values
            sigs.loc[long_mask, "tp"] = tp.loc[long_mask].values
        if short_mask.any():
            entry = ltf["close"]
            sl = entry + sl_d
            tp = entry - rr * (sl - entry)
            sigs.loc[short_mask, "action"] = "enter_short"
            sigs.loc[short_mask, "sl"] = sl.loc[short_mask].values
            sigs.loc[short_mask, "tp"] = tp.loc[short_mask].values

        validate_signals(sigs, len(ltf))
        return StrategyResult(signals=sigs)


# =============================================================================
# METHOD B — Deep pullback swing (wide TP)
# =============================================================================

class XauDeepPullback(Strategy):
    """Limit-style deep pullback to a swing extreme, wide TP (R:R ~1.5).

    UG 'PRI GOLD / Signals' analog. In an HTF up-bias, buy when price pulls
    back to (near) a recent M5 swing low; SL below the swing, TP at a wide R:R.
    Mirror for short. Big winners, lower WR.
    """

    ltf = "M5"
    required_htfs = ("H1",)

    default_params: dict[str, Any] = {
        "h1_ema_fast":      34,
        "h1_ema_slow":      89,
        "require_h1_trend": True,      # only trade with HTF bias
        "swing_left":       5,
        "swing_right":      5,
        "touch_atr":        0.2,       # how close to the swing to count as 'reached'
        "atr_period":       14,
        "session_filter":   True,
        "trade_start_hour": 7,
        "trade_end_hour":   18,
        "sl_buffer_atr":    0.5,       # SL beyond the swing
        "tp_rr":            1.5,       # wide TP (Method B signature)
        "min_sl_atr":       0.5,
    }

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        if "H1" not in htfs:
            raise ValueError("XauDeepPullback requires HTF 'H1'")
        p = self.merged_params(params)
        a = atr(ltf, int(p["atr_period"]))

        # HTF bias
        if bool(p["require_h1_trend"]):
            h1 = htfs["H1"].copy()
            h1["__ef"] = ema(h1["close"], int(p["h1_ema_fast"]))
            h1["__es"] = ema(h1["close"], int(p["h1_ema_slow"]))
            al = align_htf_to_ltf(ltf=ltf, htf=h1[["timestamp", "__ef", "__es"]],
                                  ltf_tf=self.ltf, htf_tf="H1",
                                  htf_cols=["__ef", "__es"], suffix="")
            h1_long = al["__ef"] > al["__es"]
            h1_short = al["__ef"] < al["__es"]
        else:
            h1_long = pd.Series(True, index=ltf.index)
            h1_short = pd.Series(True, index=ltf.index)

        sw = swings(ltf, left=int(p["swing_left"]), right=int(p["swing_right"]))
        sw_lo = sw["swing_low_price"]
        sw_hi = sw["swing_high_price"]
        tol = float(p["touch_atr"]) * a

        # Pull back to (near) the swing extreme
        long_touch = sw_lo.notna() & (ltf["low"] <= sw_lo + tol) & (ltf["low"] >= sw_lo - tol * 3) \
            & (ltf["close"] > ltf["open"])
        short_touch = sw_hi.notna() & (ltf["high"] >= sw_hi - tol) & (ltf["high"] <= sw_hi + tol * 3) \
            & (ltf["close"] < ltf["open"])

        if bool(p["session_filter"]):
            hr = ltf["timestamp"].dt.hour
            insess = (hr >= int(p["trade_start_hour"])) & (hr < int(p["trade_end_hour"]))
        else:
            insess = pd.Series(True, index=ltf.index)

        long_mask = h1_long & long_touch & insess & a.notna() & sw_lo.notna()
        short_mask = h1_short & short_touch & insess & a.notna() & sw_hi.notna()

        sl_buf = float(p["sl_buffer_atr"]) * a
        min_sl = float(p["min_sl_atr"]) * a
        rr = float(p["tp_rr"])
        sigs = empty_signals(ltf)
        if long_mask.any():
            entry = ltf["close"]
            raw_sl = sw_lo - sl_buf
            sl = np.minimum(raw_sl, entry - min_sl)
            tp = entry + rr * (entry - sl)
            sigs.loc[long_mask, "action"] = "enter_long"
            sigs.loc[long_mask, "sl"] = sl.loc[long_mask].values
            sigs.loc[long_mask, "tp"] = tp.loc[long_mask].values
        if short_mask.any():
            entry = ltf["close"]
            raw_sl = sw_hi + sl_buf
            sl = np.maximum(raw_sl, entry + min_sl)
            tp = entry - rr * (sl - entry)
            sigs.loc[short_mask, "action"] = "enter_short"
            sigs.loc[short_mask, "sl"] = sl.loc[short_mask].values
            sigs.loc[short_mask, "tp"] = tp.loc[short_mask].values

        # drop degenerate
        bad = sigs["action"].isin(["enter_long", "enter_short"]) & (sigs["sl"].isna() | sigs["tp"].isna())
        if bad.any():
            sigs.loc[bad, "action"] = "hold"; sigs.loc[bad, "sl"] = np.nan; sigs.loc[bad, "tp"] = np.nan

        validate_signals(sigs, len(ltf))
        return StrategyResult(signals=sigs)
