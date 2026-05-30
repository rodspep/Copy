"""XAU high-reaction-level bounce strategy.

User concept: trade a price level on the LTF where price has *reacted many
times* (lots of swing-pivot rejections clustered there). Enter a bounce off
that level in the direction of the higher-timeframe trend.

Pipeline:

  STAGE 1 — HTF (H1) trend reference (REFERENCE ONLY, like ltf_ob_entry)
      H1 EMA fast>slow → long bias; <  → short bias. Optional (require_h1_trend).

  STAGE 2 — Reaction-level detection (the new concept)
      indicators.reaction.reaction_levels() counts confirmed M5 swing pivots
      clustered within band of the current bar's low/high over a trailing
      window. A high count = a level price keeps reversing at.
        - long:  bar.low at a support cluster (react_low_count >= min_reactions)
        - short: bar.high at a resistance cluster (react_high_count >= min)

  STAGE 3 — Confirmation candle
      close>open (long) / close<open (short). Optional.

  FILTERS (the "remaining options" — each toggleable so optimizer can mix in):
    - require_m5_trend : M5 SMC structure trend agrees with direction
    - session_filter   : London + early-NY hours only
    - require_sr       : classical S/R confluence (PDH/PDL, weekly pivot, round)

  SL: reaction-level center − buffer×ATR (structural, anchored to the cluster).
  TP: R:R-relative — tp = entry ± tp_rr × |entry − sl|  (proven geometry).

Long-and-short symmetric.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import ema, atr, align_htf_to_ltf, swings, structure_breaks
from src.indicators.reaction import reaction_levels
from src.indicators.sr import prior_day_high_low, weekly_pivot, has_sr_confluence
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


class XauReactionLevel(Strategy):
    """Bounce off high-reaction LTF price levels in HTF-trend direction."""

    ltf = "M5"
    required_htfs = ("H1",)

    default_params: dict[str, Any] = {
        # HTF trend reference
        "h1_ema_fast":          34,
        "h1_ema_slow":          89,
        "require_h1_trend":     True,

        # Reaction-level detection
        "swing_left":           3,
        "swing_right":          3,
        "lookback":             500,        # trailing M5 bars (~1.7 days)
        "band_atr_mult":        0.25,       # cluster band half-width
        "min_reactions":        3,          # min pivots in band to count as a level

        # Confirmation + filters (the "remaining options")
        "require_confirm_candle": True,
        "require_m5_trend":     False,
        "session_filter":       True,
        "trade_start_hour":     7,
        "trade_end_hour":       18,
        "require_sr":           False,
        "sr_round_step":        10.0,
        "sr_tolerance_atr":     1.0,

        # Geometry
        "atr_period":           14,
        "sl_buffer_atr":        0.3,
        "tp_rr":                2.0,
        "min_sl_atr":           0.3,
    }

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        if "H1" not in htfs:
            raise ValueError("XauReactionLevel requires HTF 'H1'")
        p = self.merged_params(params)

        a = atr(ltf, int(p["atr_period"]))

        # ---- STAGE 1: HTF trend reference ----
        if bool(p["require_h1_trend"]):
            h1 = htfs["H1"].copy()
            f = int(p["h1_ema_fast"]); s = int(p["h1_ema_slow"])
            h1["__ef"] = ema(h1["close"], f)
            h1["__es"] = ema(h1["close"], s)
            aligned = align_htf_to_ltf(
                ltf=ltf, htf=h1[["timestamp", "__ef", "__es"]], ltf_tf=self.ltf,
                htf_tf="H1", htf_cols=["__ef", "__es"], suffix="",
            )
            h1_long = aligned["__ef"] > aligned["__es"]
            h1_short = aligned["__ef"] < aligned["__es"]
        else:
            h1_long = pd.Series(True, index=ltf.index)
            h1_short = pd.Series(True, index=ltf.index)

        # ---- STAGE 2: reaction levels ----
        sw = swings(ltf, left=int(p["swing_left"]), right=int(p["swing_right"]))
        rl = reaction_levels(
            ltf, sw, atr_series=a,
            lookback=int(p["lookback"]), band_atr_mult=float(p["band_atr_mult"]),
        )
        min_r = int(p["min_reactions"])
        at_support = rl["react_low_count"] >= min_r
        at_resistance = rl["react_high_count"] >= min_r
        support_level = rl["react_low_level"]
        resistance_level = rl["react_high_level"]

        # ---- STAGE 3: confirmation candle ----
        if bool(p["require_confirm_candle"]):
            bull_c = ltf["close"] > ltf["open"]
            bear_c = ltf["close"] < ltf["open"]
        else:
            bull_c = pd.Series(True, index=ltf.index)
            bear_c = pd.Series(True, index=ltf.index)

        # ---- FILTERS ----
        if bool(p["require_m5_trend"]):
            st = structure_breaks(ltf, sw)
            m5_long_ok = st["trend"] == 1
            m5_short_ok = st["trend"] == -1
        else:
            m5_long_ok = pd.Series(True, index=ltf.index)
            m5_short_ok = pd.Series(True, index=ltf.index)

        if bool(p["session_filter"]):
            hour = ltf["timestamp"].dt.hour
            in_session = (hour >= int(p["trade_start_hour"])) & (hour < int(p["trade_end_hour"]))
        else:
            in_session = pd.Series(True, index=ltf.index)

        if bool(p["require_sr"]):
            pdh_df = prior_day_high_low(ltf)
            wk_df = weekly_pivot(ltf)
            sr = has_sr_confluence(
                close=ltf["close"], atr_series=a,
                pdh=pdh_df["pdh"], pdl=pdh_df["pdl"],
                wkly_p=wk_df["wkly_p"], wkly_r1=wk_df["wkly_r1"], wkly_s1=wk_df["wkly_s1"],
                round_step=float(p["sr_round_step"]),
                tolerance_atr=float(p["sr_tolerance_atr"]),
            )
            sr_long = sr["long_sr"]; sr_short = sr["short_sr"]
        else:
            sr_long = pd.Series(True, index=ltf.index)
            sr_short = pd.Series(True, index=ltf.index)

        # ---- Compose ----
        long_mask = (
            h1_long & at_support & bull_c & m5_long_ok & in_session & sr_long
            & a.notna() & support_level.notna()
        )
        short_mask = (
            h1_short & at_resistance & bear_c & m5_short_ok & in_session & sr_short
            & a.notna() & resistance_level.notna()
        )

        # ---- SL/TP ----
        sl_buf = float(p["sl_buffer_atr"]) * a
        min_sl = float(p["min_sl_atr"]) * a
        tp_rr = float(p["tp_rr"])

        sigs = empty_signals(ltf)

        if long_mask.any():
            entry = ltf["close"]
            raw_sl = support_level - sl_buf          # below the reaction cluster
            tight_sl = entry - min_sl
            final_sl = np.minimum(raw_sl, tight_sl)
            sl_dist = entry - final_sl
            tp = entry + tp_rr * sl_dist
            sigs.loc[long_mask, "action"] = "enter_long"
            sigs.loc[long_mask, "sl"] = final_sl.loc[long_mask].values
            sigs.loc[long_mask, "tp"] = tp.loc[long_mask].values

        if short_mask.any():
            entry = ltf["close"]
            raw_sl = resistance_level + sl_buf       # above the reaction cluster
            tight_sl = entry + min_sl
            final_sl = np.maximum(raw_sl, tight_sl)
            sl_dist = final_sl - entry
            tp = entry - tp_rr * sl_dist
            sigs.loc[short_mask, "action"] = "enter_short"
            sigs.loc[short_mask, "sl"] = final_sl.loc[short_mask].values
            sigs.loc[short_mask, "tp"] = tp.loc[short_mask].values

        bad = sigs["action"].isin(["enter_long", "enter_short"]) & (
            sigs["sl"].isna() | sigs["tp"].isna()
        )
        if bad.any():
            sigs.loc[bad, "action"] = "hold"
            sigs.loc[bad, "sl"] = np.nan
            sigs.loc[bad, "tp"] = np.nan

        validate_signals(sigs, len(ltf))
        return StrategyResult(signals=sigs, debug={
            "react_low_count": rl["react_low_count"],
            "react_high_count": rl["react_high_count"],
        })
