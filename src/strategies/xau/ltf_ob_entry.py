"""XAU LTF Order-Block entry with HTF as reference-only bias.

The previous cascade `XauMtfSmcEntry` anchors SL to H4 OB low (very wide,
~3×ATR). User's hypothesis: try M5 OB instead — SL much tighter (~0.5-1×ATR),
HTF only used as trend reference (not gated/cascaded).

Pipeline:

  STAGE 1 — HTF bias (REFERENCE ONLY)
      H1 EMA fast > slow (long) or < (short). Trade only in HTF direction.
      No cascading zone requirement — H1 is just a trend filter.

  STAGE 2 — M5 OB detection
      Compute M5 swings + structure_breaks + order_blocks (same SMC primitives
      used elsewhere, just on LTF instead of HTF). The most recent M5 bull OB
      is the demand zone for longs; bear OB for shorts.

  STAGE 3 — M5 OB retest entry
      Price low touches M5 bull OB top (long) or high touches M5 bear OB bot
      (short), plus confirmation candle (close in trade direction).

  SL: M5 OB extreme + small ATR buffer (TIGHT, ~0.5-1×ATR)
  TP: R:R-relative — tp = entry + tp_rr × (entry - sl), tp_rr default 2.0

  Optional gates:
    - Session filter (London + early NY)
    - Require S/R confluence
    - M5 SMC trend agreement (current bar's trend matches direction)

Expected: signal frequency ↑ (M5 OBs more frequent than H4) and SL distance ↓
(tighter), so each loss is small but tradeable per session.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import (
    ema, atr, align_htf_to_ltf,
    swings, structure_breaks, order_blocks,
)
from src.indicators.sr import prior_day_high_low, weekly_pivot, has_sr_confluence
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


class XauLtfObEntry(Strategy):
    """M5 OB retest entry + H1 trend reference + tight SL + R:R-relative TP."""

    ltf = "M5"
    required_htfs = ("H1",)

    default_params: dict[str, Any] = {
        # HTF trend reference (REFERENCE only, not cascade-gated zone)
        "h1_ema_fast":          34,
        "h1_ema_slow":          89,
        "require_h1_trend":     True,    # if False, no HTF gate at all

        # M5 OB detection
        "m5_swing_left":        3,
        "m5_swing_right":       3,
        "ob_proximity_pct":     0.001,   # 0.1% tolerance for OB retest

        # Match LTF SMC trend at entry (light gate)
        "require_m5_trend":     True,    # M5 BOS trend must match direction

        # Optional gates
        "session_filter":       True,
        "trade_start_hour":     7,
        "trade_end_hour":       18,
        "require_sr":           False,
        "sr_round_step":        10.0,
        "sr_tolerance_atr":     1.0,

        # SL/TP
        "atr_period":           14,
        "sl_buffer_atr":        0.3,     # SL = OB extreme ± buffer × M5 ATR (tight)
        "tp_rr":                2.0,     # TP = entry ± tp_rr × |entry - sl|
        "min_sl_atr":           0.3,     # min SL distance to avoid micro stops
    }

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        if "H1" not in htfs:
            raise ValueError("XauLtfObEntry requires HTF 'H1'")
        p = self.merged_params(params)

        # ---- STAGE 1: H1 trend reference ----
        if bool(p["require_h1_trend"]):
            h1 = htfs["H1"].copy()
            f = int(p["h1_ema_fast"]); s = int(p["h1_ema_slow"])
            h1["__ef"] = ema(h1["close"], f)
            h1["__es"] = ema(h1["close"], s)
            feat = h1[["timestamp", "__ef", "__es"]]
            aligned = align_htf_to_ltf(
                ltf=ltf, htf=feat, ltf_tf=self.ltf, htf_tf="H1",
                htf_cols=["__ef", "__es"], suffix="",
            )
            h1_long = aligned["__ef"] > aligned["__es"]
            h1_short = aligned["__ef"] < aligned["__es"]
        else:
            h1_long = pd.Series(True, index=ltf.index)
            h1_short = pd.Series(True, index=ltf.index)

        # ---- STAGE 2: M5 OB detection ----
        sw = swings(ltf, left=int(p["m5_swing_left"]), right=int(p["m5_swing_right"]))
        st = structure_breaks(ltf, sw)
        ob = order_blocks(ltf, st)
        bull_top = ob["bull_ob_top"]; bull_bot = ob["bull_ob_bot"]
        bear_top = ob["bear_ob_top"]; bear_bot = ob["bear_ob_bot"]

        # M5 SMC trend (light gate)
        if bool(p["require_m5_trend"]):
            m5_trend = st["trend"]
            m5_long_ok = m5_trend == 1
            m5_short_ok = m5_trend == -1
        else:
            m5_long_ok = pd.Series(True, index=ltf.index)
            m5_short_ok = pd.Series(True, index=ltf.index)

        # ---- STAGE 3: M5 OB retest ----
        a = atr(ltf, int(p["atr_period"]))
        prox = float(p["ob_proximity_pct"])

        # Long retest: low touches bull OB top (price drops into OB zone)
        long_retest = (
            bull_top.notna() & bull_bot.notna() & a.notna()
            & (ltf["low"] <= bull_top * (1 + prox))
            & (ltf["low"] >= bull_bot * (1 - prox))
            & (ltf["close"] > ltf["open"])     # bullish confirmation candle
        )
        short_retest = (
            bear_top.notna() & bear_bot.notna() & a.notna()
            & (ltf["high"] >= bear_bot * (1 - prox))
            & (ltf["high"] <= bear_top * (1 + prox))
            & (ltf["close"] < ltf["open"])
        )

        # ---- Optional gates ----
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
        long_mask = h1_long & m5_long_ok & long_retest & in_session & sr_long
        short_mask = h1_short & m5_short_ok & short_retest & in_session & sr_short

        # ---- SL/TP ----
        sl_buf = float(p["sl_buffer_atr"]) * a
        min_sl = float(p["min_sl_atr"]) * a
        tp_rr = float(p["tp_rr"])

        sigs = empty_signals(ltf)

        if long_mask.any():
            # SL = OB bottom - buffer (or entry - min_sl, whichever wider)
            raw_sl = bull_bot - sl_buf
            entry_price = ltf["close"]
            tight_sl = entry_price - min_sl
            final_sl = np.minimum(raw_sl, tight_sl)
            sl_dist = entry_price - final_sl
            tp = entry_price + tp_rr * sl_dist

            sigs.loc[long_mask, "action"] = "enter_long"
            sigs.loc[long_mask, "sl"] = final_sl.loc[long_mask].values
            sigs.loc[long_mask, "tp"] = tp.loc[long_mask].values

        if short_mask.any():
            raw_sl = bear_top + sl_buf
            entry_price = ltf["close"]
            tight_sl = entry_price + min_sl
            final_sl = np.maximum(raw_sl, tight_sl)
            sl_dist = final_sl - entry_price
            tp = entry_price - tp_rr * sl_dist

            sigs.loc[short_mask, "action"] = "enter_short"
            sigs.loc[short_mask, "sl"] = final_sl.loc[short_mask].values
            sigs.loc[short_mask, "tp"] = tp.loc[short_mask].values

        # Drop degenerate
        bad = sigs["action"].isin(["enter_long", "enter_short"]) & (
            sigs["sl"].isna() | sigs["tp"].isna()
        )
        if bad.any():
            sigs.loc[bad, "action"] = "hold"
            sigs.loc[bad, "sl"] = np.nan
            sigs.loc[bad, "tp"] = np.nan

        validate_signals(sigs, len(ltf))
        return StrategyResult(signals=sigs)


class XauM15ObEntry(XauLtfObEntry):
    """M15 OB retest entry (vs XauLtfObEntry which uses M5 OB).

    User hypothesis: smaller TF OB tightens SL but adds noise; mid TF M15 may
    balance frequency vs cleanliness. Same logic as parent, just ltf=M15.
    """

    ltf = "M15"
    required_htfs = ("H1",)

