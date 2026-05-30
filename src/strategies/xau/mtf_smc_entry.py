"""XAU classical ICT top-down SMC entry: H4 OB → M15 FVG → M5 BOS.

The cascade (each stage strictly refines the previous):

  STAGE 1 — H4 OB  (HTF context + zone)
      Locate the most-recent bullish or bearish H4 order block (last opposing
      candle before a same-direction BOS on H4 structure). The OB zone defines:
        - BIAS: bullish OB → only long; bearish → only short.
        - ZONE: [bull_ob_bot, bull_ob_top] (or bear analog) where we may consider
                trades. Outside the zone → no signal.

  STAGE 2 — M15 FVG (MTF refinement inside zone)
      When price has TRADED INTO the H4 OB zone, require that an aligned M15
      FVG has formed AT OR INSIDE that zone within the last `m15_fvg_lookback`
      bars (default ~12 bars ≈ 3 hours). For a bullish setup we need a
      bullish FVG with bottom inside the H4 OB demand zone. The FVG zone is
      the "premium/discount" refinement of the OB.

  STAGE 3 — M5 BOS (LTF confirmation, trigger)
      With STAGE 1 + STAGE 2 already true, watch M5 for a same-direction BOS
      (close-through-swing) on a bar that is INSIDE the M15 FVG. The first
      such M5 BOS is the entry — fired at that bar's close.

This is much more selective than `XauSmcConfluence` (which gates a single-TF
event by HTF trend); each stage is a *hard prerequisite* for the next. Signal
count should drop ~10-50× vs the gated variant, with WR rising correspondingly.

SL = HTF OB extreme + buffer ATR (structural stop — invalidates the H4 thesis).
TP = ATR-based multiple (optimizer-tuned), defaults to UG-scalp R:R 0.6.

Long-and-short symmetric.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import (
    atr, align_htf_to_ltf,
    swings, structure_breaks, order_blocks, fair_value_gaps,
)
from src.indicators.sr import (
    prior_day_high_low, weekly_pivot, has_sr_confluence,
)
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


class XauMtfSmcEntry(Strategy):
    """H4 OB → M15 FVG → M5 BOS classical ICT cascade."""

    ltf = "M5"
    required_htfs = ("M15", "H4")

    default_params: dict[str, Any] = {
        # H4 OB params
        "h4_swing_left":          3,
        "h4_swing_right":         3,
        "h4_ob_max_age_bars":     30,        # H4 bars (~5 days) — OB freshness limit

        # M15 FVG params
        "m15_fvg_lookback":       12,        # how many M15 bars back to consider FVG
        "m15_fvg_min_size_atr":   0.1,       # FVG height ≥ this × M15 ATR; filters noise

        # M5 BOS params
        "m5_swing_left":          3,
        "m5_swing_right":         3,
        "m5_bos_within_bars":     20,        # require BOS within N M5 bars of zone entry

        # Confirmation extras (optional gates)
        "session_filter":         True,
        "trade_start_hour":       7,
        "trade_end_hour":         18,

        # S/R confluence gate — require ≥1 classical S/R level (PDH/PDL,
        # weekly pivot/R1/S1, nearest round number) within tolerance of close.
        # OB/FVG are SMC zones; this layer adds CLASSICAL price levels.
        "require_sr_confluence":  False,
        "sr_round_step":          10.0,       # XAU round levels every 10$
        "sr_tolerance_atr":       1.0,        # within 1×ATR of close = "near"

        # ATR + SL/TP
        "atr_period":             14,
        "sl_buffer_atr":          0.3,       # SL = OB extreme + buffer × M5 ATR
        # TP geometry mode:
        #   'atr_absolute'  = tp distance = tp_atr_mult × M5 ATR  (original)
        #   'rr_relative'   = tp distance = tp_rr × (entry - sl distance) — UG premium style
        "tp_mode":                "rr_relative",
        "tp_atr_mult":            0.6,
        "tp_rr":                  2.0,        # used when tp_mode == 'rr_relative'
    }

    # ------------------------------------------------------------------
    # Stage builders
    # ------------------------------------------------------------------

    @staticmethod
    def _h4_ob_zones(h4: pd.DataFrame, sw_left: int, sw_right: int,
                     max_age: int, atr_period: int
                     ) -> pd.DataFrame:
        """Return per-H4-bar [bull_ob_top, bull_ob_bot, bull_ob_age,
        bear_ob_top, bear_ob_bot, bear_ob_age]. Ages in H4 bars since OB formed.
        OB invalidated → NaN.
        """
        sw = swings(h4, left=sw_left, right=sw_right)
        st = structure_breaks(h4, sw)
        ob = order_blocks(h4, st)
        n = len(h4)
        bull_age = np.full(n, np.inf)
        bear_age = np.full(n, np.inf)
        bull_top = ob["bull_ob_top"].to_numpy()
        bull_bot = ob["bull_ob_bot"].to_numpy()
        bear_top = ob["bear_ob_top"].to_numpy()
        bear_bot = ob["bear_ob_bot"].to_numpy()
        # Compute age: bars since the most recent OB origin.
        last_bull = -1
        last_bear = -1
        for i in range(n):
            if not np.isnan(bull_top[i]) and (last_bull < 0 or bull_top[i] != bull_top[last_bull]
                                              or bull_bot[i] != bull_bot[last_bull]):
                last_bull = i
            if not np.isnan(bear_top[i]) and (last_bear < 0 or bear_top[i] != bear_top[last_bear]
                                              or bear_bot[i] != bear_bot[last_bear]):
                last_bear = i
            if last_bull >= 0:
                bull_age[i] = i - last_bull
            if last_bear >= 0:
                bear_age[i] = i - last_bear

        # Invalidate stale OBs
        bull_top = np.where(bull_age <= max_age, bull_top, np.nan)
        bull_bot = np.where(bull_age <= max_age, bull_bot, np.nan)
        bear_top = np.where(bear_age <= max_age, bear_top, np.nan)
        bear_bot = np.where(bear_age <= max_age, bear_bot, np.nan)

        out = pd.DataFrame({"timestamp": h4["timestamp"].reset_index(drop=True)})
        out["h4_bull_top"] = bull_top
        out["h4_bull_bot"] = bull_bot
        out["h4_bull_age"] = bull_age
        out["h4_bear_top"] = bear_top
        out["h4_bear_bot"] = bear_bot
        out["h4_bear_age"] = bear_age
        return out

    @staticmethod
    def _m15_fvg_features(m15: pd.DataFrame, lookback: int, min_size_atr: float,
                          atr_period: int
                          ) -> pd.DataFrame:
        """Return per-M15-bar bull/bear FVG zone + bars-since-FVG for filtering."""
        fv = fair_value_gaps(m15)
        a = atr(m15, atr_period)
        # Bars since each kind of FVG printed
        n = len(m15)
        bull_printed = fv["bull_fvg"].to_numpy()
        bear_printed = fv["bear_fvg"].to_numpy()
        bull_size = (fv["bull_fvg_top"] - fv["bull_fvg_bot"]).to_numpy()
        bear_size = (fv["bear_fvg_top"] - fv["bear_fvg_bot"]).to_numpy()
        a_arr = a.to_numpy()

        # Filter: FVG size must be >= min_size_atr * ATR
        bull_ok = bull_printed & (bull_size >= min_size_atr * a_arr)
        bear_ok = bear_printed & (bear_size >= min_size_atr * a_arr)

        # Carry forward the most recent qualifying FVG with bar-age
        last_bull_idx = -1
        last_bear_idx = -1
        bull_top = np.full(n, np.nan)
        bull_bot = np.full(n, np.nan)
        bear_top = np.full(n, np.nan)
        bear_bot = np.full(n, np.nan)
        bull_age = np.full(n, np.inf)
        bear_age = np.full(n, np.inf)
        for i in range(n):
            if bull_ok[i]:
                last_bull_idx = i
            if bear_ok[i]:
                last_bear_idx = i
            if last_bull_idx >= 0:
                bull_top[i] = fv["bull_fvg_top"].iloc[last_bull_idx]
                bull_bot[i] = fv["bull_fvg_bot"].iloc[last_bull_idx]
                bull_age[i] = i - last_bull_idx
            if last_bear_idx >= 0:
                bear_top[i] = fv["bear_fvg_top"].iloc[last_bear_idx]
                bear_bot[i] = fv["bear_fvg_bot"].iloc[last_bear_idx]
                bear_age[i] = i - last_bear_idx

        bull_top = np.where(bull_age <= lookback, bull_top, np.nan)
        bull_bot = np.where(bull_age <= lookback, bull_bot, np.nan)
        bear_top = np.where(bear_age <= lookback, bear_top, np.nan)
        bear_bot = np.where(bear_age <= lookback, bear_bot, np.nan)

        out = pd.DataFrame({"timestamp": m15["timestamp"].reset_index(drop=True)})
        out["m15_bull_fvg_top"] = bull_top
        out["m15_bull_fvg_bot"] = bull_bot
        out["m15_bull_fvg_age"] = bull_age
        out["m15_bear_fvg_top"] = bear_top
        out["m15_bear_fvg_bot"] = bear_bot
        out["m15_bear_fvg_age"] = bear_age
        return out

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        for tf in self.required_htfs:
            if tf not in htfs:
                raise ValueError(f"XauMtfSmcEntry requires HTF '{tf}'")
        p = self.merged_params(params)

        # ---- STAGE 1: H4 OB zones ----
        h4 = htfs["H4"]
        h4_feat = self._h4_ob_zones(
            h4, int(p["h4_swing_left"]), int(p["h4_swing_right"]),
            int(p["h4_ob_max_age_bars"]), int(p["atr_period"]),
        )
        h4_aligned = align_htf_to_ltf(
            ltf=ltf, htf=h4_feat, ltf_tf=self.ltf, htf_tf="H4",
            htf_cols=["h4_bull_top", "h4_bull_bot", "h4_bear_top", "h4_bear_bot"],
            suffix="",
        )

        # ---- STAGE 2: M15 FVG features ----
        m15 = htfs["M15"]
        m15_feat = self._m15_fvg_features(
            m15, int(p["m15_fvg_lookback"]), float(p["m15_fvg_min_size_atr"]),
            int(p["atr_period"]),
        )
        m15_aligned = align_htf_to_ltf(
            ltf=ltf, htf=m15_feat, ltf_tf=self.ltf, htf_tf="M15",
            htf_cols=["m15_bull_fvg_top", "m15_bull_fvg_bot",
                      "m15_bear_fvg_top", "m15_bear_fvg_bot"],
            suffix="",
        )

        # ---- STAGE 3: M5 BOS confirmation ----
        m5_sw = swings(ltf, left=int(p["m5_swing_left"]), right=int(p["m5_swing_right"]))
        m5_st = structure_breaks(ltf, m5_sw)
        m5_trend = m5_st["trend"]
        m5_bos = m5_st["bos"]

        m5_atr = atr(ltf, int(p["atr_period"]))

        # ---- Composition ----
        # Bullish setup requires:
        #   (a) H4 bull OB is active and current bar's LOW is inside that OB zone
        #       (or just below — give a small tolerance)
        #   (b) M15 bull FVG is active and inside the H4 OB zone (overlap)
        #   (c) M5 trend just flipped/continued bullish (BOS bar in uptrend state)
        h4_bull_top = h4_aligned["h4_bull_top"]
        h4_bull_bot = h4_aligned["h4_bull_bot"]
        h4_bear_top = h4_aligned["h4_bear_top"]
        h4_bear_bot = h4_aligned["h4_bear_bot"]
        m15_bull_fvg_top = m15_aligned["m15_bull_fvg_top"]
        m15_bull_fvg_bot = m15_aligned["m15_bull_fvg_bot"]
        m15_bear_fvg_top = m15_aligned["m15_bear_fvg_top"]
        m15_bear_fvg_bot = m15_aligned["m15_bear_fvg_bot"]

        # Inside H4 OB? Use bar low for bull, high for bear.
        in_h4_bull_zone = (
            h4_bull_top.notna() & h4_bull_bot.notna()
            & (ltf["low"] <= h4_bull_top)
            & (ltf["low"] >= h4_bull_bot)
        )
        in_h4_bear_zone = (
            h4_bear_top.notna() & h4_bear_bot.notna()
            & (ltf["high"] >= h4_bear_bot)
            & (ltf["high"] <= h4_bear_top)
        )

        # M15 FVG overlaps H4 OB? Conservative: FVG center inside OB.
        m15_bull_center = (m15_bull_fvg_top + m15_bull_fvg_bot) / 2.0
        m15_bear_center = (m15_bear_fvg_top + m15_bear_fvg_bot) / 2.0
        m15_bull_in_ob = (
            m15_bull_fvg_top.notna() & h4_bull_top.notna()
            & (m15_bull_center <= h4_bull_top * 1.001)
            & (m15_bull_center >= h4_bull_bot * 0.999)
        )
        m15_bear_in_ob = (
            m15_bear_fvg_top.notna() & h4_bear_top.notna()
            & (m15_bear_center >= h4_bear_bot * 0.999)
            & (m15_bear_center <= h4_bear_top * 1.001)
        )

        # M5 BOS in same direction
        bull_bos_ltf = m5_bos & (m5_trend == 1)
        bear_bos_ltf = m5_bos & (m5_trend == -1)

        # Session filter
        if bool(p["session_filter"]):
            hour = ltf["timestamp"].dt.hour
            in_session = (hour >= int(p["trade_start_hour"])) & (hour < int(p["trade_end_hour"]))
        else:
            in_session = pd.Series(True, index=ltf.index)

        # S/R confluence gate (classical levels: PDH/PDL, weekly P/R1/S1, round)
        if bool(p["require_sr_confluence"]):
            pdh_df = prior_day_high_low(ltf)
            wk_df = weekly_pivot(ltf)
            sr = has_sr_confluence(
                close=ltf["close"], atr_series=m5_atr,
                pdh=pdh_df["pdh"], pdl=pdh_df["pdl"],
                wkly_p=wk_df["wkly_p"], wkly_r1=wk_df["wkly_r1"], wkly_s1=wk_df["wkly_s1"],
                round_step=float(p["sr_round_step"]),
                tolerance_atr=float(p["sr_tolerance_atr"]),
            )
            sr_long = sr["long_sr"]
            sr_short = sr["short_sr"]
        else:
            sr_long = pd.Series(True, index=ltf.index)
            sr_short = pd.Series(True, index=ltf.index)

        # Compose
        long_mask = (
            in_h4_bull_zone & m15_bull_in_ob & bull_bos_ltf & in_session & sr_long
            & m5_atr.notna() & h4_bull_bot.notna()
        )
        short_mask = (
            in_h4_bear_zone & m15_bear_in_ob & bear_bos_ltf & in_session & sr_short
            & m5_atr.notna() & h4_bear_top.notna()
        )

        sigs = empty_signals(ltf)
        sl_buf = float(p["sl_buffer_atr"]) * m5_atr

        tp_mode = str(p.get("tp_mode", "rr_relative")).lower()
        if tp_mode == "atr_absolute":
            tp_d = float(p["tp_atr_mult"]) * m5_atr
        elif tp_mode == "rr_relative":
            tp_rr = float(p.get("tp_rr", 2.0))
        else:
            raise ValueError(f"unknown tp_mode {tp_mode!r}")

        if long_mask.any():
            sigs.loc[long_mask, "action"] = "enter_long"
            # SL beyond H4 OB low
            long_sl = (h4_bull_bot - sl_buf)
            sigs.loc[long_mask, "sl"] = long_sl.loc[long_mask].values
            if tp_mode == "atr_absolute":
                sigs.loc[long_mask, "tp"] = (ltf["close"] + tp_d).loc[long_mask].values
            else:  # rr_relative
                sl_dist = ltf["close"] - long_sl
                sigs.loc[long_mask, "tp"] = (ltf["close"] + tp_rr * sl_dist).loc[long_mask].values
        if short_mask.any():
            sigs.loc[short_mask, "action"] = "enter_short"
            short_sl = (h4_bear_top + sl_buf)
            sigs.loc[short_mask, "sl"] = short_sl.loc[short_mask].values
            if tp_mode == "atr_absolute":
                sigs.loc[short_mask, "tp"] = (ltf["close"] - tp_d).loc[short_mask].values
            else:
                sl_dist = short_sl - ltf["close"]
                sigs.loc[short_mask, "tp"] = (ltf["close"] - tp_rr * sl_dist).loc[short_mask].values

        # Drop entries with degenerate SL/TP
        bad = sigs["action"].isin(["enter_long", "enter_short"]) & (
            sigs["sl"].isna() | sigs["tp"].isna()
        )
        if bad.any():
            sigs.loc[bad, "action"] = "hold"
            sigs.loc[bad, "sl"] = np.nan
            sigs.loc[bad, "tp"] = np.nan

        validate_signals(sigs, len(ltf))
        return StrategyResult(signals=sigs, debug={
            "in_h4_bull_zone": in_h4_bull_zone.astype(bool),
            "in_h4_bear_zone": in_h4_bear_zone.astype(bool),
            "m15_bull_in_ob": m15_bull_in_ob.astype(bool),
            "m15_bear_in_ob": m15_bear_in_ob.astype(bool),
            "bull_bos_ltf": bull_bos_ltf.astype(bool),
            "bear_bos_ltf": bear_bos_ltf.astype(bool),
        })
