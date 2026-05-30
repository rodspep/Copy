"""XAU HTF-trend + LTF reversal-pattern scalp.

User's hypothesis: previous strategies either had wide structural SL (mtf_smc_entry
WR 82% but R:R 0.17) or weak entry confirmation (ma34_cascade fixed SL collapsed
WR to 58%). This strategy explores the middle ground:

  - **HTF (H1) trend gate**: EMA34 > EMA89 + ADX > threshold + close above EMA34
    (longs; mirror for shorts). Stricter than ema_pullback's filter.
  - **LTF (M5) PULLBACK to local zone**: price retraces to one of:
      (a) M5 EMA34/89 zone (configurable),
      (b) recent M5 swing low (longs) / swing high (shorts).
  - **EXPLICIT REVERSAL pattern** at the zone (one or more):
      (a) Engulfing candle (current bar's body engulfs prior bar's body),
      (b) Pin bar / hammer (long lower wick, small body, close near high — for longs),
      (c) RSI cross out of extreme (RSI crosses up through 30 in long setup).
  - **TIGHT LOCAL SL**: SL = recent local low − small ATR buffer (NOT HTF OB).
  - **FIXED R:R**: tp = entry + tp_rr × sl_distance (configurable 0.5-2.0).

Long-and-short symmetric.

Compared to existing strategies:
  - Stricter than `ema_pullback` (explicit reversal pattern, not just bullish candle)
  - Tighter SL than `mtf_smc_entry` (local extreme, not H4 OB)
  - Fixed SL/TP geometry but anchored to LOCAL bar low (not arbitrary N×ATR)
  - Multiple reversal-pattern options (OR'd) → high signal frequency
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import ema, atr, adx, rsi, align_htf_to_ltf, swings
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


def _engulfing(open_, close, side: int) -> pd.Series:
    """Bullish/bearish engulfing: current body engulfs prior body in the direction.

    side=+1 → bullish engulfing (close > open AND close > prev_open AND open < prev_close)
    side=-1 → bearish engulfing (mirror)
    """
    body_up = (close > open_)
    body_dn = (close < open_)
    if side == 1:
        return (
            body_up
            & body_dn.shift(1, fill_value=False)
            & (close > open_.shift(1))
            & (open_ < close.shift(1))
        )
    return (
        body_dn
        & body_up.shift(1, fill_value=False)
        & (close < open_.shift(1))
        & (open_ > close.shift(1))
    )


def _pin_bar(open_, high, low, close, side: int, body_ratio: float = 0.35,
             wick_ratio: float = 0.6) -> pd.Series:
    """Pin bar detection.

    For longs (side=+1): long lower wick, body in upper third.
        lower_wick / range >= wick_ratio
        body / range <= body_ratio
        close > open OR close >= midpoint (rejection of low)
    Shorts mirror.
    """
    rng = (high - low).where(high > low, np.nan)
    body = (close - open_).abs()
    upper_wick = high - close.where(close > open_, open_)
    lower_wick = close.where(close < open_, open_) - low
    body_pct = body / rng
    if side == 1:
        wick_pct = lower_wick / rng
        return (
            (wick_pct >= wick_ratio)
            & (body_pct <= body_ratio)
            & (close >= (low + (high - low) * 0.5))
        )
    wick_pct = upper_wick / rng
    return (
        (wick_pct >= wick_ratio)
        & (body_pct <= body_ratio)
        & (close <= (high - (high - low) * 0.5))
    )


class XauHtfTrendReversal(Strategy):
    """HTF trend gate + M5 pullback to local zone + explicit reversal pattern."""

    ltf = "M5"
    required_htfs = ("H1",)

    default_params: dict[str, Any] = {
        # HTF gate
        "h1_ema_fast":          34,
        "h1_ema_slow":          89,
        "h1_adx_min":           20.0,
        "require_h1_above_fast": True,    # close > EMA fast (extra trend confirmation)

        # M5 zone selection
        "zone_use_ema":         True,
        "zone_ema":             34,
        "zone_ema_atr":         0.5,      # within zone if low/high within atr×0.5 of EMA
        "zone_use_swing":       True,
        "swing_left":           5,
        "swing_right":          5,
        "zone_swing_atr":       0.5,

        # Reversal patterns (OR-ed)
        "pattern_engulfing":    True,
        "pattern_pin_bar":      True,
        "pattern_rsi_cross":    True,
        "rsi_period":           14,
        "rsi_oversold":         35.0,
        "rsi_overbought":       65.0,

        # Session
        "session_filter":       True,
        "trade_start_hour":     7,
        "trade_end_hour":       18,

        # Risk geometry — FIXED, local-extreme anchored
        "atr_period":           14,
        "sl_buffer_atr":        0.3,      # SL = local low − buffer × M5 ATR
        "tp_rr":                1.0,      # TP = entry + tp_rr × sl_distance
        "sl_min_atr":           0.5,      # ensure SL ≥ 0.5×ATR (avoid micro stops)
    }

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        if "H1" not in htfs:
            raise ValueError("XauHtfTrendReversal requires HTF 'H1'")
        p = self.merged_params(params)

        # ---- HTF gate ----
        h1 = htfs["H1"].copy()
        f = int(p["h1_ema_fast"]); s = int(p["h1_ema_slow"])
        h1["__ef"] = ema(h1["close"], f)
        h1["__es"] = ema(h1["close"], s)
        h1["__adx"] = adx(h1, 14)["adx"]
        h1["__close"] = h1["close"]
        feat = h1[["timestamp", "__ef", "__es", "__adx", "__close"]]
        aligned = align_htf_to_ltf(
            ltf=ltf, htf=feat, ltf_tf=self.ltf, htf_tf="H1",
            htf_cols=["__ef", "__es", "__adx", "__close"], suffix="",
        )
        h1_ef = aligned["__ef"]; h1_es = aligned["__es"]
        h1_adx = aligned["__adx"]; h1_close = aligned["__close"]

        adx_min = float(p["h1_adx_min"])
        require_above = bool(p["require_h1_above_fast"])
        h1_long = (h1_ef > h1_es) & (h1_adx >= adx_min)
        h1_short = (h1_ef < h1_es) & (h1_adx >= adx_min)
        if require_above:
            h1_long = h1_long & (h1_close > h1_ef)
            h1_short = h1_short & (h1_close < h1_ef)

        # ---- M5 zone detection ----
        m5_atr = atr(ltf, int(p["atr_period"]))
        in_zone_long = pd.Series(False, index=ltf.index)
        in_zone_short = pd.Series(False, index=ltf.index)

        if bool(p["zone_use_ema"]):
            zone_ema_v = ema(ltf["close"], int(p["zone_ema"]))
            tol = m5_atr * float(p["zone_ema_atr"])
            # Long zone: low touches/penetrates EMA from above
            in_zone_long |= (ltf["low"] <= zone_ema_v + tol) & (ltf["low"] >= zone_ema_v - tol * 2)
            # Short zone: high touches EMA from below
            in_zone_short |= (ltf["high"] >= zone_ema_v - tol) & (ltf["high"] <= zone_ema_v + tol * 2)

        if bool(p["zone_use_swing"]):
            sw = swings(ltf, left=int(p["swing_left"]), right=int(p["swing_right"]))
            sw_lo = sw["swing_low_price"]
            sw_hi = sw["swing_high_price"]
            tol2 = m5_atr * float(p["zone_swing_atr"])
            in_zone_long |= sw_lo.notna() & (ltf["low"] <= sw_lo + tol2) & (ltf["low"] >= sw_lo - tol2 * 2)
            in_zone_short |= sw_hi.notna() & (ltf["high"] >= sw_hi - tol2) & (ltf["high"] <= sw_hi + tol2 * 2)

        # ---- Reversal patterns ----
        open_, high, low, close = ltf["open"], ltf["high"], ltf["low"], ltf["close"]
        pattern_long = pd.Series(False, index=ltf.index)
        pattern_short = pd.Series(False, index=ltf.index)

        if bool(p["pattern_engulfing"]):
            pattern_long |= _engulfing(open_, close, side=1)
            pattern_short |= _engulfing(open_, close, side=-1)
        if bool(p["pattern_pin_bar"]):
            pattern_long |= _pin_bar(open_, high, low, close, side=1)
            pattern_short |= _pin_bar(open_, high, low, close, side=-1)
        if bool(p["pattern_rsi_cross"]):
            r = rsi(ltf["close"], int(p["rsi_period"]))
            os_ = float(p["rsi_oversold"]); ob = float(p["rsi_overbought"])
            # Cross up through oversold (was below, now above)
            pattern_long |= (r > os_) & (r.shift(1) <= os_)
            pattern_short |= (r < ob) & (r.shift(1) >= ob)

        # ---- Session filter ----
        if bool(p["session_filter"]):
            hour = ltf["timestamp"].dt.hour
            in_session = (hour >= int(p["trade_start_hour"])) & (hour < int(p["trade_end_hour"]))
        else:
            in_session = pd.Series(True, index=ltf.index)

        # ---- Compose ----
        long_mask = h1_long & in_zone_long & pattern_long & in_session & m5_atr.notna()
        short_mask = h1_short & in_zone_short & pattern_short & in_session & m5_atr.notna()

        # ---- Local SL + fixed-RR TP ----
        sl_min = float(p["sl_min_atr"]) * m5_atr
        sl_buf = float(p["sl_buffer_atr"]) * m5_atr
        tp_rr = float(p["tp_rr"])

        sigs = empty_signals(ltf)

        if long_mask.any():
            # SL = bar low − buffer (or entry − sl_min, whichever is lower / wider)
            entry_price = close
            raw_sl = (low - sl_buf)
            min_sl = entry_price - sl_min
            final_sl = np.minimum(raw_sl, min_sl)
            sl_dist = entry_price - final_sl
            final_tp = entry_price + tp_rr * sl_dist

            sigs.loc[long_mask, "action"] = "enter_long"
            sigs.loc[long_mask, "sl"] = final_sl.loc[long_mask].values
            sigs.loc[long_mask, "tp"] = final_tp.loc[long_mask].values

        if short_mask.any():
            entry_price = close
            raw_sl = (high + sl_buf)
            min_sl = entry_price + sl_min
            final_sl = np.maximum(raw_sl, min_sl)
            sl_dist = final_sl - entry_price
            final_tp = entry_price - tp_rr * sl_dist

            sigs.loc[short_mask, "action"] = "enter_short"
            sigs.loc[short_mask, "sl"] = final_sl.loc[short_mask].values
            sigs.loc[short_mask, "tp"] = final_tp.loc[short_mask].values

        validate_signals(sigs, len(ltf))
        return StrategyResult(signals=sigs, debug={
            "h1_long": h1_long.astype(bool),
            "h1_short": h1_short.astype(bool),
            "in_zone_long": in_zone_long,
            "in_zone_short": in_zone_short,
            "pattern_long": pattern_long,
            "pattern_short": pattern_short,
        })
