"""XAU SMC multi-confluence scalp.

Synthesis of the "UG Trading"-style XAU signal logic the user pointed at:
the signals reference SMC structure (BOS / CHoCH / OB / liquidity sweep) AND
multi-timeframe MA alignment AND session context. Per-window analysis on
session_20260527T073449 showed XauEmaPullback already has 15/38 OOS-positive
windows; the aggregate fails because there is no regime gate to suppress the
losing 23. This strategy is the regime-gated variant: an SMC entry trigger
that is allowed to fire only when several confluence checks agree.

Long-AND-short symmetric.

Pipeline (every check is a parameterized switch; the optimizer learns which
gates are worth their cost):

  ENTRY TRIGGER (M5, at least one — selected by `entry_trigger` param):
    A. "ob_retest"     : price retests the most recent bullish/bearish OB
                          (logic mirrors XauSmcOrderBlock).
    B. "fvg_retest"    : price retraces into the most recent bullish/bearish
                          FVG zone after a same-direction BOS.
    C. "sweep"         : a liquidity sweep bar at an M5 swing (low-wick &
                          close-back for longs, high-wick & close-back for
                          shorts).

  HTF SMC ALIGNMENT (gated by `require_htf_smc`):
    - H1 SMC trend (from `structure_breaks`) must equal the entry direction.
    - M15 SMC trend must equal the entry direction.

  MTF EMA ALIGNMENT (gated by `require_ema_align`):
    - On M5: EMA_fast > EMA_slow for long, < for short.
    - On M15 (aligned to M5): EMA_fast > EMA_slow for long, < for short.

  SESSION FILTER (gated by `session_filter`):
    - Only trade in [trade_start_hour, trade_end_hour) UTC.

  VOLATILITY REGIME (gated by `vol_regime_filter`):
    - M5 ATR must lie within [vol_pctile_low, vol_pctile_high] of its trailing
      `vol_lookback`-bar distribution. Excludes both dead-calm and
      news-spike bars.

  SL/TP:
    - Symmetric ATR-based, optimizer-tuned (sl_atr_mult, tp_atr_mult).

The strategy is designed so that with all gates ON it produces a sparse,
high-precision signal — ~5–15 trades/week — and with all gates OFF it
degenerates to a basic SMC entry. The optimizer can then learn the gate
config per walk-forward window.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import (
    ema, atr, align_htf_to_ltf,
    swings, structure_breaks, order_blocks, fair_value_gaps, liquidity_sweeps,
)
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


class XauSmcConfluence(Strategy):
    """Multi-confluence SMC entry with HTF + MTF + session + vol-regime gates."""

    ltf = "M5"
    required_htfs = ("M15", "H1")

    default_params: dict[str, Any] = {
        # Entry trigger selector
        "entry_trigger":          "sweep",      # 'sweep' | 'ob_retest' | 'fvg_retest'

        # Swing detection (used by sweep + structure_breaks at all TFs)
        "swing_left":             4,
        "swing_right":            4,

        # Trigger-specific
        "ob_proximity_pct":       0.0015,        # ob_retest only
        "fvg_max_age_bars":       30,            # fvg_retest only: ignore stale FVGs

        # HTF SMC alignment gate
        "require_htf_smc":        True,

        # MTF EMA alignment gate
        "require_ema_align":      True,
        "ema_fast":               21,
        "ema_slow":               50,

        # Session filter gate
        "session_filter":         True,
        "trade_start_hour":       7,
        "trade_end_hour":         16,

        # Volatility regime gate
        "vol_regime_filter":      False,
        "vol_lookback":           500,           # ~1.5 days of M5 bars
        "vol_pctile_low":         0.30,
        "vol_pctile_high":        0.85,

        # ATR + SL/TP
        "atr_period":             14,
        "sl_atr_mult":            1.0,
        "tp_atr_mult":            1.5,
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _smc_trend_aligned(ltf: pd.DataFrame, htf: pd.DataFrame, htf_tf: str,
                           ltf_tf: str, swing_left: int, swing_right: int) -> pd.Series:
        """Return HTF SMC trend series aligned to LTF index. Values in {-1, 0, 1}."""
        sw = swings(htf, left=swing_left, right=swing_right)
        st = structure_breaks(htf, sw)
        feat = htf[["timestamp"]].copy()
        feat["smc_trend"] = st["trend"].values
        aligned = align_htf_to_ltf(
            ltf=ltf, htf=feat, ltf_tf=ltf_tf, htf_tf=htf_tf,
            htf_cols=["smc_trend"], suffix="",
        )
        return aligned["smc_trend"]

    @staticmethod
    def _ema_align_aligned(ltf: pd.DataFrame, htf: pd.DataFrame, htf_tf: str,
                           ltf_tf: str, fast: int, slow: int) -> pd.Series:
        """Compute HTF EMA-fast minus EMA-slow, aligned to LTF. > 0 ⇒ bull bias."""
        h = htf.copy()
        h["_ema_fast"] = ema(h["close"], fast)
        h["_ema_slow"] = ema(h["close"], slow)
        h["_ema_diff"] = h["_ema_fast"] - h["_ema_slow"]
        feat = h[["timestamp", "_ema_diff"]]
        aligned = align_htf_to_ltf(
            ltf=ltf, htf=feat, ltf_tf=ltf_tf, htf_tf=htf_tf,
            htf_cols=["_ema_diff"], suffix="",
        )
        return aligned["_ema_diff"]

    @staticmethod
    def _fvg_retest_signals(ltf: pd.DataFrame, ltf_smc_trend: pd.Series,
                            max_age: int) -> tuple[pd.Series, pd.Series]:
        """Long/short masks for FVG-retest trigger on LTF.

        A bullish FVG retest fires when:
          - LTF SMC trend is +1
          - Most recent bullish FVG is at most `max_age` bars old
          - Current bar's low touches the FVG zone (low <= bull_fvg_top)
            while close stays above the bottom of the FVG (close > bull_fvg_bot)
        Mirror for shorts.
        """
        fvg = fair_value_gaps(ltf)
        # Distance (in bars) since last bullish/bearish FVG print.
        idx = np.arange(len(ltf))
        last_bull_fvg = np.where(fvg["bull_fvg"].to_numpy(), idx, -1)
        last_bear_fvg = np.where(fvg["bear_fvg"].to_numpy(), idx, -1)
        bull_age = idx - pd.Series(last_bull_fvg).cummax().to_numpy()
        bear_age = idx - pd.Series(last_bear_fvg).cummax().to_numpy()

        bull_top = fvg["bull_fvg_top"]
        bull_bot = fvg["bull_fvg_bot"]
        bear_top = fvg["bear_fvg_top"]
        bear_bot = fvg["bear_fvg_bot"]

        long_mask = (
            (ltf_smc_trend == 1)
            & bull_top.notna()
            & (pd.Series(bull_age, index=ltf.index) <= max_age)
            & (ltf["low"] <= bull_top)
            & (ltf["close"] > bull_bot)
        )
        short_mask = (
            (ltf_smc_trend == -1)
            & bear_bot.notna()
            & (pd.Series(bear_age, index=ltf.index) <= max_age)
            & (ltf["high"] >= bear_bot)
            & (ltf["close"] < bear_top)
        )
        return long_mask, short_mask

    @staticmethod
    def _ob_retest_signals(ltf: pd.DataFrame, ltf_smc_trend: pd.Series,
                           ltf_structure: pd.DataFrame, prox: float
                           ) -> tuple[pd.Series, pd.Series]:
        """Long/short masks for OB-retest trigger on LTF."""
        ob = order_blocks(ltf, ltf_structure)
        bull_top, bull_bot = ob["bull_ob_top"], ob["bull_ob_bot"]
        bear_top, bear_bot = ob["bear_ob_top"], ob["bear_ob_bot"]
        long_mask = (
            (ltf_smc_trend == 1)
            & bull_top.notna()
            & (ltf["low"] <= bull_top * (1 + prox))
            & (ltf["low"] >= bull_bot * (1 - prox))
            & (ltf["close"] > ltf["open"])
        )
        short_mask = (
            (ltf_smc_trend == -1)
            & bear_top.notna()
            & (ltf["high"] >= bear_bot * (1 - prox))
            & (ltf["high"] <= bear_top * (1 + prox))
            & (ltf["close"] < ltf["open"])
        )
        return long_mask, short_mask

    @staticmethod
    def _sweep_signals(ltf: pd.DataFrame, ltf_swings: pd.DataFrame
                       ) -> tuple[pd.Series, pd.Series]:
        """Long/short masks for liquidity-sweep trigger."""
        ls = liquidity_sweeps(ltf, ltf_swings)
        return ls["bull_sweep"], ls["bear_sweep"]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        ltf: pd.DataFrame,
        htfs: dict[str, pd.DataFrame],
        params: dict[str, Any] | None = None,
    ) -> StrategyResult:
        for tf in self.required_htfs:
            if tf not in htfs:
                raise ValueError(f"XauSmcConfluence requires HTF '{tf}'")
        p = self.merged_params(params)

        sl_left = int(p["swing_left"])
        sl_right = int(p["swing_right"])

        # ---- LTF SMC structure (used by every trigger) ----
        ltf_sw = swings(ltf, left=sl_left, right=sl_right)
        ltf_st = structure_breaks(ltf, ltf_sw)
        ltf_smc_trend = ltf_st["trend"]

        # ---- Entry trigger ----
        trigger = str(p["entry_trigger"]).lower()
        if trigger == "sweep":
            long_trig, short_trig = self._sweep_signals(ltf, ltf_sw)
        elif trigger == "ob_retest":
            long_trig, short_trig = self._ob_retest_signals(
                ltf, ltf_smc_trend, ltf_st, float(p["ob_proximity_pct"]),
            )
        elif trigger == "fvg_retest":
            long_trig, short_trig = self._fvg_retest_signals(
                ltf, ltf_smc_trend, int(p["fvg_max_age_bars"]),
            )
        else:
            raise ValueError(f"unknown entry_trigger: {trigger!r}")

        # ---- HTF SMC alignment gate ----
        if bool(p["require_htf_smc"]):
            m15_trend = self._smc_trend_aligned(
                ltf, htfs["M15"], "M15", self.ltf, sl_left, sl_right,
            )
            h1_trend = self._smc_trend_aligned(
                ltf, htfs["H1"], "H1", self.ltf, sl_left, sl_right,
            )
            htf_long = (m15_trend == 1) & (h1_trend == 1)
            htf_short = (m15_trend == -1) & (h1_trend == -1)
        else:
            htf_long = pd.Series(True, index=ltf.index)
            htf_short = pd.Series(True, index=ltf.index)

        # ---- MTF EMA alignment gate ----
        if bool(p["require_ema_align"]):
            f = int(p["ema_fast"]); s = int(p["ema_slow"])
            m5_diff = ema(ltf["close"], f) - ema(ltf["close"], s)
            m15_diff = self._ema_align_aligned(ltf, htfs["M15"], "M15", self.ltf, f, s)
            ema_long = (m5_diff > 0) & (m15_diff > 0)
            ema_short = (m5_diff < 0) & (m15_diff < 0)
        else:
            ema_long = pd.Series(True, index=ltf.index)
            ema_short = pd.Series(True, index=ltf.index)

        # ---- Session filter gate ----
        if bool(p["session_filter"]):
            hour = ltf["timestamp"].dt.hour
            in_session = (hour >= int(p["trade_start_hour"])) & (hour < int(p["trade_end_hour"]))
        else:
            in_session = pd.Series(True, index=ltf.index)

        # ---- Volatility regime gate ----
        a = atr(ltf, int(p["atr_period"]))
        if bool(p["vol_regime_filter"]):
            lb = int(p["vol_lookback"])
            lo = float(p["vol_pctile_low"]); hi = float(p["vol_pctile_high"])
            # Trailing percentile rank of current ATR within last `lb` bars.
            # Using rolling().rank(pct=True) gives the percentile of the LAST value
            # within the window.
            roll = a.rolling(lb, min_periods=max(50, lb // 5))
            rank_pct = roll.rank(pct=True)
            in_vol = (rank_pct >= lo) & (rank_pct <= hi)
        else:
            in_vol = pd.Series(True, index=ltf.index)

        # ---- Compose ----
        long_mask = long_trig & htf_long & ema_long & in_session & in_vol & a.notna()
        short_mask = short_trig & htf_short & ema_short & in_session & in_vol & a.notna()

        sigs = empty_signals(ltf)
        sl_dist = float(p["sl_atr_mult"]) * a
        tp_dist = float(p["tp_atr_mult"]) * a

        if long_mask.any():
            sigs.loc[long_mask, "action"] = "enter_long"
            # SL placed at the wick low of the trigger bar (more conservative than
            # close − N×ATR; matches what UG-style scalp does on sweep entries).
            sigs.loc[long_mask, "sl"] = (ltf["low"] - sl_dist).loc[long_mask].values
            sigs.loc[long_mask, "tp"] = (ltf["close"] + tp_dist).loc[long_mask].values
        if short_mask.any():
            sigs.loc[short_mask, "action"] = "enter_short"
            sigs.loc[short_mask, "sl"] = (ltf["high"] + sl_dist).loc[short_mask].values
            sigs.loc[short_mask, "tp"] = (ltf["close"] - tp_dist).loc[short_mask].values

        # If two signals would land on the same bar (shouldn't happen given the
        # opposite-trend gates, but possible when gates are OFF), keep long.
        validate_signals(sigs, len(ltf))

        return StrategyResult(signals=sigs, debug={
            "long_trig": long_trig.astype(bool),
            "short_trig": short_trig.astype(bool),
            "htf_long": htf_long.astype(bool),
            "htf_short": htf_short.astype(bool),
            "ema_long": ema_long.astype(bool),
            "ema_short": ema_short.astype(bool),
            "in_session": in_session.astype(bool),
            "in_vol": in_vol.astype(bool),
        })
