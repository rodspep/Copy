"""XAU MA34/89 multi-timeframe cascade — UG Trading-style pullback strategy.

The user observed that UG's reference signals consistently quote MA34/MA89
on M5/M15/M30/H1 as the trend framework, and use FIXED SL/TP (not
structural). The session also discovered that our existing `XauMtfSmcEntry`
maintains WR 82% only because its SL is anchored to the H4 OB low (very
wide), absorbing noise — replacing that SL with a fixed ATR-mult SL caused
WR to COLLAPSE to 65%.

This strategy isolates the question: **can the UG entry framework alone
sustain a high WR with FIXED SL/TP?**

Pipeline:

  STAGE 1 — MTF MA34/89 alignment (the bias)
      Require EMA34 > EMA89 on M5 AND M15 AND M30 AND H1 (all four) for
      longs; mirrored for shorts. This is much stricter than a single-TF
      EMA filter and matches the framework UG's analysis text references.

  STAGE 2 — M5 pullback to EMA34
      Trigger long when M5 low touches within `pullback_atr` × ATR of M5
      EMA34 from above (pullback into trend MA). Mirror for short.

  STAGE 3 — Confirmation candle
      M5 close > open (bullish) for longs; < for shorts. Engine standard.

  Optional STAGE 4 — Light SMC confluence (`require_smc`)
      M15 SMC trend agrees with the trade direction. Light touch, not full
      cascade gating.

  Optional STAGE 5 — Session filter (`session_filter`)

  SL: entry − sl_atr × M5_ATR   (long; mirror for short). FIXED, not
      structural. This is the experimental contrast vs `XauMtfSmcEntry`.

  TP: entry + tp_atr × M5_ATR   (long; mirror for short). FIXED.

Long-and-short symmetric.

What we expect to learn:
  - If WR stays ≥ 80% with R:R 0.5-1.0 → the UG entry framework is the
    real edge, deploy this strategy.
  - If WR collapses to 60-65% (as fixed-SL did on the cascade) → the
    framework alone is insufficient and the structural-SL crutch is
    necessary for any high-WR XAU scalp.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import (
    ema, atr, align_htf_to_ltf,
    swings, structure_breaks,
)
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


class XauMa34Cascade(Strategy):
    """MA34/89 MTF cascade + M5 pullback + FIXED ATR SL/TP."""

    ltf = "M5"
    required_htfs = ("M15", "M30", "H1")

    default_params: dict[str, Any] = {
        # MA framework (UG uses 34/89; we keep them as defaults but allow tuning)
        "ma_fast":            34,
        "ma_slow":            89,

        # Pullback proximity to M5 MA-fast (long: low within X × ATR of EMA-fast)
        "pullback_atr":       0.4,

        # MTF alignment requirement: how many of {M5, M15, M30, H1} must agree
        # (matches UG framework exactly with M30 resampled from M5)
        "min_tf_agree":       4,    # default = all four (strictest, UG-style)

        # Optional light SMC confluence on M15
        "require_smc":        False,
        "smc_swing_left":     3,
        "smc_swing_right":    3,

        # Session
        "session_filter":     True,
        "trade_start_hour":   7,
        "trade_end_hour":     18,

        # FIXED ATR-based SL/TP — the experimental design
        "atr_period":         14,
        "sl_atr_mult":        1.0,    # M5 ATR ≈ $1-2 on XAU → SL ~$1-2
        "tp_atr_mult":        0.5,    # R:R 0.5 = UG TP1 style
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trend_series(closes: pd.Series, fast: int, slow: int) -> pd.Series:
        f = ema(closes, fast)
        s = ema(closes, slow)
        out = pd.Series(0, index=closes.index, dtype=np.int8)
        out[f > s] = 1
        out[f < s] = -1
        out[~(f.notna() & s.notna())] = 0
        return out

    @staticmethod
    def _ma_align_htf(ltf: pd.DataFrame, htf: pd.DataFrame, htf_tf: str,
                      ltf_tf: str, fast: int, slow: int) -> pd.Series:
        """HTF MA trend (-1/0/+1) aligned to LTF index."""
        h = htf.copy()
        h["__t"] = XauMa34Cascade._trend_series(h["close"], fast, slow).astype(float)
        feat = h[["timestamp", "__t"]]
        aligned = align_htf_to_ltf(
            ltf=ltf, htf=feat, ltf_tf=ltf_tf, htf_tf=htf_tf,
            htf_cols=["__t"], suffix="",
        )
        return aligned["__t"]

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        for tf in self.required_htfs:
            if tf not in htfs:
                raise ValueError(f"XauMa34Cascade requires HTF '{tf}'")
        p = self.merged_params(params)
        f = int(p["ma_fast"]); s = int(p["ma_slow"])

        # ---- STAGE 1: MTF MA alignment (M5 + M15 + M30 + H1) ----
        m5_t = self._trend_series(ltf["close"], f, s)
        m15_t = self._ma_align_htf(ltf, htfs["M15"], "M15", self.ltf, f, s)
        m30_t = self._ma_align_htf(ltf, htfs["M30"], "M30", self.ltf, f, s)
        h1_t = self._ma_align_htf(ltf, htfs["H1"], "H1", self.ltf, f, s)

        # Count agreements per side (4 TFs now, matching UG)
        n_bull = ((m5_t == 1).astype(int) + (m15_t == 1).astype(int)
                  + (m30_t == 1).astype(int) + (h1_t == 1).astype(int))
        n_bear = ((m5_t == -1).astype(int) + (m15_t == -1).astype(int)
                  + (m30_t == -1).astype(int) + (h1_t == -1).astype(int))

        min_agree = int(p["min_tf_agree"])
        bull_align = n_bull >= min_agree
        bear_align = n_bear >= min_agree

        # ---- STAGE 2: M5 pullback to EMA-fast ----
        m5_ema_fast = ema(ltf["close"], f)
        a = atr(ltf, int(p["atr_period"]))
        prox = float(p["pullback_atr"]) * a
        # Long pullback: low <= EMA-fast + prox AND close > EMA-fast (held above)
        pullback_long = (
            (ltf["low"] <= m5_ema_fast + prox)
            & (ltf["close"] > m5_ema_fast)
        )
        # Short pullback: high >= EMA-fast - prox AND close < EMA-fast
        pullback_short = (
            (ltf["high"] >= m5_ema_fast - prox)
            & (ltf["close"] < m5_ema_fast)
        )

        # ---- STAGE 3: Confirmation candle ----
        bull_cand = ltf["close"] > ltf["open"]
        bear_cand = ltf["close"] < ltf["open"]

        # ---- STAGE 4: Optional M15 SMC light gate ----
        if bool(p["require_smc"]):
            m15 = htfs["M15"].copy()
            m15_sw = swings(m15, left=int(p["smc_swing_left"]), right=int(p["smc_swing_right"]))
            m15_st = structure_breaks(m15, m15_sw)
            feat = m15[["timestamp"]].copy()
            feat["__t"] = m15_st["trend"].values
            aligned = align_htf_to_ltf(
                ltf=ltf, htf=feat, ltf_tf=self.ltf, htf_tf="M15",
                htf_cols=["__t"], suffix="",
            )
            smc_t = aligned["__t"]
            smc_long = smc_t == 1
            smc_short = smc_t == -1
        else:
            smc_long = pd.Series(True, index=ltf.index)
            smc_short = pd.Series(True, index=ltf.index)

        # ---- STAGE 5: Session filter ----
        if bool(p["session_filter"]):
            hour = ltf["timestamp"].dt.hour
            in_session = (hour >= int(p["trade_start_hour"])) & (hour < int(p["trade_end_hour"]))
        else:
            in_session = pd.Series(True, index=ltf.index)

        # ---- Compose ----
        long_mask = (
            bull_align & pullback_long & bull_cand & smc_long & in_session
            & a.notna() & m5_ema_fast.notna()
        )
        short_mask = (
            bear_align & pullback_short & bear_cand & smc_short & in_session
            & a.notna() & m5_ema_fast.notna()
        )

        # ---- FIXED SL/TP — the experimental contrast ----
        sl_dist = float(p["sl_atr_mult"]) * a
        tp_dist = float(p["tp_atr_mult"]) * a

        sigs = empty_signals(ltf)
        if long_mask.any():
            sigs.loc[long_mask, "action"] = "enter_long"
            sigs.loc[long_mask, "sl"] = (ltf["close"] - sl_dist).loc[long_mask].values
            sigs.loc[long_mask, "tp"] = (ltf["close"] + tp_dist).loc[long_mask].values
        if short_mask.any():
            sigs.loc[short_mask, "action"] = "enter_short"
            sigs.loc[short_mask, "sl"] = (ltf["close"] + sl_dist).loc[short_mask].values
            sigs.loc[short_mask, "tp"] = (ltf["close"] - tp_dist).loc[short_mask].values

        validate_signals(sigs, len(ltf))
        return StrategyResult(signals=sigs, debug={
            "n_bull": n_bull, "n_bear": n_bear,
            "bull_align": bull_align.astype(bool),
            "bear_align": bear_align.astype(bool),
            "pullback_long": pullback_long.astype(bool),
            "pullback_short": pullback_short.astype(bool),
        })
