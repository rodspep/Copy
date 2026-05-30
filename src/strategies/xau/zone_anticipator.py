"""XAU Anticipatory Zone strategy — UG Trading-style scalp.

This strategy reverses the timing of the existing SMC strategies in this repo:

  - `XauSmcOrderBlock` and `XauSmcConfluence` are REACTIVE — they fire AFTER
    a sweep/retest event has printed. By the time the signal lands, the
    reversal has already happened (often partially), so an additional limit
    pullback typically misses, and the natural target is small.

  - `XauZoneAnticipator` is ANTICIPATORY — it identifies confluence ZONES
    while price is *still away* from them, then offers two execution modes:
      * `mode='limit'`    : place a pending limit at the zone edge; only fill
                            if price reaches the zone within `expire_bars`.
      * `mode='reaction'` : do nothing until price touches the zone, then
                            wait one bar for a confirmation candle in the
                            reversal direction, then enter at market.

This matches the UG Trading XAU bot pattern the user pointed at:
  * Entry RANGE (zone with width) rather than a single price.
  * Wait for retracement rather than chase.
  * Tight TP (R:R 0.3–0.7) and full SL beyond zone — designed for high WR,
    small per-trade R.

ZONES considered (union — optimizer can disable each):
  Z1. Bullish OB top / Bearish OB bottom (from LTF M5 structure).
  Z2. Bullish FVG zone / Bearish FVG zone.
  Z3. Last unbroken swing high / swing low (liquidity).
  Z4. Asian-session range high / low (`asian_start_hour` .. `asian_end_hour`).

CONFIRMATION GATES (each toggleable):
  G1. MTF MA34/89 alignment on M5 + M15 (matches UG's MA34/MA89 framework).
  G2. HTF (H1) SMC trend agrees with the trade direction.
  G3. Session filter (London + early NY UTC hours).

Long-AND-short symmetric.

The strategy is parameter-rich on purpose: Optuna will learn which zones +
which gates + which entry mode produces sustainable OOS performance for the
asset's current regime. Every gate has an off-switch so the optimizer can
peel them off when they cost more signals than they save.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import (
    ema, atr, align_htf_to_ltf,
    swings, structure_breaks, order_blocks, fair_value_gaps,
)
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


# -----------------------------------------------------------------------------
# Zone helpers — compute the most-recent zone edges per LTF bar, all causal.
# -----------------------------------------------------------------------------

def _asian_zone_edges(ltf: pd.DataFrame, start_hour: int = 22, end_hour: int = 7
                      ) -> tuple[pd.Series, pd.Series]:
    """Return (asian_high, asian_low) aligned to ltf, value 'available_at' = end_hour UTC.

    Logic identical to XauSessionBreakout._asian_range but returns Series, not DF.
    """
    ts = ltf["timestamp"]
    if ts.dt.tz is None or str(ts.dt.tz) != "UTC":
        raise ValueError("asian zone requires UTC timestamps")
    hour = ts.dt.hour
    date = ts.dt.normalize()
    session_end = pd.Series(pd.NaT, index=ts.index, dtype=ts.dtype)
    mask_early = hour < end_hour
    mask_late = hour >= start_hour
    session_end.loc[mask_early] = date.loc[mask_early]
    session_end.loc[mask_late] = date.loc[mask_late] + pd.Timedelta(days=1)

    in_asian = mask_early | mask_late
    df = pd.DataFrame({"session_end": session_end, "high": ltf["high"],
                       "low": ltf["low"], "in_asian": in_asian, "ts": ts})
    df_a = df[df["in_asian"]].copy()
    if df_a.empty:
        empty = pd.Series(np.nan, index=ltf.index)
        return empty, empty.copy()
    grp = df_a.groupby("session_end", sort=False)
    df_a["cum_high"] = grp["high"].cummax()
    df_a["cum_low"] = grp["low"].cummin()
    sess_final = df_a.groupby("session_end", sort=False).agg(
        asian_high=("cum_high", "last"), asian_low=("cum_low", "last")
    ).reset_index()
    sess_final["available_at"] = sess_final["session_end"] + pd.Timedelta(hours=end_hour)
    sess_final = sess_final.sort_values("available_at")
    merged = pd.merge_asof(
        ltf[["timestamp"]].sort_values("timestamp").reset_index(drop=False),
        sess_final[["available_at", "asian_high", "asian_low"]],
        left_on="timestamp", right_on="available_at",
        direction="backward", allow_exact_matches=True,
    ).sort_values("index").reset_index(drop=True)
    return (pd.Series(merged["asian_high"].to_numpy(), index=ltf.index),
            pd.Series(merged["asian_low"].to_numpy(), index=ltf.index))


class XauZoneAnticipator(Strategy):
    """Anticipate price arrival at confluence zones; enter limit OR reaction.

    `mode` controls how a zone touch becomes an entry:
      - 'limit'    : enter as soon as the bar's wick crosses the zone edge.
      - 'reaction' : after the wick touches the zone, wait for the NEXT bar to
                     close back inside the trade direction (one-bar
                     confirmation), then enter at that confirmation bar.

    Both modes carry SL beyond the zone with an ATR buffer and TP at R:R fixed
    by `tp_atr_mult` / `sl_atr_mult`.
    """

    ltf = "M5"
    required_htfs = ("M15", "H1")

    default_params: dict[str, Any] = {
        # Entry mode
        "mode":                "limit",      # 'limit' | 'reaction'

        # Zone selection (each independently toggleable so the optimizer can
        # decide which zones are useful for the current regime).
        "use_ob_zone":         True,
        "use_fvg_zone":        True,
        "use_swing_zone":      True,
        "use_asian_zone":      True,

        # Swing detection (also used for LTF structure)
        "swing_left":          4,
        "swing_right":         4,

        # Zone buffer: how close (in ATR) the bar wick must come to count as touch
        "zone_touch_atr":      0.05,

        # Asian session (UTC hours)
        "asian_start_hour":    22,
        "asian_end_hour":      7,

        # Confirmation gates
        "require_ma_align":    True,         # MA34/89 on M5 + M15 align
        "ma_fast":             34,
        "ma_slow":             89,
        "require_htf_smc":     False,        # H1 SMC trend agrees with side
        "session_filter":      True,
        "trade_start_hour":    7,
        "trade_end_hour":      18,

        # Reaction mode: how many bars after wick-touch we still accept confirmation
        "reaction_max_wait":   2,

        # SL/TP (R:R-driven, default short TP = UG-style scalp)
        "atr_period":          14,
        "sl_buffer_atr":       0.5,          # SL placed = zone_edge ± sl_buffer*ATR (beyond)
        "tp_atr_mult":         0.5,          # TP distance from entry, in ATR multiples
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ma_align_series(closes: pd.Series, fast: int, slow: int) -> pd.Series:
        """+1 if EMA(fast) > EMA(slow), -1 if <, 0 if NaN."""
        f = ema(closes, fast)
        s = ema(closes, slow)
        out = pd.Series(0, index=closes.index, dtype=np.int8)
        out[f > s] = 1
        out[f < s] = -1
        out[~(f.notna() & s.notna())] = 0
        return out

    def _build_zones(self, ltf: pd.DataFrame, p: dict
                     ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Compute the consolidated demand/supply zone edges aligned to LTF.

        For each LTF bar i, returns:
          - demand_top, demand_bot : "buy-side" zone (longs entered here)
          - supply_top, supply_bot : "sell-side" zone (shorts entered here)
        Edges = the closest of all enabled zone sources (so we trade the
        zone that's nearest to current price).
        """
        n = len(ltf)
        # Initialize with NaN
        d_top = pd.Series(np.nan, index=ltf.index)
        d_bot = pd.Series(np.nan, index=ltf.index)
        s_top = pd.Series(np.nan, index=ltf.index)
        s_bot = pd.Series(np.nan, index=ltf.index)

        # Z1 + Z2: need LTF SMC structure
        if p["use_ob_zone"] or p["use_fvg_zone"] or p["use_swing_zone"]:
            sw = swings(ltf, left=int(p["swing_left"]), right=int(p["swing_right"]))
        # OB
        if p["use_ob_zone"]:
            st = structure_breaks(ltf, sw)
            ob = order_blocks(ltf, st)
            d_top = ob["bull_ob_top"]
            d_bot = ob["bull_ob_bot"]
            s_top = ob["bear_ob_top"]
            s_bot = ob["bear_ob_bot"]
        # FVG
        if p["use_fvg_zone"]:
            fv = fair_value_gaps(ltf)
            # Use most recent FVG zone. If both OB and FVG present, take the one
            # closest to current close (latest = freshest).
            close = ltf["close"]
            # Bullish FVG can be a demand zone
            fvg_d_top = fv["bull_fvg_top"]
            fvg_d_bot = fv["bull_fvg_bot"]
            # Replace demand zone if FVG is closer to current close from above.
            need_d = d_top.isna() | (close > d_top)  # use FVG if either no OB or current price above OB
            d_top = d_top.where(~need_d, fvg_d_top)
            d_bot = d_bot.where(~need_d, fvg_d_bot)
            # Bearish FVG = supply zone
            fvg_s_top = fv["bear_fvg_top"]
            fvg_s_bot = fv["bear_fvg_bot"]
            need_s = s_top.isna() | (close < s_bot)
            s_top = s_top.where(~need_s, fvg_s_top)
            s_bot = s_bot.where(~need_s, fvg_s_bot)
        # Swing zones (last confirmed swing high/low) — used as PURE liquidity edges,
        # turn into 1-bar-wide zones (width = 1*ATR / 4 so the "zone" is a small band).
        if p["use_swing_zone"]:
            sh = sw["swing_high_price"]
            sl = sw["swing_low_price"]
            a = atr(ltf, int(p["atr_period"]))
            half = a * 0.25
            # swing_low band is demand zone (potential bounce)
            sw_d_top = sl + half
            sw_d_bot = sl - half
            close = ltf["close"]
            need_d = d_top.isna() | (close > d_top * 1.001)
            d_top = d_top.where(~need_d, sw_d_top)
            d_bot = d_bot.where(~need_d, sw_d_bot)
            # swing_high band = supply
            sw_s_top = sh + half
            sw_s_bot = sh - half
            need_s = s_top.isna() | (close < s_bot * 0.999)
            s_top = s_top.where(~need_s, sw_s_top)
            s_bot = s_bot.where(~need_s, sw_s_bot)
        # Asian zone
        if p["use_asian_zone"]:
            a_high, a_low = _asian_zone_edges(
                ltf, int(p["asian_start_hour"]), int(p["asian_end_hour"]),
            )
            atr_v = atr(ltf, int(p["atr_period"]))
            half = atr_v * 0.25
            close = ltf["close"]
            # asian low = demand zone (price below = below recent range)
            need_d = d_top.isna() | (close > d_top * 1.001)
            d_top = d_top.where(~need_d, a_low + half)
            d_bot = d_bot.where(~need_d, a_low - half)
            # asian high = supply
            need_s = s_top.isna() | (close < s_bot * 0.999)
            s_top = s_top.where(~need_s, a_high + half)
            s_bot = s_bot.where(~need_s, a_high - half)

        return d_top, d_bot, s_top, s_bot

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        for tf in self.required_htfs:
            if tf not in htfs:
                raise ValueError(f"XauZoneAnticipator requires HTF '{tf}'")
        p = self.merged_params(params)
        mode = str(p["mode"]).lower()
        if mode not in ("limit", "reaction"):
            raise ValueError(f"unknown mode {mode!r}")

        a = atr(ltf, int(p["atr_period"]))
        d_top, d_bot, s_top, s_bot = self._build_zones(ltf, p)

        touch = float(p["zone_touch_atr"]) * a

        # --- Zone touch detection (causal: uses only bar i's high/low) ---
        bull_touch = (
            d_top.notna()
            & a.notna()
            & (ltf["low"] <= d_top + touch)   # wick reached demand zone
            & (ltf["low"] >= d_bot - touch * 3)  # not so far below it's irrelevant
        )
        bear_touch = (
            s_top.notna()
            & a.notna()
            & (ltf["high"] >= s_bot - touch)
            & (ltf["high"] <= s_top + touch * 3)
        )

        # --- Confirmation gates ---
        if bool(p["require_ma_align"]):
            f = int(p["ma_fast"]); s_ = int(p["ma_slow"])
            m5_align = self._ma_align_series(ltf["close"], f, s_)
            # M15 align aligned to LTF
            m15 = htfs["M15"].copy()
            m15["__a"] = self._ma_align_series(m15["close"], f, s_).astype(float)
            m15_feat = m15[["timestamp", "__a"]]
            aligned = align_htf_to_ltf(
                ltf=ltf, htf=m15_feat, ltf_tf=self.ltf, htf_tf="M15",
                htf_cols=["__a"], suffix="",
            )
            m15_align = aligned["__a"]
            ma_long = (m5_align == 1) & (m15_align == 1)
            ma_short = (m5_align == -1) & (m15_align == -1)
        else:
            ma_long = pd.Series(True, index=ltf.index)
            ma_short = pd.Series(True, index=ltf.index)

        if bool(p["require_htf_smc"]):
            h1 = htfs["H1"].copy()
            sw_h1 = swings(h1, left=int(p["swing_left"]), right=int(p["swing_right"]))
            st_h1 = structure_breaks(h1, sw_h1)
            feat = h1[["timestamp"]].copy()
            feat["__trend"] = st_h1["trend"].values
            aligned = align_htf_to_ltf(
                ltf=ltf, htf=feat, ltf_tf=self.ltf, htf_tf="H1",
                htf_cols=["__trend"], suffix="",
            )
            smc_long = aligned["__trend"] == 1
            smc_short = aligned["__trend"] == -1
        else:
            smc_long = pd.Series(True, index=ltf.index)
            smc_short = pd.Series(True, index=ltf.index)

        if bool(p["session_filter"]):
            hour = ltf["timestamp"].dt.hour
            in_session = (hour >= int(p["trade_start_hour"])) & (hour < int(p["trade_end_hour"]))
        else:
            in_session = pd.Series(True, index=ltf.index)

        long_ok = ma_long & smc_long & in_session
        short_ok = ma_short & smc_short & in_session

        sigs = empty_signals(ltf)

        # --- Entry semantics ---
        if mode == "limit":
            # Enter at the bar where the wick crossed the zone (i.e. the touch
            # bar itself) — this matches a pending-limit at zone top that fills
            # as price falls into it. SL goes BELOW zone_bot with buffer.
            long_mask = bull_touch & long_ok
            short_mask = bear_touch & short_ok
            sl_buf = float(p["sl_buffer_atr"]) * a
            tp_d = float(p["tp_atr_mult"]) * a

            if long_mask.any():
                sigs.loc[long_mask, "action"] = "enter_long"
                # Long SL = zone bottom − buffer (full structural stop)
                sigs.loc[long_mask, "sl"] = (d_bot - sl_buf).loc[long_mask].values
                # Long TP = close + tp_atr * ATR
                sigs.loc[long_mask, "tp"] = (ltf["close"] + tp_d).loc[long_mask].values
            if short_mask.any():
                sigs.loc[short_mask, "action"] = "enter_short"
                sigs.loc[short_mask, "sl"] = (s_top + sl_buf).loc[short_mask].values
                sigs.loc[short_mask, "tp"] = (ltf["close"] - tp_d).loc[short_mask].values

        else:  # 'reaction'
            # After a touch at bar i, watch bars i+1 .. i+reaction_max_wait for a
            # confirmation candle (close in reversal direction) — enter at THAT bar.
            n = len(ltf)
            max_wait = int(p["reaction_max_wait"])
            sl_buf = float(p["sl_buffer_atr"]) * a
            tp_d = float(p["tp_atr_mult"]) * a

            bull_touch_arr = bull_touch.to_numpy()
            bear_touch_arr = bear_touch.to_numpy()
            close = ltf["close"].to_numpy()
            open_ = ltf["open"].to_numpy()
            long_ok_arr = long_ok.to_numpy()
            short_ok_arr = short_ok.to_numpy()
            d_bot_arr = d_bot.to_numpy()
            s_top_arr = s_top.to_numpy()
            sl_buf_arr = sl_buf.to_numpy()
            tp_arr = tp_d.to_numpy()

            for i in range(n):
                if bull_touch_arr[i] and long_ok_arr[i]:
                    for j in range(i + 1, min(n, i + 1 + max_wait)):
                        if close[j] > open_[j] and long_ok_arr[j]:
                            # Confirmation candle. Enter at bar j.
                            if sigs.at[j, "action"] == "hold":
                                sigs.at[j, "action"] = "enter_long"
                                sigs.at[j, "sl"] = float(d_bot_arr[i] - sl_buf_arr[i])
                                sigs.at[j, "tp"] = float(close[j] + tp_arr[j])
                            break
                if bear_touch_arr[i] and short_ok_arr[i]:
                    for j in range(i + 1, min(n, i + 1 + max_wait)):
                        if close[j] < open_[j] and short_ok_arr[j]:
                            if sigs.at[j, "action"] == "hold":
                                sigs.at[j, "action"] = "enter_short"
                                sigs.at[j, "sl"] = float(s_top_arr[i] + sl_buf_arr[i])
                                sigs.at[j, "tp"] = float(close[j] - tp_arr[j])
                            break

        # Drop entries where SL/TP geometry is invalid (zone disappeared / NaN)
        bad = sigs["action"].isin(["enter_long", "enter_short"]) & (
            sigs["sl"].isna() | sigs["tp"].isna()
        )
        if bad.any():
            sigs.loc[bad, "action"] = "hold"
            sigs.loc[bad, "sl"] = np.nan
            sigs.loc[bad, "tp"] = np.nan

        validate_signals(sigs, len(ltf))
        return StrategyResult(signals=sigs)
