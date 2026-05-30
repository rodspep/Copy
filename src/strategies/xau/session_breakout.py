"""XAU London-session breakout with VWAP filter.

Idea: The Asian session (roughly 22:00-07:00 UTC) is typically range-bound; the
London open (~07:00-08:00 UTC) often breaks that range. We:
  1. Compute the Asian range (high/low between 22:00 prev day and 07:00).
  2. After 07:00 UTC, look for a close above Asian high (long) or below
     Asian low (short).
  3. Filter by VWAP direction (session-anchored daily_2200): only long if
     price > session VWAP, only short if price < session VWAP.
  4. SL = ATR-based beyond the breakout candle's opposite extreme.
  5. TP = ATR-based multiple.

Symmetric long-and-short. Designed for XAU's session structure (won't transfer
to BTC, which has no session boundary).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import atr, vwap
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


def _asian_range(ltf: pd.DataFrame, start_hour: int = 22, end_hour: int = 7) -> pd.DataFrame:
    """For each LTF bar, compute the most-recent Asian-session [high, low].

    Asian session for a given UTC date D = bars with timestamp in
    [D-1 22:00, D 07:00). The range is "available" once the bar at D 07:00 closes.

    Returns DataFrame with columns ['asian_high','asian_low'] aligned to ltf.
    Values are NaN before the first complete Asian session.
    """
    ts = ltf["timestamp"]
    if ts.dt.tz is None or str(ts.dt.tz) != "UTC":
        raise ValueError("Asian range requires UTC timestamps")

    # Compute session id: same id for all bars in [prev day 22:00, this day 07:00).
    # bar at hour h on date D belongs to session ending on D if h < 7;
    # to session ending on D+1 if h >= 22.
    # Use pandas (not np.where) so the UTC tz is preserved through the assignment.
    hour = ts.dt.hour
    date = ts.dt.normalize()
    session_end = pd.Series(pd.NaT, index=ts.index, dtype=ts.dtype)
    mask_early = hour < end_hour
    mask_late = hour >= start_hour
    session_end.loc[mask_early] = date.loc[mask_early]
    session_end.loc[mask_late] = date.loc[mask_late] + pd.Timedelta(days=1)

    # Compute per-session high/low using only bars within the session (h < 7 or h >= 22).
    in_asian = (hour < end_hour) | (hour >= start_hour)
    df = pd.DataFrame({
        "session_end": session_end,
        "high": ltf["high"],
        "low": ltf["low"],
        "in_asian": in_asian,
        "ts": ts,
    })
    df_asian = df[df["in_asian"]].copy()
    if df_asian.empty:
        return pd.DataFrame({"asian_high": np.full(len(ltf), np.nan), "asian_low": np.full(len(ltf), np.nan)}, index=ltf.index)
    # Cumulative max/min per session up to each bar (causal).
    grp = df_asian.groupby("session_end", sort=False)
    df_asian["cum_high"] = grp["high"].cummax()
    df_asian["cum_low"] = grp["low"].cummin()
    # Last value per session = the FULL session range, "available" at the session_end day's 07:00.
    sess_final = df_asian.groupby("session_end", sort=False).agg(asian_high=("cum_high", "last"), asian_low=("cum_low", "last")).reset_index()
    # Each session's range becomes available at 07:00 of its session_end date.
    sess_final["available_at"] = sess_final["session_end"] + pd.Timedelta(hours=end_hour)
    sess_final = sess_final.sort_values("available_at")

    # For each LTF bar, attach the latest session range whose available_at <= ts.
    merged = pd.merge_asof(
        ltf[["timestamp"]].sort_values("timestamp").reset_index(drop=False),
        sess_final[["available_at", "asian_high", "asian_low"]],
        left_on="timestamp", right_on="available_at",
        direction="backward", allow_exact_matches=True,
    )
    merged = merged.sort_values("index").reset_index(drop=True)
    return pd.DataFrame({
        "asian_high": merged["asian_high"].to_numpy(),
        "asian_low": merged["asian_low"].to_numpy(),
    }, index=ltf.index)


class XauSessionBreakout(Strategy):
    """London-open breakout of the Asian range, filtered by session VWAP.

    Trades only during the European/early-US session window (07:00-15:00 UTC by
    default — avoids the choppy late-NY hours).
    """

    ltf = "M5"
    required_htfs: tuple[str, ...] = ()

    default_params: dict[str, Any] = {
        "asian_start_hour": 22,
        "asian_end_hour": 7,
        "trade_start_hour": 7,
        "trade_end_hour": 15,
        "atr_period": 14,
        "sl_atr_mult": 1.0,
        "tp_atr_mult": 2.0,
        "breakout_buffer_atr": 0.1,  # require close > asian_high + buf*ATR
    }

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        p = self.merged_params(params)
        rng = _asian_range(ltf, int(p["asian_start_hour"]), int(p["asian_end_hour"]))
        v = vwap(ltf, anchor="daily_2200")  # XAU CME-style session
        ltf_atr = atr(ltf, int(p["atr_period"]))

        hour = ltf["timestamp"].dt.hour
        in_window = (hour >= int(p["trade_start_hour"])) & (hour < int(p["trade_end_hour"]))

        buf = float(p["breakout_buffer_atr"]) * ltf_atr
        long_break = (
            in_window
            & rng["asian_high"].notna()
            & ltf_atr.notna()
            & (ltf["close"] > rng["asian_high"] + buf)
            & (ltf["close"] > v["vwap"])
        )
        short_break = (
            in_window
            & rng["asian_low"].notna()
            & ltf_atr.notna()
            & (ltf["close"] < rng["asian_low"] - buf)
            & (ltf["close"] < v["vwap"])
        )

        sigs = empty_signals(ltf)
        sl_dist = float(p["sl_atr_mult"]) * ltf_atr
        tp_dist = float(p["tp_atr_mult"]) * ltf_atr

        if long_break.any():
            sigs.loc[long_break, "action"] = "enter_long"
            sigs.loc[long_break, "sl"] = (ltf["close"] - sl_dist).loc[long_break].values
            sigs.loc[long_break, "tp"] = (ltf["close"] + tp_dist).loc[long_break].values
        if short_break.any():
            sigs.loc[short_break, "action"] = "enter_short"
            sigs.loc[short_break, "sl"] = (ltf["close"] + sl_dist).loc[short_break].values
            sigs.loc[short_break, "tp"] = (ltf["close"] - tp_dist).loc[short_break].values

        validate_signals(sigs, len(ltf))
        return StrategyResult(signals=sigs, debug={"asian_high": rng["asian_high"], "asian_low": rng["asian_low"]})
