"""Support / Resistance levels — classical price levels for confluence.

All functions are lookahead-safe: the value reported at bar i uses ONLY
bars j ≤ i, never future bars.

Provided:
  - prior_day_high(ltf), prior_day_low(ltf)
      Prior calendar-day high/low, "available" from the start of the next UTC
      day (forward-fills until the day rolls over).

  - weekly_pivot(ltf)
      Classical (H+L+C)/3 pivot of the prior ISO-week, available from Monday
      00:00 UTC. Returns (pivot, r1, s1) Series.

  - session_high_low(ltf, start_hour, end_hour)
      Rolling high/low of the most recent fully closed session window. Useful
      for Asian/London/NY range markers.

  - round_levels_near(close, step, n_levels=3)
      Return the n nearest round-number levels above and below each close.

  - nearest_sr_distance(close, levels_df, atr_series)
      For each bar, distance (in ATR units) to the closest level in `levels_df`.

Patterns mirror src/indicators/smc.py: pure pandas/numpy, vectorized,
single-pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _require_utc(ltf: pd.DataFrame) -> None:
    ts = ltf["timestamp"]
    if ts.dt.tz is None or str(ts.dt.tz) != "UTC":
        raise ValueError("S/R helpers require tz-aware UTC timestamps")


def prior_day_high_low(ltf: pd.DataFrame) -> pd.DataFrame:
    """Return per-bar (pdh, pdl) — the high/low of the PRIOR UTC calendar day.

    Availability: PDH/PDL becomes valid at 00:00 UTC of the new day. Bars
    inside the same day continue to see yesterday's PDH/PDL; the bar where
    the date rolls picks up today's NEW prior-day values.
    """
    _require_utc(ltf)
    df = pd.DataFrame({
        "timestamp": ltf["timestamp"],
        "high": ltf["high"], "low": ltf["low"],
    })
    df["date"] = df["timestamp"].dt.normalize()  # UTC midnight
    daily = df.groupby("date").agg(day_high=("high", "max"), day_low=("low", "min"))
    # Shift by one day → "prior" day
    daily["pdh"] = daily["day_high"].shift(1)
    daily["pdl"] = daily["day_low"].shift(1)
    pdh = df["date"].map(daily["pdh"]).to_numpy()
    pdl = df["date"].map(daily["pdl"]).to_numpy()
    return pd.DataFrame({"pdh": pdh, "pdl": pdl}, index=ltf.index)


def weekly_pivot(ltf: pd.DataFrame) -> pd.DataFrame:
    """Return (weekly_pivot, weekly_r1, weekly_s1) per bar.

    Standard formulas (uses PRIOR ISO-week H/L/C):
      P  = (H + L + C) / 3
      R1 = 2P - L
      S1 = 2P - H
    """
    _require_utc(ltf)
    df = pd.DataFrame({
        "timestamp": ltf["timestamp"],
        "high": ltf["high"], "low": ltf["low"], "close": ltf["close"],
    })
    # ISO week start (Monday)
    df["wk"] = df["timestamp"].dt.to_period("W-SUN")  # weeks ending Sunday
    weekly = df.groupby("wk").agg(
        wh=("high", "max"), wl=("low", "min"),
        wc=("close", "last"),
    )
    weekly["P"] = (weekly["wh"] + weekly["wl"] + weekly["wc"]) / 3.0
    weekly["R1"] = 2 * weekly["P"] - weekly["wl"]
    weekly["S1"] = 2 * weekly["P"] - weekly["wh"]
    # Prior-week values
    weekly["P_prior"] = weekly["P"].shift(1)
    weekly["R1_prior"] = weekly["R1"].shift(1)
    weekly["S1_prior"] = weekly["S1"].shift(1)
    return pd.DataFrame({
        "wkly_p": df["wk"].map(weekly["P_prior"]).to_numpy(),
        "wkly_r1": df["wk"].map(weekly["R1_prior"]).to_numpy(),
        "wkly_s1": df["wk"].map(weekly["S1_prior"]).to_numpy(),
    }, index=ltf.index)


def round_level_distance(close: pd.Series, step: float) -> pd.DataFrame:
    """For each close, distance to the nearest round level above and below.

    A round level is any multiple of `step`. For XAU step=10 marks levels
    like 2000, 2010, 2020 ...; step=50 marks 2000, 2050, 2100 ...

    Returns:
      - 'level_above': nearest round level >= close
      - 'level_below': nearest round level <= close
      - 'dist_above': level_above - close
      - 'dist_below': close - level_below
    """
    if step <= 0:
        raise ValueError("round_level step must be > 0")
    c = close.to_numpy()
    above = np.ceil(c / step) * step
    below = np.floor(c / step) * step
    return pd.DataFrame({
        "level_above": above, "level_below": below,
        "dist_above": above - c, "dist_below": c - below,
    }, index=close.index)


def session_range(ltf: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
    """Per-bar (sess_high, sess_low) = high/low of the most recently CLOSED
    session window in [start_hour, end_hour) UTC.

    A session ending at end_hour on date D is "available" from end_hour on D.
    For wrap-around sessions (start_hour > end_hour, e.g. Asian 22→7), the
    session belongs to the date of end_hour.
    """
    _require_utc(ltf)
    ts = ltf["timestamp"]
    hour = ts.dt.hour
    date = ts.dt.normalize()
    if start_hour < end_hour:
        in_sess = (hour >= start_hour) & (hour < end_hour)
        sess_end = date  # session ends on same date
    else:
        # wrap: e.g. 22..(next day)07. Bars with hour >= start belong to session
        # ending on date+1; bars with hour < end belong to session ending on date.
        in_sess = (hour >= start_hour) | (hour < end_hour)
        sess_end = date.where(hour < end_hour, date + pd.Timedelta(days=1))

    df = pd.DataFrame({"date": sess_end, "high": ltf["high"], "low": ltf["low"], "in": in_sess})
    df_in = df[df["in"]].copy()
    if df_in.empty:
        empty = pd.Series(np.nan, index=ltf.index)
        return pd.DataFrame({"sess_high": empty, "sess_low": empty.copy()}, index=ltf.index)
    sess = df_in.groupby("date").agg(h=("high", "max"), l=("low", "min"))
    sess["h_prior"] = sess["h"].shift(1)
    sess["l_prior"] = sess["l"].shift(1)
    return pd.DataFrame({
        "sess_high": df["date"].map(sess["h_prior"]).to_numpy(),
        "sess_low": df["date"].map(sess["l_prior"]).to_numpy(),
    }, index=ltf.index)


def has_sr_confluence(close: pd.Series, atr_series: pd.Series,
                      pdh: pd.Series, pdl: pd.Series,
                      wkly_p: pd.Series, wkly_r1: pd.Series, wkly_s1: pd.Series,
                      round_step: float, tolerance_atr: float) -> pd.DataFrame:
    """For each bar, did at least one S/R level land within `tolerance_atr × ATR`
    of the bar's close (above or below)?

    Returns (long_sr, short_sr) where:
      - long_sr: True if any SUPPORT level (pdl, wkly_s1, wkly_p, round-below) is
        within tolerance of close from BELOW (close − level ∈ [0, tol]).
      - short_sr: True if any RESISTANCE level (pdh, wkly_r1, wkly_p, round-above)
        is within tolerance from ABOVE (level − close ∈ [0, tol]).
    """
    rl = round_level_distance(close, round_step)
    tol = (atr_series * float(tolerance_atr)).to_numpy()
    c = close.to_numpy()
    pdh_v = pdh.to_numpy(); pdl_v = pdl.to_numpy()
    p_v = wkly_p.to_numpy(); r1_v = wkly_r1.to_numpy(); s1_v = wkly_s1.to_numpy()
    ra = rl["level_above"].to_numpy(); rb = rl["level_below"].to_numpy()

    # Distance >= 0 and <= tolerance
    def _near_below(level):
        # level is BELOW close; close - level ∈ [0, tol]
        return (~np.isnan(level)) & (c >= level) & ((c - level) <= tol)
    def _near_above(level):
        return (~np.isnan(level)) & (level >= c) & ((level - c) <= tol)

    long_sr = (
        _near_below(pdl_v) | _near_below(s1_v) | _near_below(p_v) | _near_below(rb)
    )
    short_sr = (
        _near_above(pdh_v) | _near_above(r1_v) | _near_above(p_v) | _near_above(ra)
    )
    return pd.DataFrame({"long_sr": long_sr, "short_sr": short_sr}, index=close.index)
