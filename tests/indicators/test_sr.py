"""Tests for support/resistance helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.sr import (
    prior_day_high_low, weekly_pivot, round_level_distance,
    session_range, has_sr_confluence,
)


def _synth_3days(seed: int = 0) -> pd.DataFrame:
    """3 days of M5 with predictable highs/lows so we can hand-verify PDH/PDL."""
    rng = np.random.default_rng(seed)
    n = 3 * 24 * 12  # 3 days × 24h × 12 M5
    ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    close = 2000.0 + rng.normal(0, 1, size=n).cumsum() * 0.1
    high = close + 0.5
    low = close - 0.5
    open_ = close.copy()
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                         "close": close, "volume": np.ones(n) * 100})


# -------- PDH/PDL --------

def test_pdh_pdl_NaN_first_day() -> None:
    df = _synth_3days()
    out = prior_day_high_low(df)
    first_day_mask = df["timestamp"].dt.normalize() == pd.Timestamp("2025-01-01", tz="UTC")
    assert out.loc[first_day_mask, "pdh"].isna().all()
    assert out.loc[first_day_mask, "pdl"].isna().all()


def test_pdh_pdl_matches_prior_day_aggregation() -> None:
    df = _synth_3days()
    out = prior_day_high_low(df)
    # Day-2 PDH should equal day-1 max(high)
    day1_high = df[df["timestamp"].dt.normalize() == pd.Timestamp("2025-01-01", tz="UTC")]["high"].max()
    day2_mask = df["timestamp"].dt.normalize() == pd.Timestamp("2025-01-02", tz="UTC")
    assert np.allclose(out.loc[day2_mask, "pdh"].unique(), [day1_high])


def test_pdh_pdl_requires_utc() -> None:
    df = _synth_3days()
    df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    with pytest.raises(ValueError, match="UTC"):
        prior_day_high_low(df)


# -------- Weekly pivot --------

def test_weekly_pivot_computes_finite_values() -> None:
    """Need ≥ 2 ISO weeks of data for the second week to have a prior-week pivot."""
    rng = np.random.default_rng(1)
    n = 14 * 24 * 12  # 2 weeks of M5
    ts = pd.date_range("2025-01-06", periods=n, freq="5min", tz="UTC")  # Monday
    close = 2000.0 + rng.normal(0, 1, size=n).cumsum() * 0.1
    df = pd.DataFrame({"timestamp": ts, "open": close, "high": close + 0.5,
                       "low": close - 0.5, "close": close, "volume": np.ones(n)})
    out = weekly_pivot(df)
    # Second-week bars should have finite pivot
    second_week_mask = df["timestamp"] >= pd.Timestamp("2025-01-13", tz="UTC")
    assert out.loc[second_week_mask, "wkly_p"].notna().any()


# -------- Round levels --------

def test_round_level_distance_basic() -> None:
    close = pd.Series([2003.5, 2010.0, 2017.2, 2050.0])
    out = round_level_distance(close, step=10.0)
    assert list(out["level_above"]) == [2010.0, 2010.0, 2020.0, 2050.0]
    assert list(out["level_below"]) == [2000.0, 2010.0, 2010.0, 2050.0]
    assert np.allclose(out["dist_above"], [6.5, 0.0, 2.8, 0.0])
    assert np.allclose(out["dist_below"], [3.5, 0.0, 7.2, 0.0])


def test_round_level_step_must_be_positive() -> None:
    with pytest.raises(ValueError):
        round_level_distance(pd.Series([2000.0]), step=0)


# -------- Session range --------

def test_session_range_london() -> None:
    df = _synth_3days()
    out = session_range(df, start_hour=7, end_hour=12)
    # London session for day-1 = bars 07-12 UTC on day-1. Day-2 bars after 12 should
    # see those values as "sess_high/low" of the prior session.
    day2_after_12 = (df["timestamp"] >= pd.Timestamp("2025-01-02 12:00", tz="UTC")) & \
                    (df["timestamp"] < pd.Timestamp("2025-01-02 23:00", tz="UTC"))
    if day2_after_12.any():
        vals = out.loc[day2_after_12, "sess_high"].dropna().unique()
        assert len(vals) >= 1


def test_session_range_asian_wraps() -> None:
    """Asian session 22..07 wraps midnight — make sure the helper doesn't crash."""
    df = _synth_3days()
    out = session_range(df, start_hour=22, end_hour=7)
    assert len(out) == len(df)


# -------- has_sr_confluence --------

def test_has_sr_confluence_long_when_close_just_above_pdl() -> None:
    """If price sits 1 ATR above PDL, long_sr should be True (PDL = support nearby)."""
    n = 10
    close = pd.Series([2010.0] * n)
    atr_s = pd.Series([1.0] * n)
    pdh = pd.Series([2020.0] * n)
    pdl = pd.Series([2009.0] * n)        # close − pdl = 1 == 1 × ATR
    wp = pd.Series([np.nan] * n); r1 = pd.Series([np.nan] * n); s1 = pd.Series([np.nan] * n)
    out = has_sr_confluence(close, atr_s, pdh, pdl, wp, r1, s1, round_step=50.0, tolerance_atr=1.0)
    assert out["long_sr"].all()
    assert not out["short_sr"].any()  # PDH 10 above close, > 1 ATR away


def test_has_sr_confluence_short_when_close_just_below_pdh() -> None:
    n = 10
    close = pd.Series([2019.0] * n)
    atr_s = pd.Series([1.0] * n)
    pdh = pd.Series([2020.0] * n)         # 1 = 1 × ATR away
    pdl = pd.Series([1995.0] * n)
    wp = pd.Series([np.nan] * n); r1 = pd.Series([np.nan] * n); s1 = pd.Series([np.nan] * n)
    out = has_sr_confluence(close, atr_s, pdh, pdl, wp, r1, s1, round_step=50.0, tolerance_atr=1.0)
    assert out["short_sr"].all()
    assert not out["long_sr"].any()
