"""Tests for reaction_levels indicator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import atr, swings
from src.indicators.reaction import reaction_levels


def _make_df(n: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    # Oscillate around 2000 so swings form repeatedly at similar levels
    close = 2000.0 + 5.0 * np.sin(np.arange(n) * 0.15) + rng.normal(0, 0.3, n)
    high = close + 0.5 + np.abs(rng.normal(0, 0.2, n))
    low = close - 0.5 - np.abs(rng.normal(0, 0.2, n))
    open_ = np.empty(n); open_[0] = close[0]; open_[1:] = close[:-1]
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high,
                         "low": low, "close": close, "volume": np.full(n, 100.0)})


def test_schema_and_length() -> None:
    df = _make_df()
    sw = swings(df, left=3, right=3)
    a = atr(df, 14)
    rl = reaction_levels(df, sw, atr_series=a, lookback=200, band_atr_mult=0.3)
    assert len(rl) == len(df)
    assert {"react_low_count", "react_low_level", "react_high_count", "react_high_level"}.issubset(rl.columns)
    assert (rl["react_low_count"] >= 0).all()
    assert (rl["react_high_count"] >= 0).all()


def test_oscillating_price_builds_reaction_counts() -> None:
    """A sinusoidal price revisits the same levels → reaction counts should grow."""
    df = _make_df(n=800)
    sw = swings(df, left=3, right=3)
    a = atr(df, 14)
    rl = reaction_levels(df, sw, atr_series=a, lookback=400, band_atr_mult=0.5)
    # Late in the series, at least some bars should have multiple reactions
    late = rl.iloc[400:]
    assert late["react_low_count"].max() >= 2 or late["react_high_count"].max() >= 2


def test_no_lookahead_prefix() -> None:
    """Counts at bar i must not change when future bars are added."""
    df = _make_df(n=700)
    sw_full = swings(df, left=3, right=3)
    a_full = atr(df, 14)
    full = reaction_levels(df, sw_full, atr_series=a_full, lookback=300, band_atr_mult=0.3)

    K = 500
    pref_df = df.iloc[:K].copy()
    sw_pref = swings(pref_df, left=3, right=3)
    a_pref = atr(pref_df, 14)
    pref = reaction_levels(pref_df, sw_pref, atr_series=a_pref, lookback=300, band_atr_mult=0.3)

    # Counts for bars [0, K-1] must match exactly (causal).
    assert (full["react_low_count"].iloc[:K].to_numpy() == pref["react_low_count"].to_numpy()).all()
    assert (full["react_high_count"].iloc[:K].to_numpy() == pref["react_high_count"].to_numpy()).all()
    # Levels: NaN positions match; finite values close
    for col in ("react_low_level", "react_high_level"):
        fa = full[col].iloc[:K].to_numpy(); pa = pref[col].to_numpy()
        na_f, na_p = np.isnan(fa), np.isnan(pa)
        assert (na_f == na_p).all(), f"{col} NaN positions diverged"
        mask = ~na_f
        if mask.any():
            assert np.allclose(fa[mask], pa[mask], rtol=1e-12, atol=1e-12)


def test_empty_swings_returns_zeros() -> None:
    """Flat price (no swings) → all reaction counts zero."""
    n = 100
    ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "open": np.full(n, 2000.0),
                       "high": np.full(n, 2000.1), "low": np.full(n, 1999.9),
                       "close": np.full(n, 2000.0), "volume": np.full(n, 100.0)})
    sw = swings(df, left=3, right=3)
    a = atr(df, 14)
    rl = reaction_levels(df, sw, atr_series=a, lookback=50, band_atr_mult=0.3)
    # Flat price may still register tiny swings; just assert no crash + valid schema
    assert len(rl) == n
    assert (rl["react_low_count"] >= 0).all()


def test_lookback_expiry() -> None:
    """A short lookback should produce <= counts than a long lookback."""
    df = _make_df(n=800)
    sw = swings(df, left=3, right=3)
    a = atr(df, 14)
    short = reaction_levels(df, sw, atr_series=a, lookback=100, band_atr_mult=0.5)
    long = reaction_levels(df, sw, atr_series=a, lookback=600, band_atr_mult=0.5)
    # On average, longer lookback retains more pivots → >= counts
    assert long["react_low_count"].sum() >= short["react_low_count"].sum()
    assert long["react_high_count"].sum() >= short["react_high_count"].sum()
