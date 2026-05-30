"""Tests for dxy_block_opposite_wrapper."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.signal_wrappers import dxy_block_opposite_wrapper
from src.strategies.base import empty_signals


def _make_xau() -> pd.DataFrame:
    n = 200
    ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": np.full(n, 2000.0), "high": np.full(n, 2001.0),
        "low": np.full(n, 1999.0), "close": np.full(n, 2000.0),
        "volume": np.full(n, 100.0),
    })


def _make_dxy(rally: bool) -> pd.DataFrame:
    """Daily DXY data with strong rally or drop."""
    n = 10
    ts = pd.date_range("2024-12-25", periods=n, freq="1D", tz="UTC")
    if rally:
        close = 100.0 + np.arange(n) * 1.0  # +1 per day = +10% over window
    else:
        close = 110.0 - np.arange(n) * 1.0  # -1 per day = -10%
    return pd.DataFrame({
        "timestamp": ts, "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.full(n, 1000.0),
    })


def test_blocks_long_when_dxy_rallies() -> None:
    ltf = _make_xau()
    dxy = _make_dxy(rally=True)
    sigs = empty_signals(ltf)
    sigs.at[100, "action"] = "enter_long"
    sigs.at[100, "sl"] = 1995.0
    sigs.at[100, "tp"] = 2010.0
    out = dxy_block_opposite_wrapper(ltf, sigs, dxy, slope_window=5, block_threshold_pct=2.0)
    assert out.at[100, "action"] == "hold", "long should be blocked when DXY rallies"
    assert np.isnan(out.at[100, "sl"]) and np.isnan(out.at[100, "tp"])


def test_blocks_short_when_dxy_drops() -> None:
    ltf = _make_xau()
    dxy = _make_dxy(rally=False)
    sigs = empty_signals(ltf)
    sigs.at[100, "action"] = "enter_short"
    sigs.at[100, "sl"] = 2005.0
    sigs.at[100, "tp"] = 1990.0
    out = dxy_block_opposite_wrapper(ltf, sigs, dxy, slope_window=5, block_threshold_pct=2.0)
    assert out.at[100, "action"] == "hold", "short should be blocked when DXY drops"


def test_keeps_long_when_dxy_drops() -> None:
    """DXY drops = USD weak = XAU long FAVORED. Don't block."""
    ltf = _make_xau()
    dxy = _make_dxy(rally=False)
    sigs = empty_signals(ltf)
    sigs.at[100, "action"] = "enter_long"
    sigs.at[100, "sl"] = 1995.0
    sigs.at[100, "tp"] = 2010.0
    out = dxy_block_opposite_wrapper(ltf, sigs, dxy, slope_window=5, block_threshold_pct=2.0)
    assert out.at[100, "action"] == "enter_long", "long should NOT be blocked when DXY drops"


def test_keeps_signal_when_dxy_neutral() -> None:
    """DXY flat → block-opposite condition not triggered."""
    ltf = _make_xau()
    n = 10
    ts = pd.date_range("2024-12-25", periods=n, freq="1D", tz="UTC")
    flat_dxy = pd.DataFrame({
        "timestamp": ts, "open": np.full(n, 100.0), "high": np.full(n, 100.5),
        "low": np.full(n, 99.5), "close": np.full(n, 100.0), "volume": np.full(n, 1000.0),
    })
    sigs = empty_signals(ltf)
    sigs.at[100, "action"] = "enter_long"
    sigs.at[100, "sl"] = 1995.0
    sigs.at[100, "tp"] = 2010.0
    out = dxy_block_opposite_wrapper(ltf, sigs, flat_dxy, slope_window=5, block_threshold_pct=0.5)
    assert out.at[100, "action"] == "enter_long"


def test_schema_preserved() -> None:
    ltf = _make_xau()
    dxy = _make_dxy(rally=True)
    sigs = empty_signals(ltf)
    sigs.at[50, "action"] = "enter_long"
    sigs.at[50, "sl"] = 1995.0
    sigs.at[50, "tp"] = 2010.0
    sigs.at[100, "action"] = "enter_short"
    sigs.at[100, "sl"] = 2005.0
    sigs.at[100, "tp"] = 1990.0
    out = dxy_block_opposite_wrapper(ltf, sigs, dxy, slope_window=5, block_threshold_pct=2.0)
    assert len(out) == len(ltf)
    assert set(out["action"].unique()).issubset({"hold","enter_long","enter_short","exit"})
