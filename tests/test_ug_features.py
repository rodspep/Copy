"""Tests for the UG feature/validator engine — correctness + no-lookahead."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.features import asof_index, build_feature_matrix, DEFAULT_CFG
from src.analysis.signals import Signal, normalize_direction


def _synth_ohlc(n=600, start="2026-05-01T00:00:00Z", freq="5min", seed=7):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    steps = rng.normal(0, 1.0, n).cumsum()
    close = 2600 + steps
    open_ = close - rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.4, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.4, n))
    vol = rng.integers(100, 1000, n).astype(float)
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high,
                         "low": low, "close": close, "volume": vol})


def test_asof_index():
    df = _synth_ohlc(n=10)
    ts = df["timestamp"]
    assert asof_index(ts, ts.iloc[0] - pd.Timedelta("1min")) is None      # before all
    assert asof_index(ts, ts.iloc[5]) == 5                                # exact hit
    assert asof_index(ts, ts.iloc[5] + pd.Timedelta("2min")) == 5         # between → prior
    assert asof_index(ts, ts.iloc[-1] + pd.Timedelta("1h")) == len(df) - 1  # after all


def test_normalize_direction():
    assert normalize_direction("BUY") == "long"
    assert normalize_direction("Sell") == "short"
    assert normalize_direction("mua") == "long"
    with pytest.raises(ValueError):
        normalize_direction("maybe")


def test_no_lookahead_uses_only_closed_bars():
    """Strict rule: the selected bar must have CLOSED at-or-before the signal,
    i.e. bar_open + timeframe <= signal_ts. A mid-bar signal reads the PRIOR bar."""
    df = _synth_ohlc()                       # 5min bars
    tf = pd.Timedelta("5min")
    # mid-bar signals (open + 2min): the forming bar i must NOT be used → bar i-1.
    midbar = [Signal(ts=df["timestamp"].iloc[i] + pd.Timedelta("2min"),
                     direction="long" if i % 2 else "short")
              for i in (250, 300, 350, 400)]
    fm = build_feature_matrix(midbar, df)
    assert len(fm) == len(midbar)
    for _, row in fm.iterrows():
        sel, sig = pd.Timestamp(row["ts"]), pd.Timestamp(row["signal_ts"])
        assert sel + tf <= sig               # selected bar fully closed before signal
    # a signal exactly at bar i's CLOSE time selects bar i (it has just closed).
    i = 320
    s = Signal(ts=df["timestamp"].iloc[i] + tf, direction="long")
    fm2 = build_feature_matrix([s], df)
    assert pd.Timestamp(fm2["ts"].iloc[0]) == df["timestamp"].iloc[i]


def test_skips_signals_before_data():
    df = _synth_ohlc()
    early = Signal(ts=df["timestamp"].iloc[0] - pd.Timedelta("1h"), direction="long")
    inrange = Signal(ts=df["timestamp"].iloc[300], direction="short")
    fm = build_feature_matrix([early, inrange], df)
    assert len(fm) == 1
    assert fm.attrs["skipped"] == 1


def test_geometry_features():
    df = _synth_ohlc()
    i = 300
    px = float(df["close"].iloc[i])
    # signal at bar i's CLOSE → selected bar is i; entry=close[i] → entry_minus_close≈0.
    # risk 3, reward 6 → rr = 2.0
    s = Signal(ts=df["timestamp"].iloc[i] + pd.Timedelta("5min"), direction="long",
               entry=px, sl=px - 3.0, tp=px + 6.0)
    fm = build_feature_matrix([s], df)
    assert pd.Timestamp(fm["ts"].iloc[0]) == df["timestamp"].iloc[i]
    assert fm["rr"].iloc[0] == pytest.approx(2.0, rel=1e-6)
    assert fm["entry_minus_close_atr"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_expected_feature_columns_present():
    df = _synth_ohlc()
    s = Signal(ts=df["timestamp"].iloc[300], direction="long")
    fm = build_feature_matrix([s], df)
    for col in ("rsi", "adx", "atr", "struct_trend", "session",
                "bull_candle", "ema50_gt_ema100", "dist_last_swing_low_atr",
                "recent_bos", "in_bull_ob", "vol_vs_ma", "hour_utc"):
        assert col in fm.columns, f"missing feature {col}"
