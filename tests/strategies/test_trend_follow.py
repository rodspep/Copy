"""Tests for XauTrendFollow (H4 trend-following, stop-and-reverse)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.xau.trend_follow import XauTrendFollow


def _synth_h4(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    # trend up then down then up
    seg = n // 3
    drift = np.concatenate([np.full(seg, 0.002), np.full(seg, -0.002), np.full(n - 2 * seg, 0.002)])
    rets = drift + rng.normal(0, 0.004, n)
    close = 2000 * np.exp(np.cumsum(rets))
    noise = np.abs(rng.normal(0, 0.003, n)) * close
    high = close + noise; low = close - noise
    open_ = np.empty(n); open_[0] = close[0]; open_[1:] = close[:-1]
    high = np.maximum(high, np.maximum(open_, close)); low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                         "close": close, "volume": np.full(n, 100.0)})


def test_schema_and_actions():
    df = _synth_h4()
    s = XauTrendFollow().generate_signals(df, {}).signals
    assert {"action", "sl", "tp"}.issubset(s.columns) and len(s) == len(df)
    assert set(s["action"].unique()).issubset({"hold", "enter_long", "enter_short", "exit"})
    # a trending synthetic should produce at least one entry and one exit
    assert (s["action"].isin(["enter_long", "enter_short"])).any()


def test_no_lookahead_prefix():
    df = _synth_h4()
    full = XauTrendFollow().generate_signals(df, {}).signals
    K = 800
    pref = XauTrendFollow().generate_signals(df.iloc[:K].copy(), {}).signals
    assert (full["action"].iloc[:K].to_numpy() == pref["action"].to_numpy()).all()
    for col in ("sl", "tp"):
        a = full[col].iloc[:K].to_numpy(); b = pref[col].to_numpy()
        na, nb = np.isnan(a), np.isnan(b)
        assert (na == nb).all()
        m = ~na
        if m.any():
            assert np.allclose(a[m], b[m], rtol=1e-9, atol=1e-9)


def test_deterministic():
    df = _synth_h4()
    a = XauTrendFollow().generate_signals(df, {}).signals
    b = XauTrendFollow().generate_signals(df, {}).signals
    assert (a["action"].to_numpy() == b["action"].to_numpy()).all()


def test_donchian_mode_runs():
    df = _synth_h4()
    s = XauTrendFollow().generate_signals(df, {}, params={"entry_mode": "donchian", "donchian_n": 20}).signals
    assert (s["action"] != "hold").any()


def test_no_short_when_disabled():
    df = _synth_h4()
    s = XauTrendFollow().generate_signals(df, {}, params={"allow_short": False}).signals
    assert (s["action"] == "enter_short").sum() == 0


def test_sl_geometry():
    df = _synth_h4()
    s = XauTrendFollow().generate_signals(df, {}).signals
    for i in s.index[s["action"] == "enter_long"]:
        assert s.at[i, "sl"] < df.at[i, "close"]
    for i in s.index[s["action"] == "enter_short"]:
        assert s.at[i, "sl"] > df.at[i, "close"]
