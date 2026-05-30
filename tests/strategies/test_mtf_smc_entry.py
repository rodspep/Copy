"""Tests for XauMtfSmcEntry — schema, cascade gates, no-lookahead."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.xau.mtf_smc_entry import XauMtfSmcEntry


def _synth_with_h4(n_per_seg: int = 5000, seed: int = 23) -> dict[str, pd.DataFrame]:
    """Build M5 + M15 + H4 with one uptrend then one downtrend. Need enough
    bars for H4 swings to confirm (a few weeks)."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_seg  # 10000 M5 bars = ~35 days
    ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    drift = np.concatenate([np.full(n_per_seg, 0.0004),
                            np.full(n_per_seg, -0.0004)])
    rets = drift + rng.normal(0.0, 0.0009, size=n)
    close = 2000.0 * np.exp(np.cumsum(rets))
    noise = np.abs(rng.normal(0.0, 0.0006, size=n)) * close
    high = close + noise
    low = close - noise
    open_ = np.empty(n); open_[0] = close[0]; open_[1:] = close[:-1]
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    vol = rng.uniform(50, 500, size=n)
    m5 = pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                       "close": close, "volume": vol})
    idx = m5.set_index("timestamp")
    m15 = idx.resample("15min", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna().reset_index()
    h4 = idx.resample("4h", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna().reset_index()
    return {"m5": m5, "m15": m15, "h4": h4}


def test_schema_valid() -> None:
    d = _synth_with_h4()
    strat = XauMtfSmcEntry()
    sigs = strat.generate_signals(d["m5"], {"M15": d["m15"], "H4": d["h4"]}).signals
    assert {"action", "sl", "tp"}.issubset(sigs.columns)
    assert len(sigs) == len(d["m5"])
    assert set(sigs["action"].unique()).issubset({"hold", "enter_long", "enter_short", "exit"})


def test_required_htfs_enforced() -> None:
    d = _synth_with_h4(n_per_seg=500)
    strat = XauMtfSmcEntry()
    with pytest.raises(ValueError, match="HTF"):
        strat.generate_signals(d["m5"], {"M15": d["m15"]})
    with pytest.raises(ValueError, match="HTF"):
        strat.generate_signals(d["m5"], {"H4": d["h4"]})


def test_deterministic() -> None:
    d = _synth_with_h4(n_per_seg=3000)
    strat = XauMtfSmcEntry()
    htfs = {"M15": d["m15"], "H4": d["h4"]}
    a = strat.generate_signals(d["m5"], htfs).signals
    b = strat.generate_signals(d["m5"], htfs).signals
    assert (a["action"].to_numpy() == b["action"].to_numpy()).all()


def test_no_lookahead_on_prefix() -> None:
    d = _synth_with_h4(n_per_seg=4000)
    m5, m15, h4 = d["m5"], d["m15"], d["h4"]
    strat = XauMtfSmcEntry()
    full = strat.generate_signals(m5, {"M15": m15, "H4": h4}).signals

    K = 6000
    last_avail = m5["timestamp"].iloc[K - 1] + pd.Timedelta(minutes=5)
    m15_pref = m15[m15["timestamp"] + pd.Timedelta(minutes=15) <= last_avail].copy()
    h4_pref = h4[h4["timestamp"] + pd.Timedelta(hours=4) <= last_avail].copy()
    pref = strat.generate_signals(m5.iloc[:K].copy(), {"M15": m15_pref, "H4": h4_pref}).signals

    fh = full.iloc[:K].reset_index(drop=True)
    pr = pref.reset_index(drop=True)
    assert (fh["action"].to_numpy() == pr["action"].to_numpy()).all(), (
        "lookahead leak — action column diverged"
    )
    for col in ("sl", "tp"):
        a = fh[col].to_numpy()
        b = pr[col].to_numpy()
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        assert (nan_a == nan_b).all(), f"{col} NaN positions diverged"
        mask = ~nan_a
        if mask.any():
            assert np.allclose(a[mask], b[mask], rtol=1e-10, atol=1e-10), f"{col} values diverged"


def test_cascade_sparser_than_smc_confluence() -> None:
    """Cascade architecture should produce ≤ as many signals as a single-gate
    setup. (We don't compare to smc_confluence directly here since synthetic
    may bias either way; just check the cascade is meaningfully selective.)"""
    d = _synth_with_h4(n_per_seg=4000)
    strat = XauMtfSmcEntry()
    sigs = strat.generate_signals(d["m5"], {"M15": d["m15"], "H4": d["h4"]}).signals
    n_signals = (sigs["action"] != "hold").sum()
    # Sparse: < 5% of M5 bars should be signals
    assert n_signals < 0.05 * len(d["m5"]), f"{n_signals} signals on {len(d['m5'])} bars is too dense"


def test_session_filter_excludes_off_hours() -> None:
    d = _synth_with_h4(n_per_seg=4000)
    strat = XauMtfSmcEntry()
    sigs = strat.generate_signals(d["m5"], {"M15": d["m15"], "H4": d["h4"]}, params={
        "session_filter": True, "trade_start_hour": 8, "trade_end_hour": 15,
    }).signals
    fire = sigs["action"] != "hold"
    if fire.sum() == 0:
        pytest.skip("no signals fired")
    hours = d["m5"]["timestamp"].dt.hour.to_numpy()[fire.to_numpy()]
    assert (hours >= 8).all() and (hours < 15).all()


def test_sl_tp_geometry() -> None:
    d = _synth_with_h4(n_per_seg=4000)
    strat = XauMtfSmcEntry()
    sigs = strat.generate_signals(d["m5"], {"M15": d["m15"], "H4": d["h4"]}).signals
    m5 = d["m5"].reset_index(drop=True)
    long_idx = sigs.index[sigs["action"] == "enter_long"]
    short_idx = sigs.index[sigs["action"] == "enter_short"]
    if len(long_idx) == 0 and len(short_idx) == 0:
        pytest.skip("no signals fired")
    for i in long_idx:
        assert sigs.at[i, "sl"] < m5.at[i, "close"], f"long SL not below close at {i}"
        assert sigs.at[i, "tp"] > m5.at[i, "close"], f"long TP not above close at {i}"
    for i in short_idx:
        assert sigs.at[i, "sl"] > m5.at[i, "close"]
        assert sigs.at[i, "tp"] < m5.at[i, "close"]
