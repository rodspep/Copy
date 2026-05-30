"""Tests for XauHtfTrendReversal — schema, no-lookahead, gates, geometry."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.xau.htf_trend_reversal import XauHtfTrendReversal


def _synth(n_per_seg: int = 3000, seed: int = 31) -> dict[str, pd.DataFrame]:
    """Two trending segments + M5/H1 OHLCV."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_seg
    ts = pd.date_range("2025-03-01", periods=n, freq="5min", tz="UTC")
    drift = np.concatenate([np.full(n_per_seg, 0.0005), np.full(n_per_seg, -0.0005)])
    rets = drift + rng.normal(0.0, 0.0008, size=n)
    close = 2000.0 * np.exp(np.cumsum(rets))
    noise = np.abs(rng.normal(0.0, 0.0006, size=n)) * close
    high = close + noise; low = close - noise
    open_ = np.empty(n); open_[0] = close[0]; open_[1:] = close[:-1]
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    vol = rng.uniform(50, 500, size=n)
    m5 = pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                       "close": close, "volume": vol})
    h1 = m5.set_index("timestamp").resample("1h", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna().reset_index()
    return {"m5": m5, "h1": h1}


def test_schema_valid() -> None:
    d = _synth()
    sigs = XauHtfTrendReversal().generate_signals(d["m5"], {"H1": d["h1"]}).signals
    assert {"action","sl","tp"}.issubset(sigs.columns)
    assert len(sigs) == len(d["m5"])
    assert set(sigs["action"].unique()).issubset({"hold","enter_long","enter_short","exit"})


def test_required_htf() -> None:
    d = _synth(n_per_seg=500)
    with pytest.raises(ValueError, match="HTF"):
        XauHtfTrendReversal().generate_signals(d["m5"], {})


def test_deterministic() -> None:
    d = _synth(n_per_seg=2000)
    htfs = {"H1": d["h1"]}
    a = XauHtfTrendReversal().generate_signals(d["m5"], htfs).signals
    b = XauHtfTrendReversal().generate_signals(d["m5"], htfs).signals
    assert (a["action"].to_numpy() == b["action"].to_numpy()).all()


def test_no_lookahead_on_prefix() -> None:
    d = _synth(n_per_seg=2500)
    m5, h1 = d["m5"], d["h1"]
    full = XauHtfTrendReversal().generate_signals(m5, {"H1": h1}).signals
    K = 3500
    last_avail = m5["timestamp"].iloc[K-1] + pd.Timedelta(minutes=5)
    h1_pref = h1[h1["timestamp"] + pd.Timedelta(hours=1) <= last_avail].copy()
    pref = XauHtfTrendReversal().generate_signals(m5.iloc[:K].copy(), {"H1": h1_pref}).signals
    fh = full.iloc[:K].reset_index(drop=True)
    pr = pref.reset_index(drop=True)
    assert (fh["action"].to_numpy() == pr["action"].to_numpy()).all()
    for col in ("sl", "tp"):
        a = fh[col].to_numpy(); b = pr[col].to_numpy()
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        assert (nan_a == nan_b).all()
        mask = ~nan_a
        if mask.any():
            assert np.allclose(a[mask], b[mask], rtol=1e-10, atol=1e-10), f"{col} diverged"


def test_gates_off_more_signals() -> None:
    d = _synth(n_per_seg=2500)
    htfs = {"H1": d["h1"]}
    base = {"h1_adx_min": 0.0, "require_h1_above_fast": False,
            "session_filter": False,
            "pattern_engulfing": True, "pattern_pin_bar": True, "pattern_rsi_cross": True}
    n_open = (XauHtfTrendReversal().generate_signals(d["m5"], htfs, base).signals["action"] != "hold").sum()
    # Turn ADX strict on
    p = dict(base); p["h1_adx_min"] = 25.0
    n_strict = (XauHtfTrendReversal().generate_signals(d["m5"], htfs, p).signals["action"] != "hold").sum()
    assert n_strict <= n_open, f"strict ADX should not produce more signals ({n_strict} vs {n_open})"


def test_sl_tp_geometry() -> None:
    d = _synth(n_per_seg=2500)
    sigs = XauHtfTrendReversal().generate_signals(d["m5"], {"H1": d["h1"]}).signals
    m5 = d["m5"]
    long_idx = sigs.index[sigs["action"] == "enter_long"]
    short_idx = sigs.index[sigs["action"] == "enter_short"]
    if len(long_idx) == 0 and len(short_idx) == 0:
        pytest.skip("no signals fired")
    for i in long_idx:
        assert sigs.at[i, "sl"] < m5.at[i, "close"]
        assert sigs.at[i, "tp"] > m5.at[i, "close"]
    for i in short_idx:
        assert sigs.at[i, "sl"] > m5.at[i, "close"]
        assert sigs.at[i, "tp"] < m5.at[i, "close"]


def test_fixed_rr_holds() -> None:
    """With tp_rr=1.0, |entry-tp| should equal |entry-sl|."""
    d = _synth(n_per_seg=2500)
    sigs = XauHtfTrendReversal().generate_signals(d["m5"], {"H1": d["h1"]},
                                                  params={"tp_rr": 1.0}).signals
    m5 = d["m5"]
    enter_idx = sigs.index[sigs["action"].isin(["enter_long","enter_short"])]
    if len(enter_idx) == 0:
        pytest.skip("no signals fired")
    for i in enter_idx[:10]:
        close = m5.at[i, "close"]
        sl_dist = abs(close - sigs.at[i, "sl"])
        tp_dist = abs(close - sigs.at[i, "tp"])
        assert abs(sl_dist - tp_dist) / sl_dist < 0.02, f"R:R not 1.0 at bar {i}: sl={sl_dist:.4f} tp={tp_dist:.4f}"
