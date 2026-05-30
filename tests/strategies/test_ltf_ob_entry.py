"""Tests for XauLtfObEntry."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.xau.ltf_ob_entry import XauLtfObEntry


def _synth(n_per_seg: int = 3000, seed: int = 41) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = 2 * n_per_seg
    ts = pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC")
    drift = np.concatenate([np.full(n_per_seg, 0.0005), np.full(n_per_seg, -0.0005)])
    rets = drift + rng.normal(0, 0.0008, size=n)
    close = 2000.0 * np.exp(np.cumsum(rets))
    noise = np.abs(rng.normal(0, 0.0006, size=n)) * close
    high = close + noise; low = close - noise
    open_ = np.empty(n); open_[0] = close[0]; open_[1:] = close[:-1]
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    m5 = pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                       "close": close, "volume": rng.uniform(50, 500, n)})
    h1 = m5.set_index("timestamp").resample("1h", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna().reset_index()
    return {"m5": m5, "h1": h1}


def test_schema_valid() -> None:
    d = _synth()
    sigs = XauLtfObEntry().generate_signals(d["m5"], {"H1": d["h1"]}).signals
    assert {"action","sl","tp"}.issubset(sigs.columns)
    assert len(sigs) == len(d["m5"])


def test_required_htf() -> None:
    d = _synth(n_per_seg=500)
    with pytest.raises(ValueError, match="HTF"):
        XauLtfObEntry().generate_signals(d["m5"], {})


def test_deterministic() -> None:
    d = _synth(n_per_seg=2000)
    htfs = {"H1": d["h1"]}
    a = XauLtfObEntry().generate_signals(d["m5"], htfs).signals
    b = XauLtfObEntry().generate_signals(d["m5"], htfs).signals
    assert (a["action"].to_numpy() == b["action"].to_numpy()).all()


def test_no_lookahead_on_prefix() -> None:
    d = _synth(n_per_seg=2500)
    m5, h1 = d["m5"], d["h1"]
    full = XauLtfObEntry().generate_signals(m5, {"H1": h1}).signals
    K = 3500
    last_avail = m5["timestamp"].iloc[K-1] + pd.Timedelta(minutes=5)
    h1_pref = h1[h1["timestamp"] + pd.Timedelta(hours=1) <= last_avail].copy()
    pref = XauLtfObEntry().generate_signals(m5.iloc[:K].copy(), {"H1": h1_pref}).signals
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


def test_rr_geometry() -> None:
    """TP distance should equal tp_rr × SL distance."""
    d = _synth(n_per_seg=2500)
    sigs = XauLtfObEntry().generate_signals(d["m5"], {"H1": d["h1"]},
                                            params={"tp_rr": 2.0}).signals
    m5 = d["m5"]
    enter_idx = sigs.index[sigs["action"].isin(["enter_long","enter_short"])]
    if len(enter_idx) == 0:
        pytest.skip("no signals fired")
    for i in enter_idx[:10]:
        close = m5.at[i, "close"]
        sl_dist = abs(close - sigs.at[i, "sl"])
        tp_dist = abs(close - sigs.at[i, "tp"])
        if sl_dist == 0: continue
        rr = tp_dist / sl_dist
        assert abs(rr - 2.0) < 0.05, f"R:R != 2.0 at bar {i}: got {rr:.3f}"


def test_sl_tp_geometry() -> None:
    d = _synth(n_per_seg=2500)
    sigs = XauLtfObEntry().generate_signals(d["m5"], {"H1": d["h1"]}).signals
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
