"""Tests for XauReactionLevel strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.xau.reaction_level import XauReactionLevel


def _synth(n_per_seg: int = 2500, seed: int = 51) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = 2 * n_per_seg
    ts = pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC")
    # Trend + oscillation so reaction levels form and HTF trend exists
    drift = np.concatenate([np.full(n_per_seg, 0.0004), np.full(n_per_seg, -0.0004)])
    osc = 3.0 * np.sin(np.arange(n) * 0.08)
    rets = drift + rng.normal(0, 0.0007, size=n)
    close = 2000.0 * np.exp(np.cumsum(rets)) + osc
    noise = np.abs(rng.normal(0, 0.0005, size=n)) * close
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
    sigs = XauReactionLevel().generate_signals(d["m5"], {"H1": d["h1"]}).signals
    assert {"action", "sl", "tp"}.issubset(sigs.columns)
    assert len(sigs) == len(d["m5"])
    assert set(sigs["action"].unique()).issubset({"hold", "enter_long", "enter_short", "exit"})


def test_required_htf() -> None:
    d = _synth(n_per_seg=400)
    with pytest.raises(ValueError, match="HTF"):
        XauReactionLevel().generate_signals(d["m5"], {})


def test_deterministic() -> None:
    d = _synth(n_per_seg=1500)
    htfs = {"H1": d["h1"]}
    a = XauReactionLevel().generate_signals(d["m5"], htfs).signals
    b = XauReactionLevel().generate_signals(d["m5"], htfs).signals
    assert (a["action"].to_numpy() == b["action"].to_numpy()).all()


def test_no_lookahead_on_prefix() -> None:
    d = _synth(n_per_seg=2000)
    m5, h1 = d["m5"], d["h1"]
    full = XauReactionLevel().generate_signals(m5, {"H1": h1}).signals
    K = 3000
    last_avail = m5["timestamp"].iloc[K-1] + pd.Timedelta(minutes=5)
    h1_pref = h1[h1["timestamp"] + pd.Timedelta(hours=1) <= last_avail].copy()
    pref = XauReactionLevel().generate_signals(m5.iloc[:K].copy(), {"H1": h1_pref}).signals
    fh = full.iloc[:K].reset_index(drop=True)
    pr = pref.reset_index(drop=True)
    assert (fh["action"].to_numpy() == pr["action"].to_numpy()).all(), "lookahead leak"
    for col in ("sl", "tp"):
        a = fh[col].to_numpy(); b = pr[col].to_numpy()
        na, nb = np.isnan(a), np.isnan(b)
        assert (na == nb).all()
        mask = ~na
        if mask.any():
            assert np.allclose(a[mask], b[mask], rtol=1e-9, atol=1e-9), f"{col} diverged"


def test_min_reactions_reduces_signals() -> None:
    """Higher min_reactions threshold must produce <= signals."""
    d = _synth(n_per_seg=2000)
    htfs = {"H1": d["h1"]}
    counts = []
    for mr in (2, 4, 6):
        sigs = XauReactionLevel().generate_signals(d["m5"], htfs, params={
            "min_reactions": mr, "require_h1_trend": False,
            "session_filter": False, "require_confirm_candle": False,
        }).signals
        counts.append((sigs["action"] != "hold").sum())
    for i in range(1, len(counts)):
        assert counts[i] <= counts[i-1], f"min_reactions={2*(i+1)} gave more signals"


def test_rr_geometry() -> None:
    d = _synth(n_per_seg=2500)
    sigs = XauReactionLevel().generate_signals(d["m5"], {"H1": d["h1"]},
                                               params={"tp_rr": 2.0}).signals
    m5 = d["m5"]
    enter_idx = sigs.index[sigs["action"].isin(["enter_long", "enter_short"])]
    if len(enter_idx) == 0:
        pytest.skip("no signals")
    for i in enter_idx[:10]:
        close = m5.at[i, "close"]
        sl_d = abs(close - sigs.at[i, "sl"]); tp_d = abs(close - sigs.at[i, "tp"])
        if sl_d == 0: continue
        assert abs(tp_d / sl_d - 2.0) < 0.05, f"R:R off at {i}"


def test_sl_tp_geometry() -> None:
    d = _synth(n_per_seg=2500)
    sigs = XauReactionLevel().generate_signals(d["m5"], {"H1": d["h1"]}).signals
    m5 = d["m5"]
    for i in sigs.index[sigs["action"] == "enter_long"]:
        assert sigs.at[i, "sl"] < m5.at[i, "close"]
        assert sigs.at[i, "tp"] > m5.at[i, "close"]
    for i in sigs.index[sigs["action"] == "enter_short"]:
        assert sigs.at[i, "sl"] > m5.at[i, "close"]
        assert sigs.at[i, "tp"] < m5.at[i, "close"]
