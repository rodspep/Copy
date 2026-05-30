"""Tests for XauScalpFade (Method A) and XauDeepPullback (Method B)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.xau.ug_methods import XauScalpFade, XauDeepPullback


def _synth(n_per_seg=2500, seed=61):
    rng = np.random.default_rng(seed)
    n = 2 * n_per_seg
    ts = pd.date_range("2025-04-01", periods=n, freq="5min", tz="UTC")
    drift = np.concatenate([np.full(n_per_seg, 0.0004), np.full(n_per_seg, -0.0004)])
    osc = 3.0 * np.sin(np.arange(n) * 0.07)
    rets = drift + rng.normal(0, 0.0008, n)
    close = 2000.0 * np.exp(np.cumsum(rets)) + osc
    noise = np.abs(rng.normal(0, 0.0006, n)) * close
    high = close + noise; low = close - noise
    open_ = np.empty(n); open_[0] = close[0]; open_[1:] = close[:-1]
    high = np.maximum(high, np.maximum(open_, close)); low = np.minimum(low, np.minimum(open_, close))
    m5 = pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                       "close": close, "volume": rng.uniform(50, 500, n)})
    h1 = m5.set_index("timestamp").resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
    return {"m5": m5, "h1": h1}


def test_scalpfade_schema_and_nolookahead():
    d = _synth()
    s = XauScalpFade()
    full = s.generate_signals(d["m5"], {}).signals
    assert {"action", "sl", "tp"}.issubset(full.columns) and len(full) == len(d["m5"])
    K = 3000
    pref = s.generate_signals(d["m5"].iloc[:K].copy(), {}).signals
    assert (full["action"].iloc[:K].to_numpy() == pref["action"].to_numpy()).all()


def test_deeppullback_schema_and_nolookahead():
    d = _synth()
    s = XauDeepPullback()
    full = s.generate_signals(d["m5"], {"H1": d["h1"]}).signals
    assert {"action", "sl", "tp"}.issubset(full.columns) and len(full) == len(d["m5"])
    K = 3000
    last = d["m5"]["timestamp"].iloc[K-1] + pd.Timedelta(minutes=5)
    h1p = d["h1"][d["h1"]["timestamp"] + pd.Timedelta(hours=1) <= last].copy()
    pref = s.generate_signals(d["m5"].iloc[:K].copy(), {"H1": h1p}).signals
    assert (full["action"].iloc[:K].to_numpy() == pref["action"].to_numpy()).all()


def test_deeppullback_requires_h1():
    d = _synth(400)
    with pytest.raises(ValueError, match="HTF"):
        XauDeepPullback().generate_signals(d["m5"], {})


def test_rr_geometry_both():
    d = _synth()
    # Method A: tp_rr 0.5
    sa = XauScalpFade().generate_signals(d["m5"], {}, params={"tp_rr": 0.5}).signals
    m5 = d["m5"]
    for i in sa.index[sa["action"].isin(["enter_long", "enter_short"])][:10]:
        sl_d = abs(m5.at[i, "close"] - sa.at[i, "sl"]); tp_d = abs(m5.at[i, "close"] - sa.at[i, "tp"])
        if sl_d > 0:
            assert abs(tp_d / sl_d - 0.5) < 0.05
    # Method B: tp_rr 1.5
    sb = XauDeepPullback().generate_signals(d["m5"], {"H1": d["h1"]}, params={"tp_rr": 1.5}).signals
    for i in sb.index[sb["action"].isin(["enter_long", "enter_short"])][:10]:
        sl_d = abs(m5.at[i, "close"] - sb.at[i, "sl"]); tp_d = abs(m5.at[i, "close"] - sb.at[i, "tp"])
        if sl_d > 0:
            assert abs(tp_d / sl_d - 1.5) < 0.05
