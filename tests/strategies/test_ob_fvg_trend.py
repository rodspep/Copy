"""Tests for XauObFvgTrend (OB∩FVG overlap, trend-aligned, wide-TP)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.xau.ob_fvg_trend import XauObFvgTrend


def _synth_h1(n=1500, seed=11):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    seg = n // 3
    drift = np.concatenate([np.full(seg, 0.0015), np.full(seg, -0.0015),
                            np.full(n - 2 * seg, 0.0015)])
    rets = drift + rng.normal(0, 0.004, n)
    close = 2000 * np.exp(np.cumsum(rets))
    noise = np.abs(rng.normal(0, 0.003, n)) * close
    high = close + noise; low = close - noise
    open_ = np.empty(n); open_[0] = close[0]; open_[1:] = close[:-1]
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                         "close": close, "volume": np.full(n, 100.0)})


def test_schema_and_actions():
    df = _synth_h1()
    s = XauObFvgTrend().generate_signals(df, {}).signals
    assert {"action", "sl", "tp"}.issubset(s.columns) and len(s) == len(df)
    assert set(s["action"].unique()).issubset({"hold", "enter_long", "enter_short"})


def test_no_lookahead_prefix():
    df = _synth_h1()
    full = XauObFvgTrend().generate_signals(df, {}).signals
    K = 1000
    pref = XauObFvgTrend().generate_signals(df.iloc[:K].copy(), {}).signals
    assert (full["action"].iloc[:K].to_numpy() == pref["action"].to_numpy()).all()
    for col in ("sl", "tp"):
        a = full[col].iloc[:K].to_numpy(); b = pref[col].to_numpy()
        na, nb = np.isnan(a), np.isnan(b)
        assert (na == nb).all()
        m = ~na
        if m.any():
            assert np.allclose(a[m], b[m], rtol=1e-9, atol=1e-9)


def test_deterministic():
    df = _synth_h1()
    a = XauObFvgTrend().generate_signals(df, {}).signals
    b = XauObFvgTrend().generate_signals(df, {}).signals
    assert (a["action"].to_numpy() == b["action"].to_numpy()).all()


def test_sl_tp_geometry():
    df = _synth_h1()
    s = XauObFvgTrend().generate_signals(df, {}).signals
    rr = XauObFvgTrend.default_params["tp_rr"]
    for i in s.index[s["action"] == "enter_long"]:
        e = df.at[i, "close"]; sl = s.at[i, "sl"]; tp = s.at[i, "tp"]
        assert sl < e < tp
        assert np.isclose(tp - e, rr * (e - sl), rtol=1e-6)
    for i in s.index[s["action"] == "enter_short"]:
        e = df.at[i, "close"]; sl = s.at[i, "sl"]; tp = s.at[i, "tp"]
        assert tp < e < sl
        assert np.isclose(e - tp, rr * (sl - e), rtol=1e-6)


def test_no_short_when_disabled():
    df = _synth_h1()
    s = XauObFvgTrend().generate_signals(df, {}, params={"allow_short": False}).signals
    assert (s["action"] == "enter_short").sum() == 0


def test_registered():
    from src.strategies.registry import get_strategies_for_symbol
    reg = get_strategies_for_symbol("XAUUSD")
    assert "ob_fvg_trend" in reg
    assert reg["ob_fvg_trend"]["strategy_cls"] is XauObFvgTrend
