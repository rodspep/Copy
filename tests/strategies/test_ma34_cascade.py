"""Tests for XauMa34Cascade — schema, MTF alignment, no-lookahead, fixed SL/TP."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.xau.ma34_cascade import XauMa34Cascade


def _synth_trending(n_per_seg: int = 4000, seed: int = 17) -> dict[str, pd.DataFrame]:
    """M5 + M15 + M30 + H1 with two trending segments (up then down)."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_seg
    ts = pd.date_range("2025-02-01", periods=n, freq="5min", tz="UTC")
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
    idx = m5.set_index("timestamp")
    out = {"m5": m5}
    for tf, freq in [("m15", "15min"), ("m30", "30min"), ("h1", "1h")]:
        df = idx.resample(freq, label="left", closed="left").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
        }).dropna().reset_index()
        out[tf] = df
    return out


def test_schema_valid() -> None:
    d = _synth_trending()
    strat = XauMa34Cascade()
    sigs = strat.generate_signals(d["m5"], {"M15": d["m15"], "M30": d["m30"], "H1": d["h1"]}).signals
    assert {"action", "sl", "tp"}.issubset(sigs.columns)
    assert len(sigs) == len(d["m5"])
    assert set(sigs["action"].unique()).issubset({"hold", "enter_long", "enter_short", "exit"})


def test_required_htfs_enforced() -> None:
    d = _synth_trending(n_per_seg=500)
    strat = XauMa34Cascade()
    with pytest.raises(ValueError, match="HTF"):
        strat.generate_signals(d["m5"], {"M15": d["m15"]})


def test_deterministic() -> None:
    d = _synth_trending(n_per_seg=2500)
    strat = XauMa34Cascade()
    htfs = {"M15": d["m15"], "M30": d["m30"], "H1": d["h1"]}
    a = strat.generate_signals(d["m5"], htfs).signals
    b = strat.generate_signals(d["m5"], htfs).signals
    assert (a["action"].to_numpy() == b["action"].to_numpy()).all()


def test_no_lookahead_on_prefix() -> None:
    d = _synth_trending(n_per_seg=3000)
    m5, m15, m30, h1 = d["m5"], d["m15"], d["m30"], d["h1"]
    strat = XauMa34Cascade()
    full = strat.generate_signals(m5, {"M15": m15, "M30": m30, "H1": h1}).signals

    K = 4500
    last_avail = m5["timestamp"].iloc[K - 1] + pd.Timedelta(minutes=5)
    m15_p = m15[m15["timestamp"] + pd.Timedelta(minutes=15) <= last_avail].copy()
    m30_p = m30[m30["timestamp"] + pd.Timedelta(minutes=30) <= last_avail].copy()
    h1_p = h1[h1["timestamp"] + pd.Timedelta(hours=1) <= last_avail].copy()
    pref = strat.generate_signals(m5.iloc[:K].copy(),
                                  {"M15": m15_p, "M30": m30_p, "H1": h1_p}).signals

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


def test_min_tf_agree_threshold_reduces_signals() -> None:
    """Requiring more TFs to agree must produce ≤ signals."""
    d = _synth_trending(n_per_seg=3000)
    strat = XauMa34Cascade()
    htfs = {"M15": d["m15"], "M30": d["m30"], "H1": d["h1"]}
    counts = []
    for k in (1, 2, 3):
        sigs = strat.generate_signals(d["m5"], htfs, params={"min_tf_agree": k}).signals
        counts.append((sigs["action"] != "hold").sum())
    # Monotonically non-increasing
    for i in range(1, len(counts)):
        assert counts[i] <= counts[i-1], f"min_tf_agree={i+1} produced more signals than ={i}"


def test_fixed_sl_tp_distance_matches_atr_mult() -> None:
    """For enters, |entry-sl| should equal sl_atr_mult × ATR(entry_bar)."""
    d = _synth_trending(n_per_seg=3000)
    strat = XauMa34Cascade()
    sigs = strat.generate_signals(d["m5"], {"M15": d["m15"], "M30": d["m30"], "H1": d["h1"]},
                                  params={"sl_atr_mult": 1.0, "tp_atr_mult": 0.5}).signals
    m5 = d["m5"]
    enter_idx = sigs.index[sigs["action"].isin(["enter_long", "enter_short"])]
    if len(enter_idx) == 0:
        pytest.skip("no signals fired")
    from src.indicators import atr
    a = atr(m5, 14).to_numpy()
    close = m5["close"].to_numpy()
    sl = sigs["sl"].to_numpy()
    tp = sigs["tp"].to_numpy()
    for i in enter_idx[:5]:  # sample a few
        if not np.isfinite(a[i]):
            continue
        sl_dist = abs(close[i] - sl[i])
        tp_dist = abs(close[i] - tp[i])
        # Within 1% tolerance for rounding/float
        assert abs(sl_dist - 1.0 * a[i]) < 0.01 * a[i], f"bar {i}: sl dist {sl_dist} != ATR {a[i]}"
        assert abs(tp_dist - 0.5 * a[i]) < 0.01 * a[i], f"bar {i}: tp dist {tp_dist} != 0.5 ATR"


def test_sl_tp_geometry() -> None:
    d = _synth_trending(n_per_seg=3000)
    strat = XauMa34Cascade()
    sigs = strat.generate_signals(d["m5"], {"M15": d["m15"], "M30": d["m30"], "H1": d["h1"]}).signals
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
