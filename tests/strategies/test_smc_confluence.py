"""Tests for XauSmcConfluence — schema, lookahead, gate behavior, trigger paths.

Run: python -m pytest tests/strategies/test_smc_confluence.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.xau.smc_confluence import XauSmcConfluence


# -----------------------------------------------------------------------------
# Synthetic data helpers
# -----------------------------------------------------------------------------

def _synth_m5_trending_then_reversing(n_per_seg: int = 3000, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Build M5 + M15 + H1 with one uptrend segment then one downtrend segment.

    Both regimes exposed so long+short paths can be exercised.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_seg
    ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")

    drift = np.concatenate([
        np.full(n_per_seg, 0.0006),
        np.full(n_per_seg, -0.0006),
    ])
    rets = drift + rng.normal(0.0, 0.0009, size=n)
    close = 2000.0 * np.exp(np.cumsum(rets))

    noise = np.abs(rng.normal(0.0, 0.0006, size=n)) * close
    high = close + noise
    low = close - noise
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    vol = rng.uniform(50, 500, size=n)

    m5 = pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    })

    idx = m5.set_index("timestamp")
    m15 = idx.resample("15min", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna().reset_index()
    h1 = idx.resample("1h", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna().reset_index()
    return {"m5": m5, "m15": m15, "h1": h1}


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------

def test_signals_schema_valid() -> None:
    d = _synth_m5_trending_then_reversing()
    strat = XauSmcConfluence()
    sigs = strat.generate_signals(d["m5"], {"M15": d["m15"], "H1": d["h1"]}).signals
    assert {"action", "sl", "tp"}.issubset(sigs.columns)
    assert len(sigs) == len(d["m5"])
    assert set(sigs["action"].unique()).issubset({"hold", "enter_long", "enter_short", "exit"})


def test_required_htfs_enforced() -> None:
    d = _synth_m5_trending_then_reversing(n_per_seg=200)
    strat = XauSmcConfluence()
    with pytest.raises(ValueError, match="HTF"):
        strat.generate_signals(d["m5"], {"M15": d["m15"]})
    with pytest.raises(ValueError, match="HTF"):
        strat.generate_signals(d["m5"], {"H1": d["h1"]})


# -----------------------------------------------------------------------------
# Determinism + no-lookahead
# -----------------------------------------------------------------------------

def test_deterministic_repeated_call() -> None:
    d = _synth_m5_trending_then_reversing(n_per_seg=2000)
    strat = XauSmcConfluence()
    htfs = {"M15": d["m15"], "H1": d["h1"]}
    a = strat.generate_signals(d["m5"], htfs).signals
    b = strat.generate_signals(d["m5"], htfs).signals
    assert (a["action"].to_numpy() == b["action"].to_numpy()).all()
    for col in ("sl", "tp"):
        av, bv = a[col].to_numpy(), b[col].to_numpy()
        nan_a, nan_b = np.isnan(av), np.isnan(bv)
        assert (nan_a == nan_b).all()
        mask = ~nan_a
        if mask.any():
            assert np.array_equal(av[mask], bv[mask])


def test_no_lookahead_on_prefix() -> None:
    """Common-prefix signals must match exactly when running on truncated input."""
    d = _synth_m5_trending_then_reversing(n_per_seg=2500)
    m5, m15, h1 = d["m5"], d["m15"], d["h1"]
    strat = XauSmcConfluence()
    params = {"vol_regime_filter": False}  # vol filter uses long lookback — disable to keep test fast

    full = strat.generate_signals(m5, {"M15": m15, "H1": h1}, params).signals

    K = 3500
    last_avail = m5["timestamp"].iloc[K - 1] + pd.Timedelta(minutes=5)
    m15_pref = m15[m15["timestamp"] + pd.Timedelta(minutes=15) <= last_avail].copy()
    h1_pref = h1[h1["timestamp"] + pd.Timedelta(hours=1) <= last_avail].copy()
    pref = strat.generate_signals(m5.iloc[:K].copy(), {"M15": m15_pref, "H1": h1_pref}, params).signals

    fh = full.iloc[:K].reset_index(drop=True)
    pr = pref.reset_index(drop=True)
    assert (fh["action"].to_numpy() == pr["action"].to_numpy()).all(), "action column diverged: lookahead leak"
    for col in ("sl", "tp"):
        a = fh[col].to_numpy()
        b = pr[col].to_numpy()
        nan_a = np.isnan(a); nan_b = np.isnan(b)
        assert (nan_a == nan_b).all(), f"{col} NaN positions diverged"
        mask = ~nan_a
        if mask.any():
            assert np.allclose(a[mask], b[mask], rtol=1e-10, atol=1e-10), f"{col} values diverged"


# -----------------------------------------------------------------------------
# Trigger selection — each trigger must be reachable
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("trigger", ["sweep", "ob_retest", "fvg_retest"])
def test_trigger_path_runs_without_error(trigger: str) -> None:
    d = _synth_m5_trending_then_reversing(n_per_seg=2000)
    strat = XauSmcConfluence()
    sigs = strat.generate_signals(d["m5"], {"M15": d["m15"], "H1": d["h1"]}, params={
        "entry_trigger": trigger,
        # With all gates on the synthetic data may produce ~0 signals; that's fine,
        # this test only checks the path doesn't raise.
        "require_htf_smc": False,
        "require_ema_align": False,
        "session_filter": False,
        "vol_regime_filter": False,
    }).signals
    assert {"action", "sl", "tp"}.issubset(sigs.columns)


def test_unknown_trigger_raises() -> None:
    d = _synth_m5_trending_then_reversing(n_per_seg=200)
    strat = XauSmcConfluence()
    with pytest.raises(ValueError, match="entry_trigger"):
        strat.generate_signals(d["m5"], {"M15": d["m15"], "H1": d["h1"]}, params={
            "entry_trigger": "no_such_trigger",
        })


# -----------------------------------------------------------------------------
# Gate behavior — turning gates ON must monotonically reduce (or hold) signals
# -----------------------------------------------------------------------------

def test_gates_off_produces_more_signals_than_gates_on() -> None:
    """Removing every gate should produce ≥ as many signals as having gates active."""
    d = _synth_m5_trending_then_reversing(n_per_seg=2500)
    strat = XauSmcConfluence()
    htfs = {"M15": d["m15"], "H1": d["h1"]}

    base_params = {
        "entry_trigger": "sweep",
        "require_htf_smc": False,
        "require_ema_align": False,
        "session_filter": False,
        "vol_regime_filter": False,
    }
    sigs_off = strat.generate_signals(d["m5"], htfs, base_params).signals
    n_off = (sigs_off["action"] != "hold").sum()

    # Turn each gate on individually; signal count must not exceed `n_off`.
    for gate in ("require_htf_smc", "require_ema_align", "session_filter"):
        params = dict(base_params)
        params[gate] = True
        sigs = strat.generate_signals(d["m5"], htfs, params).signals
        n_on = (sigs["action"] != "hold").sum()
        assert n_on <= n_off, f"gate {gate}=True produced MORE signals than OFF ({n_on} > {n_off})"


def test_session_filter_excludes_off_hours() -> None:
    d = _synth_m5_trending_then_reversing(n_per_seg=2500)
    strat = XauSmcConfluence()
    sigs = strat.generate_signals(d["m5"], {"M15": d["m15"], "H1": d["h1"]}, params={
        "entry_trigger": "sweep",
        "require_htf_smc": False,
        "require_ema_align": False,
        "session_filter": True,
        "trade_start_hour": 7,
        "trade_end_hour": 16,
        "vol_regime_filter": False,
    }).signals

    actions = sigs["action"].to_numpy()
    fire_mask = actions != "hold"
    if fire_mask.sum() == 0:
        pytest.skip("no signals fired on synthetic — covered by other tests")
    hours = d["m5"]["timestamp"].dt.hour.to_numpy()
    fire_hours = hours[fire_mask]
    assert (fire_hours >= 7).all() and (fire_hours < 16).all(), (
        f"signals fired outside [7,16): {sorted(set(fire_hours.tolist()))}"
    )


# -----------------------------------------------------------------------------
# SL/TP placement sanity
# -----------------------------------------------------------------------------

def test_sl_below_close_for_long_above_for_short() -> None:
    d = _synth_m5_trending_then_reversing(n_per_seg=3000)
    strat = XauSmcConfluence()
    sigs = strat.generate_signals(d["m5"], {"M15": d["m15"], "H1": d["h1"]}, params={
        "entry_trigger": "sweep",
        "require_htf_smc": False,
        "require_ema_align": False,
        "session_filter": False,
        "vol_regime_filter": False,
    }).signals

    m5 = d["m5"].reset_index(drop=True)
    long_idx = sigs.index[sigs["action"] == "enter_long"]
    short_idx = sigs.index[sigs["action"] == "enter_short"]
    if len(long_idx) == 0 and len(short_idx) == 0:
        pytest.skip("no signals fired")
    for i in long_idx:
        assert sigs.at[i, "sl"] < m5.at[i, "close"], f"long SL not below close at i={i}"
        assert sigs.at[i, "tp"] > m5.at[i, "close"], f"long TP not above close at i={i}"
    for i in short_idx:
        assert sigs.at[i, "sl"] > m5.at[i, "close"], f"short SL not above close at i={i}"
        assert sigs.at[i, "tp"] < m5.at[i, "close"], f"short TP not below close at i={i}"
