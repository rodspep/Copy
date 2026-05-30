"""Tests for XauZoneAnticipator — schema, modes, no-lookahead, gate behavior."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.xau.zone_anticipator import XauZoneAnticipator


def _synth_oscillating(n_per_seg: int = 2500, seed: int = 13) -> dict[str, pd.DataFrame]:
    """Mild oscillating market — gives both demand and supply zone touches and
    enough swing/OB/FVG events for zones to populate."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_seg
    ts = pd.date_range("2025-02-01", periods=n, freq="5min", tz="UTC")
    # Two segments with different directional bias
    drift = np.concatenate([np.full(n_per_seg, 0.0003), np.full(n_per_seg, -0.0003)])
    rets = drift + rng.normal(0.0, 0.0010, size=n)
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
    h1 = idx.resample("1h", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna().reset_index()
    return {"m5": m5, "m15": m15, "h1": h1}


# -----------------------------------------------------------------------------
# Schema + HTF requirement
# -----------------------------------------------------------------------------

def test_schema_valid_both_modes() -> None:
    d = _synth_oscillating()
    strat = XauZoneAnticipator()
    for mode in ("limit", "reaction"):
        sigs = strat.generate_signals(d["m5"], {"M15": d["m15"], "H1": d["h1"]},
                                      params={"mode": mode}).signals
        assert {"action", "sl", "tp"}.issubset(sigs.columns)
        assert len(sigs) == len(d["m5"])
        assert set(sigs["action"].unique()).issubset({"hold", "enter_long", "enter_short", "exit"})


def test_required_htfs_enforced() -> None:
    d = _synth_oscillating(n_per_seg=200)
    strat = XauZoneAnticipator()
    with pytest.raises(ValueError, match="HTF"):
        strat.generate_signals(d["m5"], {"M15": d["m15"]})


def test_unknown_mode_raises() -> None:
    d = _synth_oscillating(n_per_seg=200)
    strat = XauZoneAnticipator()
    with pytest.raises(ValueError, match="mode"):
        strat.generate_signals(d["m5"], {"M15": d["m15"], "H1": d["h1"]},
                               params={"mode": "no_such_mode"})


# -----------------------------------------------------------------------------
# Determinism + no-lookahead
# -----------------------------------------------------------------------------

def test_deterministic() -> None:
    d = _synth_oscillating(n_per_seg=1500)
    strat = XauZoneAnticipator()
    htfs = {"M15": d["m15"], "H1": d["h1"]}
    a = strat.generate_signals(d["m5"], htfs).signals
    b = strat.generate_signals(d["m5"], htfs).signals
    assert (a["action"].to_numpy() == b["action"].to_numpy()).all()


def test_no_lookahead_on_prefix_limit_mode() -> None:
    d = _synth_oscillating(n_per_seg=1800)
    m5, m15, h1 = d["m5"], d["m15"], d["h1"]
    strat = XauZoneAnticipator()
    params = {"mode": "limit"}

    full = strat.generate_signals(m5, {"M15": m15, "H1": h1}, params).signals

    K = 2500
    last_avail = m5["timestamp"].iloc[K - 1] + pd.Timedelta(minutes=5)
    m15_pref = m15[m15["timestamp"] + pd.Timedelta(minutes=15) <= last_avail].copy()
    h1_pref = h1[h1["timestamp"] + pd.Timedelta(hours=1) <= last_avail].copy()
    pref = strat.generate_signals(m5.iloc[:K].copy(), {"M15": m15_pref, "H1": h1_pref},
                                  params).signals

    fh = full.iloc[:K].reset_index(drop=True)
    pr = pref.reset_index(drop=True)
    assert (fh["action"].to_numpy() == pr["action"].to_numpy()).all(), (
        "limit-mode action column diverged → lookahead leak"
    )
    for col in ("sl", "tp"):
        a = fh[col].to_numpy()
        b = pr[col].to_numpy()
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        assert (nan_a == nan_b).all()
        mask = ~nan_a
        if mask.any():
            assert np.allclose(a[mask], b[mask], rtol=1e-10, atol=1e-10), f"{col} diverged"


def test_no_lookahead_on_prefix_reaction_mode() -> None:
    """Reaction mode looks at bars i+1..i+max_wait AFTER a touch at i. Verify
    that the prefix-cutoff still produces matching signals up to bar K-max_wait."""
    d = _synth_oscillating(n_per_seg=1800)
    m5, m15, h1 = d["m5"], d["m15"], d["h1"]
    strat = XauZoneAnticipator()
    params = {"mode": "reaction", "reaction_max_wait": 2}

    full = strat.generate_signals(m5, {"M15": m15, "H1": h1}, params).signals

    K = 2500
    last_avail = m5["timestamp"].iloc[K - 1] + pd.Timedelta(minutes=5)
    m15_pref = m15[m15["timestamp"] + pd.Timedelta(minutes=15) <= last_avail].copy()
    h1_pref = h1[h1["timestamp"] + pd.Timedelta(hours=1) <= last_avail].copy()
    pref = strat.generate_signals(m5.iloc[:K].copy(), {"M15": m15_pref, "H1": h1_pref},
                                  params).signals

    # Entries within bars 0..K-1 of FULL must match prefix, EXCEPT the last
    # `reaction_max_wait` bars (those may be incomplete in prefix since the
    # touch at K-2 / K-3 hasn't seen its post-touch window yet at prefix time).
    edge = params["reaction_max_wait"]
    fh = full.iloc[:K - edge].reset_index(drop=True)
    pr = pref.iloc[:K - edge].reset_index(drop=True)
    assert (fh["action"].to_numpy() == pr["action"].to_numpy()).all(), (
        "reaction-mode action column diverged → lookahead leak"
    )


# -----------------------------------------------------------------------------
# Gate behavior
# -----------------------------------------------------------------------------

def test_gates_off_at_least_as_many_signals() -> None:
    """Turning every gate OFF must produce ≥ as many entries as having gates ON."""
    d = _synth_oscillating(n_per_seg=2000)
    strat = XauZoneAnticipator()
    htfs = {"M15": d["m15"], "H1": d["h1"]}
    base_off = {
        "mode": "limit",
        "require_ma_align": False, "require_htf_smc": False, "session_filter": False,
    }
    n_off = (strat.generate_signals(d["m5"], htfs, base_off).signals["action"] != "hold").sum()
    for gate in ("require_ma_align", "require_htf_smc", "session_filter"):
        p = dict(base_off); p[gate] = True
        n_on = (strat.generate_signals(d["m5"], htfs, p).signals["action"] != "hold").sum()
        assert n_on <= n_off, f"gate {gate}=True produced MORE signals ({n_on} > {n_off})"


def test_session_filter_excludes_off_hours() -> None:
    d = _synth_oscillating(n_per_seg=2000)
    strat = XauZoneAnticipator()
    sigs = strat.generate_signals(d["m5"], {"M15": d["m15"], "H1": d["h1"]}, params={
        "mode": "limit", "session_filter": True,
        "trade_start_hour": 8, "trade_end_hour": 15,
        "require_ma_align": False, "require_htf_smc": False,
    }).signals
    fire = sigs["action"] != "hold"
    if fire.sum() == 0:
        pytest.skip("no signals fired")
    hours = d["m5"]["timestamp"].dt.hour.to_numpy()[fire.to_numpy()]
    assert (hours >= 8).all() and (hours < 15).all()


# -----------------------------------------------------------------------------
# SL/TP placement sanity
# -----------------------------------------------------------------------------

def test_sl_tp_geometry() -> None:
    d = _synth_oscillating(n_per_seg=2500)
    strat = XauZoneAnticipator()
    sigs = strat.generate_signals(d["m5"], {"M15": d["m15"], "H1": d["h1"]}, params={
        "mode": "limit",
        "require_ma_align": False, "require_htf_smc": False, "session_filter": False,
    }).signals
    m5 = d["m5"].reset_index(drop=True)
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
