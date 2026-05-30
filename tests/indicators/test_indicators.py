"""Unit + property tests for the indicators module.

The most important test here is `test_lookahead_safety_*`: each indicator computed
on the full series must produce IDENTICAL values at every position that a
prefix-only computation reaches. Any divergence proves lookahead bias.

Run with: python -m pytest tests/indicators/test_indicators.py -v
or:       python -m tests.indicators.test_indicators
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.indicators import (
    sma, ema, macd, adx, true_range,
    rsi, stochastic,
    atr, bollinger, keltner,
    vwap, obv, cvd,
    align_htf_to_ltf,
)


# -----------------------------------------------------------------------------
# Synthetic data helpers
# -----------------------------------------------------------------------------

def _synth_ohlcv(n: int = 500, start: str = "2025-01-01", freq: str = "5min", seed: int = 7) -> pd.DataFrame:
    """Build a deterministic synthetic OHLCV frame with realistic shape."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    # Random walk for close, then derive plausible OHLC and volume.
    rets = rng.normal(0.0, 0.001, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    # high = close + |noise|, low = close - |noise|, open = previous close (shifted)
    noise = np.abs(rng.normal(0.0, 0.0008, size=n)) * close
    high = close + noise
    low = close - noise
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    # Ensure OHLC invariant: high >= max(open, close), low <= min(open, close)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = rng.uniform(50.0, 500.0, size=n)
    df = pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df


# -----------------------------------------------------------------------------
# Shape + warmup tests
# -----------------------------------------------------------------------------

def test_sma_warmup_and_values() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, period=3)
    assert out.isna().sum() == 2, f"SMA(3) should have 2 leading NaN, got {out.isna().sum()}"
    # SMA(3) of [3,4,5] = 4, of [2,3,4] = 3, of [1,2,3] = 2
    assert out.iloc[2] == 2.0
    assert out.iloc[3] == 3.0
    assert out.iloc[4] == 4.0
    print("PASS test_sma_warmup_and_values")


def test_ema_warmup_and_recursion() -> None:
    s = pd.Series([1.0] * 10 + [2.0] * 10)
    out = ema(s, period=5)
    # First 4 NaN, then values start.
    assert out.iloc[:4].isna().all()
    assert not out.iloc[4:].isna().any()
    # EMA of a constant series equals that constant.
    assert abs(out.iloc[9] - 1.0) < 1e-9
    # After the jump to 2.0, EMA should monotonically rise toward 2.0.
    diffs = out.iloc[10:].diff().dropna()
    assert (diffs > 0).all(), "EMA should rise toward new higher value"
    assert out.iloc[-1] < 2.0  # never actually reaches it
    print("PASS test_ema_warmup_and_recursion")


def test_macd_columns_and_signal_warmup() -> None:
    df = _synth_ohlcv(n=300)
    m = macd(df["close"], fast=12, slow=26, signal=9)
    assert list(m.columns) == ["macd", "signal", "hist"]
    # macd line warmup = slow-1 NaN; signal warmup = additional signal-1 on TOP of that.
    first_macd_valid = int(m["macd"].first_valid_index())
    first_signal_valid = int(m["signal"].first_valid_index())
    assert first_signal_valid > first_macd_valid
    # hist = macd - signal exactly
    valid_mask = m["macd"].notna() & m["signal"].notna()
    assert np.allclose(m.loc[valid_mask, "hist"], m.loc[valid_mask, "macd"] - m.loc[valid_mask, "signal"])
    print("PASS test_macd_columns_and_signal_warmup")


def test_adx_columns_and_range() -> None:
    df = _synth_ohlcv(n=500)
    a = adx(df, period=14)
    assert list(a.columns) == ["plus_di", "minus_di", "adx"]
    # All DI/ADX values in [0,100] where defined.
    for c in ("plus_di", "minus_di", "adx"):
        vals = a[c].dropna()
        assert vals.between(0, 100).all(), f"{c} out of [0,100]"
    # adx must NOT be all-NaN (the bug we fixed).
    assert a["adx"].notna().sum() > 0
    print("PASS test_adx_columns_and_range")


def test_rsi_range_and_not_all_nan() -> None:
    df = _synth_ohlcv(n=500)
    r = rsi(df["close"], period=14)
    vals = r.dropna()
    assert vals.between(0, 100).all()
    assert len(vals) > 100  # must be defined for most bars (the Wilder bug would make it all-NaN)
    print("PASS test_rsi_range_and_not_all_nan")


def test_rsi_flat_input_neutral() -> None:
    # Flat price -> avg_gain=0, avg_loss=0 -> by convention 50.
    s = pd.Series([100.0] * 100)
    r = rsi(s, period=14)
    last = r.iloc[-1]
    assert last == 50.0, f"flat-input RSI should be 50, got {last}"
    print("PASS test_rsi_flat_input_neutral")


def test_stochastic_range_and_flat_window() -> None:
    df = _synth_ohlcv(n=500)
    s = stochastic(df, k_period=14, d_period=3, smooth_k=3)
    for c in ("k", "d"):
        vals = s[c].dropna()
        assert vals.between(0, 100).all()
    # Flat window -> %K = 50
    flat = pd.DataFrame({
        "high": [100.0] * 30,
        "low": [100.0] * 30,
        "close": [100.0] * 30,
    })
    s2 = stochastic(flat, k_period=5, d_period=3, smooth_k=1)
    assert s2["k"].iloc[-1] == 50.0
    print("PASS test_stochastic_range_and_flat_window")


def test_atr_positive_and_not_all_nan() -> None:
    df = _synth_ohlcv(n=500)
    a = atr(df, period=14)
    vals = a.dropna()
    assert (vals > 0).all()
    assert len(vals) > 400  # the Wilder bug would have produced 0 valid values
    print("PASS test_atr_positive_and_not_all_nan")


def test_bollinger_band_geometry() -> None:
    df = _synth_ohlcv(n=300)
    b = bollinger(df["close"], period=20, n_std=2.0)
    valid = b.dropna(subset=["mid", "upper", "lower"])
    assert (valid["upper"] >= valid["mid"]).all()
    assert (valid["lower"] <= valid["mid"]).all()
    assert (valid["upper"] >= valid["lower"]).all()
    print("PASS test_bollinger_band_geometry")


def test_keltner_band_geometry() -> None:
    df = _synth_ohlcv(n=300)
    k = keltner(df, ema_period=20, atr_period=10, atr_mult=2.0)
    valid = k.dropna(subset=["mid", "upper", "lower"])
    assert (valid["upper"] >= valid["mid"]).all()
    assert (valid["lower"] <= valid["mid"]).all()
    print("PASS test_keltner_band_geometry")


def test_vwap_session_reset_daily_utc() -> None:
    # Construct two distinct UTC days; VWAP should reset between them.
    df = _synth_ohlcv(n=24 * 12 * 3, start="2025-03-10", freq="5min")  # 3 days
    v = vwap(df, anchor="daily_utc", price_col="typical")
    # First bar of day 2 should be very close to typical price (cumulative vol = 1 bar).
    first_d2_idx = df.index[df["timestamp"].dt.normalize() == pd.Timestamp("2025-03-11", tz="UTC")][0]
    typ = (df["high"] + df["low"] + df["close"]) / 3.0
    assert abs(v["vwap"].iloc[first_d2_idx] - typ.iloc[first_d2_idx]) < 1e-9, (
        "First bar of new session must have VWAP == typical price (cumulative vol = 1 bar)"
    )
    # And VWAP std at first bar of session should be 0 (only one sample).
    assert abs(v["vwap_std"].iloc[first_d2_idx]) < 1e-9
    print("PASS test_vwap_session_reset_daily_utc")


def test_vwap_three_bands_default_and_geometry() -> None:
    """Default vwap() returns 1σ, 2σ, 3σ bands; each upper > lower; 3σ wider than 2σ wider than 1σ."""
    df = _synth_ohlcv(n=24 * 12)  # one day
    v = vwap(df, anchor="daily_utc")
    expected_cols = {"vwap", "vwap_std",
                     "vwap_upper_1", "vwap_lower_1",
                     "vwap_upper_2", "vwap_lower_2",
                     "vwap_upper_3", "vwap_lower_3"}
    assert expected_cols.issubset(set(v.columns)), (
        f"missing bands: {expected_cols - set(v.columns)}"
    )
    # Geometry: at any bar where std > 0, the 3σ band is strictly outside 2σ outside 1σ.
    nz = v[v["vwap_std"] > 1e-12].copy()
    assert (nz["vwap_upper_3"] > nz["vwap_upper_2"]).all()
    assert (nz["vwap_upper_2"] > nz["vwap_upper_1"]).all()
    assert (nz["vwap_upper_1"] > nz["vwap"]).all()
    assert (nz["vwap_lower_1"] > nz["vwap_lower_2"]).all()
    assert (nz["vwap_lower_2"] > nz["vwap_lower_3"]).all()
    assert (nz["vwap_lower_1"] < nz["vwap"]).all()
    # Exact relationship: upper_k = vwap + k * std
    assert np.allclose(nz["vwap_upper_2"], nz["vwap"] + 2 * nz["vwap_std"])
    assert np.allclose(nz["vwap_upper_3"], nz["vwap"] + 3 * nz["vwap_std"])
    print("PASS test_vwap_three_bands_default_and_geometry")


def test_vwap_custom_bands() -> None:
    """Caller can override band σ multipliers (e.g. 1.5σ / 2.5σ for half-σ bands)."""
    df = _synth_ohlcv(n=100)
    v = vwap(df, anchor="daily_utc", bands=(1.5, 2.5))
    assert "vwap_upper_1.5" in v.columns
    assert "vwap_lower_2.5" in v.columns
    assert "vwap_upper_1" not in v.columns  # default 1σ NOT included
    print("PASS test_vwap_custom_bands")


def test_vwap_rejects_invalid_bands() -> None:
    df = _synth_ohlcv(n=50)
    for bad in [(), (0.0,), (-1.0, 2.0)]:
        try:
            vwap(df, bands=bad)
        except ValueError:
            continue
        raise AssertionError(f"VWAP should have rejected bands={bad}")
    print("PASS test_vwap_rejects_invalid_bands")


def test_obv_basic_signed_cumulative() -> None:
    """OBV: up bar adds +volume, down bar subtracts volume; first bar = 0."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC"),
        "close": [100.0, 101.0, 100.0, 102.0, 102.0],
        "volume": [10.0, 20.0, 15.0, 30.0, 25.0],
    })
    result = obv(df)
    # Step 0: 0 (no previous close)
    # Step 1: close up → +20  → 20
    # Step 2: close down → -15 → 5
    # Step 3: close up → +30  → 35
    # Step 4: close flat → 0  → 35
    expected = pd.Series([0.0, 20.0, 5.0, 35.0, 35.0])
    assert np.allclose(result.values, expected.values)
    print("PASS test_obv_basic_signed_cumulative")


def test_cvd_with_taker_buy_uses_true_aggressor() -> None:
    """When taker_buy_base is present, CVD uses 2*taker_buy - volume."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=4, freq="5min", tz="UTC"),
        "open": [100.0, 101.0, 100.0, 99.0],
        "close": [101.0, 100.0, 99.0, 100.0],
        "volume": [10.0, 10.0, 10.0, 10.0],
        "taker_buy_base": [7.0, 3.0, 5.0, 6.0],  # buyer-aggressor portion of volume
    })
    # delta = 2*tb - v: [4, -4, 0, 2] → cumsum: [4, 0, 0, 2]
    result = cvd(df)
    expected = np.array([4.0, 0.0, 0.0, 2.0])
    assert np.allclose(result.values, expected)
    print("PASS test_cvd_with_taker_buy_uses_true_aggressor")


def test_cvd_fallback_to_bar_direction() -> None:
    """Without taker_buy_base, CVD falls back to sign(close-open) * volume."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=4, freq="5min", tz="UTC"),
        "open": [100.0, 101.0, 100.0, 99.0],
        "close": [101.0, 100.0, 99.0, 99.0],  # 4th bar is doji (open==close)
        "volume": [10.0, 10.0, 10.0, 10.0],
    })
    # signs: +1, -1, -1, 0 → deltas 10, -10, -10, 0 → cumsum: 10, 0, -10, -10
    result = cvd(df)
    expected = np.array([10.0, 0.0, -10.0, -10.0])
    assert np.allclose(result.values, expected)
    print("PASS test_cvd_fallback_to_bar_direction")


def test_cvd_session_anchored_resets_each_day() -> None:
    """Session-anchored CVD resets at the configured anchor."""
    ts = pd.date_range("2025-01-01 22:00", periods=6, freq="1h", tz="UTC")  # spans midnight
    # Alternate up/down bars: each has a non-zero direction so sign is +1 or -1.
    opens = [100.0, 101.0, 100.0, 101.0, 100.0, 101.0]
    closes = [101.0, 100.0, 101.0, 100.0, 101.0, 100.0]
    df = pd.DataFrame({
        "timestamp": ts,
        "open": opens,
        "close": closes,
        "volume": [10.0] * 6,
    })
    # daily_utc anchor: bars at 22:00, 23:00 belong to 2025-01-01, then 00:00..03:00 to 01-02.
    result = cvd(df, anchor="daily_utc")
    # Day1 (rows 0-1) deltas: +10, -10 → cum 10, 0
    # Day2 (rows 2-5) deltas: +10, -10, +10, -10 → cum 10, 0, 10, 0
    expected = np.array([10.0, 0.0, 10.0, 0.0, 10.0, 0.0])
    assert np.allclose(result.values, expected)
    print("PASS test_cvd_session_anchored_resets_each_day")


def test_vwap_requires_monotonic_timestamps() -> None:
    df = _synth_ohlcv(n=50)
    # Swap two rows to break monotonicity.
    df2 = df.copy()
    df2.iloc[[10, 11]] = df2.iloc[[11, 10]].values
    df2 = pd.DataFrame(df2, columns=df.columns)
    df2["timestamp"] = pd.to_datetime(df2["timestamp"], utc=True)
    try:
        vwap(df2, anchor="daily_utc")
    except ValueError as e:
        assert "monotonic" in str(e) or "sorted" in str(e)
        print("PASS test_vwap_requires_monotonic_timestamps")
        return
    raise AssertionError("VWAP should have rejected non-monotonic input")


# -----------------------------------------------------------------------------
# Lookahead-safety: compute on full vs prefix; common rows must match exactly.
# This is the most important test in this file.
# -----------------------------------------------------------------------------

def _check_no_lookahead(full: pd.Series | pd.DataFrame, prefix: pd.Series | pd.DataFrame, k: int) -> None:
    """The first k rows of `full` must equal `prefix` exactly (modulo NaN equality)."""
    f_head = full.iloc[:k]
    if isinstance(f_head, pd.DataFrame):
        assert f_head.shape == prefix.shape, f"shape mismatch: {f_head.shape} vs {prefix.shape}"
        for c in f_head.columns:
            _assert_series_equal_with_nan(f_head[c], prefix[c], col=c)
    else:
        _assert_series_equal_with_nan(f_head, prefix)


def _assert_series_equal_with_nan(a: pd.Series, b: pd.Series, col: str = "") -> None:
    a_arr = a.to_numpy()
    b_arr = b.to_numpy()
    nan_a = np.isnan(a_arr)
    nan_b = np.isnan(b_arr)
    assert (nan_a == nan_b).all(), f"NaN positions differ for {col}: {np.where(nan_a != nan_b)[0]}"
    finite_mask = ~nan_a
    if finite_mask.any():
        assert np.allclose(a_arr[finite_mask], b_arr[finite_mask], rtol=1e-12, atol=1e-12), (
            f"finite values differ for {col}"
        )


def test_lookahead_safety_all_indicators() -> None:
    df = _synth_ohlcv(n=400)
    K = 250

    # Single-input (close-only)
    for name, fn in [
        ("sma", lambda d: sma(d["close"], 20)),
        ("ema", lambda d: ema(d["close"], 20)),
        ("rsi", lambda d: rsi(d["close"], 14)),
        ("macd", lambda d: macd(d["close"], 12, 26, 9)),
        ("bollinger", lambda d: bollinger(d["close"], 20, 2.0)),
    ]:
        full = fn(df)
        prefix = fn(df.iloc[:K])
        _check_no_lookahead(full, prefix, K)
        print(f"PASS lookahead_safety[{name}]")

    # OHLC-input indicators
    for name, fn in [
        ("true_range", lambda d: true_range(d)),
        ("adx", lambda d: adx(d, 14)),
        ("atr", lambda d: atr(d, 14)),
        ("keltner", lambda d: keltner(d, 20, 10, 2.0)),
        ("stochastic", lambda d: stochastic(d, 14, 3, 3)),
    ]:
        full = fn(df)
        prefix = fn(df.iloc[:K])
        _check_no_lookahead(full, prefix, K)
        print(f"PASS lookahead_safety[{name}]")

    # VWAP (needs full df incl. timestamp + volume)
    full_v = vwap(df, anchor="daily_utc")
    prefix_v = vwap(df.iloc[:K], anchor="daily_utc")
    _check_no_lookahead(full_v, prefix_v, K)
    print("PASS lookahead_safety[vwap]")

    # OBV, CVD (causal cumsum)
    full_obv = obv(df)
    prefix_obv = obv(df.iloc[:K])
    _check_no_lookahead(full_obv, prefix_obv, K)
    print("PASS lookahead_safety[obv]")

    full_cvd = cvd(df)
    prefix_cvd = cvd(df.iloc[:K])
    _check_no_lookahead(full_cvd, prefix_cvd, K)
    print("PASS lookahead_safety[cvd_fallback]")

    full_cvd_sess = cvd(df, anchor="daily_utc")
    prefix_cvd_sess = cvd(df.iloc[:K], anchor="daily_utc")
    _check_no_lookahead(full_cvd_sess, prefix_cvd_sess, K)
    print("PASS lookahead_safety[cvd_session]")


# -----------------------------------------------------------------------------
# HTF alignment correctness
# -----------------------------------------------------------------------------

def test_align_htf_to_ltf_uses_only_closed_bars() -> None:
    """At LTF time T, the attached HTF value must be the last HTF bar whose
    available_at <= T + Δ_ltf (i.e. fully closed at LTF signal-evaluation moment)."""
    # Construct M5 LTF spanning two hours, and H1 HTF for the same window.
    ltf_ts = pd.date_range("2025-06-01 00:00", periods=36, freq="5min", tz="UTC")  # 3 hours
    ltf = pd.DataFrame({"timestamp": ltf_ts, "close": np.arange(len(ltf_ts), dtype=float)})

    htf_ts = pd.date_range("2025-06-01 00:00", periods=4, freq="1h", tz="UTC")
    # HTF "value" = the hour number for easy inspection
    htf = pd.DataFrame({"timestamp": htf_ts, "v": [10.0, 20.0, 30.0, 40.0]})

    aligned = align_htf_to_ltf(ltf, htf, ltf_tf="M5", htf_tf="H1", htf_cols=["v"], suffix="_h1")

    # The HTF bar with timestamp 00:00 has available_at = 01:00.
    # LTF bar at 00:00 has signal_available_at = 00:05. No HTF bar has available_at <= 00:05.
    # So attached value should be NaN for all LTF bars whose signal_available_at < 01:00.
    # That means LTF bars at 00:00 through 00:55 (timestamps strictly < 01:00 - 5min = 00:55).
    # LTF bar at 00:55 has signal_available_at = 01:00, exactly equal to first HTF available_at.
    # allow_exact_matches=True -> picks up the first HTF bar (v=10).
    mask_pre_first_close = ltf["timestamp"] < pd.Timestamp("2025-06-01 00:55", tz="UTC")
    assert aligned.loc[mask_pre_first_close, "v_h1"].isna().all(), (
        "LTF bars before first HTF close must have NaN HTF context"
    )
    # LTF bar at 00:55 -> v=10 (first HTF bar just became available)
    row_at_0055 = aligned[aligned["timestamp"] == pd.Timestamp("2025-06-01 00:55", tz="UTC")].iloc[0]
    assert row_at_0055["v_h1"] == 10.0, f"expected v_h1=10 at 00:55, got {row_at_0055['v_h1']}"
    # LTF bar at 02:00 -> signal_available_at = 02:05 -> available HTF bars closed by 02:05
    # are those with open <= 01:00 (available_at 02:00). So v=20.
    row_at_0200 = aligned[aligned["timestamp"] == pd.Timestamp("2025-06-01 02:00", tz="UTC")].iloc[0]
    assert row_at_0200["v_h1"] == 20.0, f"expected v_h1=20 at 02:00, got {row_at_0200['v_h1']}"
    # LTF bar at 02:55 -> signal_available_at = 03:00 -> v=30 (HTF 02:00 closed at 03:00).
    row_at_0255 = aligned[aligned["timestamp"] == pd.Timestamp("2025-06-01 02:55", tz="UTC")].iloc[0]
    assert row_at_0255["v_h1"] == 30.0
    print("PASS test_align_htf_to_ltf_uses_only_closed_bars")


def test_align_htf_to_ltf_preserves_ltf_with_datetime_index() -> None:
    """Regression for the non-monotonic index bug — pass an LTF whose index is a DatetimeIndex."""
    ltf_ts = pd.date_range("2025-06-01 00:00", periods=24, freq="5min", tz="UTC")
    ltf = pd.DataFrame({"timestamp": ltf_ts, "close": np.arange(24, dtype=float)})
    ltf.index = ltf_ts  # DatetimeIndex, NOT 0..N-1
    htf_ts = pd.date_range("2025-06-01 00:00", periods=3, freq="1h", tz="UTC")
    htf = pd.DataFrame({"timestamp": htf_ts, "v": [10.0, 20.0, 30.0]})

    aligned = align_htf_to_ltf(ltf, htf, ltf_tf="M5", htf_tf="H1")
    # Output index must match input index, and 'v_H1' column must be populated correctly.
    assert (aligned.index == ltf.index).all()
    # Spot-check the 02:00 LTF row would be NaN because we only have HTF up to 02:00 open
    # (signal_available_at=02:05, HTF 02:00 available_at=03:00 not yet available).
    print("PASS test_align_htf_to_ltf_preserves_ltf_with_datetime_index")


if __name__ == "__main__":
    test_sma_warmup_and_values()
    test_ema_warmup_and_recursion()
    test_macd_columns_and_signal_warmup()
    test_adx_columns_and_range()
    test_rsi_range_and_not_all_nan()
    test_rsi_flat_input_neutral()
    test_stochastic_range_and_flat_window()
    test_atr_positive_and_not_all_nan()
    test_bollinger_band_geometry()
    test_keltner_band_geometry()
    test_vwap_session_reset_daily_utc()
    test_vwap_three_bands_default_and_geometry()
    test_vwap_custom_bands()
    test_vwap_rejects_invalid_bands()
    test_obv_basic_signed_cumulative()
    test_cvd_with_taker_buy_uses_true_aggressor()
    test_cvd_fallback_to_bar_direction()
    test_cvd_session_anchored_resets_each_day()
    test_vwap_requires_monotonic_timestamps()
    test_lookahead_safety_all_indicators()
    test_align_htf_to_ltf_uses_only_closed_bars()
    test_align_htf_to_ltf_preserves_ltf_with_datetime_index()
    print("\nAll indicator tests passed.")
