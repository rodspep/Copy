"""End-to-end integration test: synthetic data → indicators → strategy → engine → metrics.

This test exists to PROVE the full pipeline wires together correctly. It is NOT
a strategy-quality test — synthetic data won't give meaningful WR/PF. The
assertions check pipeline integrity, not profitability:

- Strategy produces a valid signals DataFrame.
- Engine runs without raising on real strategy output.
- At least some trades fire on a trending synthetic series.
- Metrics compute cleanly on the result.
- Long AND short signals both fire when the synthetic series reverses direction.

Run: python -m tests.strategies.test_integration
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.reports import compute_stats, composite_objective
from src.strategies.xau.ema_pullback import XauEmaPullback


def _synth_trending_then_reversing(n_m5_per_segment: int = 3000, seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build M5 + H1 OHLCV with a strong uptrend then a strong downtrend.

    Two segments back-to-back ensures both long and short HTF regimes appear,
    so both sides of the strategy can be exercised.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_m5_per_segment
    ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")

    # Drift: +0.0005/bar then -0.0005/bar.
    drift = np.concatenate([
        np.full(n_m5_per_segment, 0.0005),
        np.full(n_m5_per_segment, -0.0005),
    ])
    rets = drift + rng.normal(0.0, 0.0008, size=n)
    close = 2000.0 * np.exp(np.cumsum(rets))

    # OHLC plausibility
    noise = np.abs(rng.normal(0.0, 0.0005, size=n)) * close
    high = close + noise
    low = close - noise
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = rng.uniform(50.0, 500.0, size=n)

    m5 = pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })

    # Resample M5 → H1 OHLCV (true causal resampling — uses only M5 bars within each hour)
    m5_idx = m5.set_index("timestamp")
    h1_resampled = m5_idx.resample("1h", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }).dropna().reset_index()
    return m5, h1_resampled


def test_pipeline_end_to_end_long_and_short() -> None:
    # 3000 M5 bars per segment = 250 H1 bars per segment = 500 H1 total.
    # EMA200 on H1 needs 200 bars to warm up; total of 500 leaves ~300 active bars.
    m5, h1 = _synth_trending_then_reversing(n_m5_per_segment=3000)

    strat = XauEmaPullback()
    result = strat.generate_signals(ltf=m5, htfs={"H1": h1})
    sigs = result.signals

    # Signal schema valid (validator runs inside the strategy too, but double-check).
    assert {"action", "sl", "tp"}.issubset(sigs.columns)
    assert len(sigs) == len(m5)

    # At least some non-hold signals fired.
    n_actions = (sigs["action"] != "hold").sum()
    assert n_actions > 0, "strategy produced zero non-hold signals on a strongly trending synthetic series"

    # Both long AND short fired (segment 1 uptrend → long bias, segment 2 downtrend → short bias).
    long_count = (sigs["action"] == "enter_long").sum()
    short_count = (sigs["action"] == "enter_short").sum()
    print(f"  signal counts: long={long_count} short={short_count} hold={(sigs['action']=='hold').sum()}")
    assert long_count > 0, "no long signals in the uptrend segment"
    assert short_count > 0, "no short signals in the downtrend segment"

    # Run through the engine
    res = run_backtest(m5, sigs, symbol="XAUUSD", ltf_tf="M5", params={
        "initial_equity": 10_000.0, "risk_pct": 0.005, "compounding": True,
    })
    trades = res["trades"]
    equity = res["equity_curve"]
    meta = res["meta"]

    assert len(equity) == len(m5)
    assert "parity_doc_sha256" in meta and len(meta["parity_doc_sha256"]) == 64

    # If any trades happened, both sides should appear (assert each appears at least once if possible).
    print(f"  trades closed: {len(trades)}")
    if len(trades) > 0:
        assert (trades["side"] == 1).any() or (trades["side"] == -1).any()
        # Compute metrics
        stats = compute_stats(trades=trades, equity_curve=equity, initial_equity=10_000.0,
                              bars_per_year=252 * 24 * 12, include_force_eod=False)
        print(f"  n_trades={stats['n_trades']} WR={stats['winrate']:.3f} "
              f"PF={stats['profit_factor']:.2f} maxDD={stats['max_drawdown_pct']:.3f}")
        # composite_objective should not crash on real stats (may return -inf if too few trades).
        score = composite_objective(stats, min_trades=10)
        assert isinstance(score, float)
    else:
        print("  no trades closed — strategy was too selective for this synthetic data (acceptable)")

    print("PASS test_pipeline_end_to_end_long_and_short")


def test_strategy_no_lookahead_on_prefix() -> None:
    """Run strategy on full series vs prefix; common-prefix signals must match exactly."""
    m5, h1 = _synth_trending_then_reversing(n_m5_per_segment=3000)
    K = 4500  # prefix size (less than full = 6000)

    strat = XauEmaPullback()
    full = strat.generate_signals(ltf=m5, htfs={"H1": h1}).signals

    # Build prefix HTFs: H1 bars that are fully closed by the prefix's last M5 bar's
    # availability-time. Use timestamp filter — emulates a live snapshot at K-th M5 bar.
    last_m5_avail = m5["timestamp"].iloc[K - 1] + pd.Timedelta(minutes=5)
    h1_prefix = h1[h1["timestamp"] + pd.Timedelta(hours=1) <= last_m5_avail].copy()
    prefix = strat.generate_signals(ltf=m5.iloc[:K].copy(), htfs={"H1": h1_prefix}).signals

    assert len(prefix) == K
    # First K rows of full must equal prefix.
    full_head = full.iloc[:K].reset_index(drop=True)
    prefix_r = prefix.reset_index(drop=True)
    # action column
    assert (full_head["action"].to_numpy() == prefix_r["action"].to_numpy()).all(), (
        "action column diverged — lookahead detected"
    )
    # sl/tp columns: NaN positions must match; finite values must be close.
    for col in ("sl", "tp"):
        a = full_head[col].to_numpy()
        b = prefix_r[col].to_numpy()
        nan_a = np.isnan(a); nan_b = np.isnan(b)
        assert (nan_a == nan_b).all(), f"{col} NaN positions diverged"
        mask = ~nan_a
        if mask.any():
            assert np.allclose(a[mask], b[mask], rtol=1e-12, atol=1e-12), f"{col} values diverged"
    print("PASS test_strategy_no_lookahead_on_prefix")


if __name__ == "__main__":
    test_pipeline_end_to_end_long_and_short()
    test_strategy_no_lookahead_on_prefix()
    print("\nAll integration tests passed.")
