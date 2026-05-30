"""End-to-end smoke test for the walk-forward optimizer.

Tests that:
1. Every strategy in REGISTRY imports and instantiates.
2. The walk-forward optimizer runs without crashing on a small synthetic
   dataset for at least ONE strategy.
3. Output artifacts are produced (manifest.json, windows.csv, etc).

Real-data optimization runs go via `scripts/optimize_all.py`, NOT this test.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.optimize import run_walkforward, WalkForwardConfig, OptimizerConfig
from src.strategies.registry import REGISTRY


# Skip the heavy test if optuna isn't installed (CI without dev deps).
try:
    import optuna  # noqa: F401
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False


def _synth_btc_m5_and_h1(n_days: int = 200, seed: int = 5):
    """Generate ~n_days of BTC-like M5 + H1 data with a regime change midway."""
    n_m5 = n_days * 24 * 12
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n_m5, freq="5min", tz="UTC")
    # Up trend half, then down half.
    drift = np.concatenate([
        np.full(n_m5 // 2, 0.0003),
        np.full(n_m5 - n_m5 // 2, -0.0003),
    ])
    rets = drift + rng.normal(0.0, 0.001, size=n_m5)
    close = 50_000.0 * np.exp(np.cumsum(rets))
    noise = np.abs(rng.normal(0.0, 0.0008, size=n_m5)) * close
    high = close + noise
    low = close - noise
    open_ = np.empty(n_m5)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = rng.uniform(1.0, 10.0, size=n_m5)
    m5 = pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                       "close": close, "volume": volume})
    h1 = m5.set_index("timestamp").resample("1h", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna().reset_index()
    return m5, h1


def test_all_strategies_in_registry_import_and_instantiate() -> None:
    """REGISTRY must be importable and every entry must instantiate cleanly."""
    assert len(REGISTRY) >= 8, f"expected at least 8 entries, got {len(REGISTRY)}"
    for (symbol, name), entry in REGISTRY.items():
        cls = entry["strategy_cls"]
        strat = cls()
        # Must declare its LTF and required HTFs
        assert isinstance(strat.ltf, str)
        assert isinstance(strat.required_htfs, tuple)
        assert isinstance(entry["param_space"], dict) and entry["param_space"], (
            f"{symbol}/{name}: empty param_space"
        )
    print(f"PASS test_all_strategies_in_registry_import_and_instantiate ({len(REGISTRY)} entries)")


def test_all_btc_strategies_run_on_synth_data() -> None:
    """Each BTC strategy in REGISTRY must produce a valid signals DataFrame."""
    m5, h1 = _synth_btc_m5_and_h1(n_days=60)
    for (symbol, name), entry in REGISTRY.items():
        if symbol != "BTCUSDT":
            continue
        strat = entry["strategy_cls"]()
        htfs = {tf: h1 for tf in strat.required_htfs}
        result = strat.generate_signals(ltf=m5, htfs=htfs)
        assert len(result.signals) == len(m5), f"{name}: signal len mismatch"
        assert set(result.signals.columns) >= {"action", "sl", "tp"}, f"{name}: missing cols"
    print(f"PASS test_all_btc_strategies_run_on_synth_data")


@pytest.mark.skipif(not OPTUNA_OK, reason="optuna not installed")
def test_walkforward_end_to_end_smoke() -> None:
    """Run the optimizer for ONE BTC strategy with a tiny budget; assert artifacts produced."""
    m5, h1 = _synth_btc_m5_and_h1(n_days=200)

    # Pick the simplest BTC strategy
    entry = REGISTRY[("BTCUSDT", "bollinger_squeeze")]
    strat = entry["strategy_cls"]()

    with tempfile.TemporaryDirectory() as tmp:
        wf_cfg = WalkForwardConfig(train_days=60, test_days=20, step_days=20,
                                   min_trades_for_eval=5)
        opt_cfg = OptimizerConfig(n_trials=3, sampler_seed=0, show_progress_bar=False)

        result = run_walkforward(
            strategy=strat, ltf=m5, htfs={tf: h1 for tf in strat.required_htfs},
            symbol="BTCUSDT", param_space=entry["param_space"],
            wf_cfg=wf_cfg, opt_cfg=opt_cfg,
            output_dir=Path(tmp),
        )

    # Artifacts present, manifest hash valid
    assert result["manifest"]["parity_doc_sha256"]
    assert result["manifest"]["n_windows"] > 0, "expected at least one walk-forward window"
    assert isinstance(result["windows"], pd.DataFrame)
    assert isinstance(result["trials"], pd.DataFrame)
    print(f"PASS test_walkforward_end_to_end_smoke "
          f"(n_windows={result['manifest']['n_windows']}, "
          f"n_trials={len(result['trials'])})")


if __name__ == "__main__":
    test_all_strategies_in_registry_import_and_instantiate()
    test_all_btc_strategies_run_on_synth_data()
    if OPTUNA_OK:
        test_walkforward_end_to_end_smoke()
    else:
        print("SKIP test_walkforward_end_to_end_smoke (optuna not installed)")
    print("\nAll optimizer smoke tests passed.")
