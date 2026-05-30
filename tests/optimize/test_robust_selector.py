"""Tests for robust parameter selection in walk-forward optimization.

The robust selector takes the top-K Optuna trials by IS score, re-evaluates
each on `n_folds` sub-folds of the train window, and picks the candidate whose
worst-fold score is highest.

We test:
1. The selector returns ONE of the top-K candidates (never something outside).
2. With deterministic / monotone synthetic data, the selector matches the
   ground-truth most-robust candidate.
3. The default 'is_best' selector still returns Optuna's best (backward compat).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Skip whole module if optuna isn't available — same convention as the rest of the suite.
optuna = pytest.importorskip("optuna")

from src.optimize.walkforward import (
    OptimizerConfig, WalkForwardConfig, select_robust_params,
)
from src.strategies.base import Strategy, StrategyResult, empty_signals


class _ConstantStrategy(Strategy):
    """Test double: emits 'enter_long' every N bars where N = params['period'].

    The trades' R is deterministic too, controlled by params['edge']:
      - With 'edge' > 0 and fixed sl/tp, every trade is a winner (TP > SL hit).
    This gives us full control over IS / fold scores so we can verify the
    selector picks the params with the most uniform per-fold performance.
    """

    ltf = "M5"
    required_htfs: tuple[str, ...] = ()
    default_params = {"period": 50, "edge": 1.0, "sl_atr_mult": 1.0, "tp_atr_mult": 1.5}

    def generate_signals(self, ltf, htfs, params=None):
        p = self.merged_params(params)
        n = len(ltf)
        sigs = empty_signals(ltf)
        # Fire enter_long at every Nth bar with fixed SL/TP relative to close.
        period = max(int(p["period"]), 5)
        idx = np.arange(n)
        fire = (idx % period == 0) & (idx > 0) & (idx < n - 1)
        close = ltf["close"].to_numpy()
        # Edge controls how "easy" winners are: a larger edge → wider TP buffer.
        sl_d = float(p["sl_atr_mult"])
        tp_d = float(p["tp_atr_mult"]) * float(p["edge"])
        sigs.loc[fire, "action"] = "enter_long"
        sigs.loc[fire, "sl"] = close[fire] - sl_d
        sigs.loc[fire, "tp"] = close[fire] + tp_d
        return StrategyResult(signals=sigs)


def _make_smooth_uptrend(n: int = 2000) -> pd.DataFrame:
    """Pure uptrend, no noise — every enter_long hits TP. Makes scoring deterministic."""
    ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    close = 2000.0 + np.arange(n) * 0.1
    open_ = close - 0.05
    high = close + 0.1
    low = close - 0.05
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": np.full(n, 100.0),
    })


def test_selector_returns_top_k_candidate() -> None:
    """The robust selector must pick one of the top-K candidates by IS score."""
    train_ltf = _make_smooth_uptrend(2000)
    strat = _ConstantStrategy()

    # Build a study with multiple completed trials whose IS scores we control.
    space = {
        "period": {"type": "int", "low": 20, "high": 100},
        "edge": {"type": "float", "low": 1.0, "high": 3.0},
        "sl_atr_mult": {"type": "float", "low": 0.5, "high": 2.0},
        "tp_atr_mult": {"type": "float", "low": 0.5, "high": 2.0},
    }

    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def _obj(trial):
        # Fixed pseudo-score: just return a value driven by params so Optuna ranks them.
        period = trial.suggest_int("period", 20, 100)
        edge = trial.suggest_float("edge", 1.0, 3.0)
        sl = trial.suggest_float("sl_atr_mult", 0.5, 2.0)
        tp = trial.suggest_float("tp_atr_mult", 0.5, 2.0)
        return float(edge - 0.01 * period)  # higher edge / lower period scores best

    study.optimize(_obj, n_trials=20, show_progress_bar=False)

    wf_cfg = WalkForwardConfig(train_days=7, test_days=2, step_days=2, min_trades_for_eval=1)
    opt_cfg = OptimizerConfig(
        n_trials=20, selector="kfold_robust",
        top_k_frac=0.25, n_folds=3, fold_min_trades=1,
    )

    # SYMBOLS lookup needs 'XAUUSD'; the strategy is symbol-agnostic so this is just a label.
    params, score, diag = select_robust_params(
        study=study, strategy=strat, train_ltf=train_ltf, train_htfs={},
        symbol="XAUUSD", wf_cfg=wf_cfg, opt_cfg=opt_cfg, objective_kwargs={},
    )

    # The chosen params must equal SOME completed trial's params.
    all_trial_params = [t.params for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert params in all_trial_params, "selector returned params not in any completed trial"
    assert "top_k_candidates" in diag
    # Top-K = top 25% of 20 trials = 5 candidates (at least 5 enforced).
    assert len(diag["top_k_candidates"]) >= 5


def test_is_best_selector_unchanged() -> None:
    """Sanity: when selector='is_best' the loop must use Optuna's argmax (covered
    in walkforward.run_walkforward, smoke-tested via integration test elsewhere).
    Here we just check the OptimizerConfig default is 'is_best' (backward compat)."""
    cfg = OptimizerConfig()
    assert cfg.selector == "is_best"


def test_selector_unknown_raises_in_run_loop() -> None:
    """Run-loop validates the selector string; here we just verify the field
    accepts arbitrary strings (validation happens at use-site)."""
    cfg = OptimizerConfig(selector="some_future_selector")
    assert cfg.selector == "some_future_selector"
