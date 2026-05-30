"""Walk-forward optimization with Optuna TPE.

Pipeline per (strategy × symbol):
  1. Split LTF data into rolling (train, test) windows. Default 90d train / 30d test,
     stepping by 30d.
  2. For each train window:
       a. Run Optuna TPE search over the strategy's `param_space` for N trials.
       b. Score each trial with `composite_objective(stats)`.
       c. Persist best params and best in-sample score.
  3. With each window's best params, re-run on the corresponding test window
     (OOS) and record OOS stats.
  4. Aggregate OOS stats across all test windows → "true" performance.

Persists results to `results/optimize/<run_id>/`:
  - `trials.csv`            — every trial (window, params, IS score, IS stats).
  - `windows.csv`           — per-window summary (best IS params, OOS stats).
  - `oos_trades.parquet`    — concatenated trades from all OOS windows.
  - `oos_equity.parquet`    — concatenated equity from all OOS windows.
  - `manifest.json`         — run metadata incl. parity-doc hash, library versions.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# Optuna is optional at import-time so unit tests don't require it.
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.backtest import run_backtest
from src.config import SYMBOLS, RESULTS_DIR, ROOT
from src.reports import compute_stats, composite_objective
from src.strategies.base import Strategy


logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Walk-forward window configuration."""
    train_days: int = 90
    test_days: int = 30
    step_days: int = 30
    min_trades_for_eval: int = 20
    initial_equity: float = 10_000.0
    risk_pct: float = 0.005
    compounding: bool = True


@dataclass
class OptimizerConfig:
    """Optuna optimizer configuration."""
    n_trials: int = 50
    timeout_per_window_sec: float | None = None  # None = no timeout
    sampler_seed: int = 42
    show_progress_bar: bool = False
    bars_per_year: float = 252 * 24 * 12  # used by stats annualization

    # ---- Robust param selection ----
    # 'is_best'      : Optuna's argmax over IS (default, prone to IS overfit)
    # 'kfold_robust' : take top-K trials by IS score, re-score on `n_folds` IS sub-folds,
    #                  pick the candidate with highest worst-fold score.
    selector: str = "is_best"
    top_k_frac: float = 0.20        # for kfold_robust: top fraction of trials to re-evaluate
    n_folds: int = 3                # for kfold_robust: number of IS sub-folds
    fold_min_trades: int = 10       # min trades inside a fold to count its score (else -inf)


# -----------------------------------------------------------------------------
# Window construction
# -----------------------------------------------------------------------------

def make_walk_forward_windows(
    ltf: pd.DataFrame,
    cfg: WalkForwardConfig,
) -> list[dict]:
    """Compute (train_start, train_end, test_start, test_end) windows over the LTF span.

    Returns list of dicts with keys 'train_start','train_end','test_start','test_end'
    as tz-aware UTC timestamps.
    """
    if ltf.empty:
        return []
    if ltf["timestamp"].dt.tz is None or str(ltf["timestamp"].dt.tz) != "UTC":
        raise ValueError("ltf must have UTC tz-aware timestamps")

    first_ts = ltf["timestamp"].iloc[0]
    last_ts = ltf["timestamp"].iloc[-1]
    train_td = pd.Timedelta(days=cfg.train_days)
    test_td = pd.Timedelta(days=cfg.test_days)
    step_td = pd.Timedelta(days=cfg.step_days)

    windows: list[dict] = []
    cur_train_start = first_ts
    while True:
        train_end = cur_train_start + train_td
        test_end = train_end + test_td
        if test_end > last_ts:
            break
        windows.append({
            "train_start": cur_train_start,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": test_end,
        })
        cur_train_start = cur_train_start + step_td
    return windows


def _slice(ltf: pd.DataFrame, start, end) -> pd.DataFrame:
    """Inclusive-start, exclusive-end slice."""
    return ltf[(ltf["timestamp"] >= start) & (ltf["timestamp"] < end)].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Strategy-param sampling helper
# -----------------------------------------------------------------------------

ParamSpace = dict[str, dict[str, Any]]
"""Schema: {param_name: {'type': 'int'|'float'|'categorical', 'low': ..., 'high': ..., 'step': ..., 'choices': [...]}}"""


def sample_params(trial: "optuna.Trial", space: ParamSpace) -> dict[str, Any]:
    """Sample one param dict from Optuna trial given the param_space schema."""
    out: dict[str, Any] = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "int":
            step = int(spec.get("step", 1))
            out[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=step)
        elif t == "float":
            log = bool(spec.get("log", False))
            step = spec.get("step", None)
            if step is not None:
                out[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), step=float(step))
            else:
                out[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=log)
        elif t == "categorical":
            out[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown param type {t!r} for {name}")
    return out


# -----------------------------------------------------------------------------
# Robust param selectors
# -----------------------------------------------------------------------------

def _score_params_on_slice(
    strategy: Strategy,
    ltf_slice: pd.DataFrame,
    htfs: dict[str, pd.DataFrame],
    params: dict[str, Any],
    symbol: str,
    wf_cfg: WalkForwardConfig,
    opt_cfg: OptimizerConfig,
    min_trades: int,
    objective_kwargs: dict,
) -> tuple[float, dict]:
    """Evaluate (params) on a single LTF slice. Returns (score, stats)."""
    if ltf_slice.empty:
        return -1e9, {}
    try:
        sigs = strategy.generate_signals(ltf=ltf_slice, htfs=htfs, params=params).signals
        bt = run_backtest(
            ltf_slice, sigs, symbol=symbol, ltf_tf=strategy.ltf,
            params={
                "initial_equity": wf_cfg.initial_equity,
                "risk_pct": wf_cfg.risk_pct,
                "compounding": wf_cfg.compounding,
            },
        )
        stats = compute_stats(
            trades=bt["trades"], equity_curve=bt["equity_curve"],
            initial_equity=wf_cfg.initial_equity,
            bars_per_year=opt_cfg.bars_per_year, include_force_eod=False,
        )
        kwargs = {**objective_kwargs, "min_trades": min_trades}
        score = composite_objective(stats, **kwargs)
        return float(score), stats
    except Exception as exc:
        logger.warning(f"_score_params_on_slice failed: {exc}")
        return -1e9, {}


def select_robust_params(
    study: "optuna.Study",
    strategy: Strategy,
    train_ltf: pd.DataFrame,
    train_htfs: dict[str, pd.DataFrame],
    symbol: str,
    wf_cfg: WalkForwardConfig,
    opt_cfg: OptimizerConfig,
    objective_kwargs: dict,
) -> tuple[dict, float, dict]:
    """K-fold robust selection: pick params whose WORST sub-fold score is highest.

    Procedure:
      1. Rank all completed trials by Optuna IS score descending.
      2. Keep the top `top_k_frac` (at least 5).
      3. Split `train_ltf` into `n_folds` equal sub-folds (no shuffling — preserves
         time order, like a k-fold CV with contiguous folds).
      4. For each top-K candidate, re-evaluate on each sub-fold with the same
         strategy.
      5. Robust score = MIN across folds. Tie-break by mean.
      6. Return the candidate's params with the highest robust score.

    Returns (params, robust_score, diagnostics). Diagnostics include per-fold
    scores for the chosen params and the top-K candidates' robust scores.
    """
    trials_ranked = sorted(
        [t for t in study.trials if t.value is not None and np.isfinite(t.value)],
        key=lambda t: t.value, reverse=True,
    )
    if not trials_ranked:
        # No finite trial — fall back to Optuna's best (may be -inf).
        return study.best_params, study.best_value, {"reason": "no_finite_trials"}

    k = max(5, int(len(trials_ranked) * float(opt_cfg.top_k_frac)))
    candidates = trials_ranked[:min(k, len(trials_ranked))]

    n_folds = max(2, int(opt_cfg.n_folds))
    n = len(train_ltf)
    fold_size = n // n_folds
    fold_slices = []
    for f in range(n_folds):
        a = f * fold_size
        b = (f + 1) * fold_size if f < n_folds - 1 else n
        fold_slices.append((a, b))

    best = None
    best_min = -np.inf
    best_mean = -np.inf
    cand_diag = []
    for cand in candidates:
        scores = []
        for (a, b) in fold_slices:
            fold_ltf = train_ltf.iloc[a:b].reset_index(drop=True)
            # HTFs are already pre-sliced to the train window's range with warmup;
            # reuse as-is (strategy is no-lookahead so it'll only use bars closing
            # at or before each LTF bar).
            sc, _ = _score_params_on_slice(
                strategy, fold_ltf, train_htfs, cand.params,
                symbol, wf_cfg, opt_cfg,
                min_trades=opt_cfg.fold_min_trades,
                objective_kwargs=objective_kwargs,
            )
            scores.append(sc)
        s_min = float(np.min(scores))
        s_mean = float(np.mean(scores))
        cand_diag.append({"params": cand.params, "is_score": cand.value,
                          "fold_scores": scores, "min": s_min, "mean": s_mean})
        # Pick: higher min wins; tie-break by mean.
        if (s_min > best_min) or (s_min == best_min and s_mean > best_mean):
            best = cand
            best_min = s_min
            best_mean = s_mean

    if best is None:
        return study.best_params, study.best_value, {"reason": "all_candidates_failed"}
    return best.params, best_min, {"top_k_candidates": cand_diag, "chosen_mean": best_mean}


# -----------------------------------------------------------------------------
# Main entry: run walk-forward for one strategy on one symbol
# -----------------------------------------------------------------------------

def run_walkforward(
    strategy: Strategy,
    ltf: pd.DataFrame,
    htfs: dict[str, pd.DataFrame],
    symbol: str,
    param_space: ParamSpace,
    wf_cfg: WalkForwardConfig | None = None,
    opt_cfg: OptimizerConfig | None = None,
    output_dir: Path | None = None,
    objective_kwargs: dict | None = None,
) -> dict:
    """Run walk-forward Optuna optimization.

    Args:
        strategy       — Strategy instance (defines required_htfs and ltf).
        ltf            — full LTF OHLCV; will be sliced into windows.
        htfs           — full HTF dict; will be sliced per window.
        symbol         — key into SYMBOLS dict.
        param_space    — see ParamSpace schema.
        wf_cfg         — walk-forward config (defaults if None).
        opt_cfg        — Optuna config (defaults if None).
        output_dir     — where to persist results. Defaults to results/optimize/<run_id>/.
        objective_kwargs — passed to composite_objective(); e.g. min_trades.

    Returns dict with 'windows', 'trials', 'oos_trades', 'oos_equity', 'manifest'.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna is required for walk-forward optimization. Install with: pip install optuna")
    if symbol not in SYMBOLS:
        raise ValueError(f"unknown symbol {symbol}")

    wf_cfg = wf_cfg or WalkForwardConfig()
    opt_cfg = opt_cfg or OptimizerConfig()
    objective_kwargs = objective_kwargs or {}

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    strat_name = type(strategy).__name__
    if output_dir is None:
        output_dir = RESULTS_DIR / "optimize" / f"{strat_name}_{symbol}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = make_walk_forward_windows(ltf, wf_cfg)
    if not windows:
        raise ValueError(
            f"No walk-forward windows fit in the data: "
            f"have {(ltf['timestamp'].iloc[-1] - ltf['timestamp'].iloc[0]).days} days, "
            f"need {wf_cfg.train_days + wf_cfg.test_days} for at least one window."
        )
    logger.info(f"{strat_name} on {symbol}: {len(windows)} walk-forward windows")

    # Quiet down Optuna's per-trial log
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    trial_rows: list[dict] = []
    window_rows: list[dict] = []
    oos_trades_all: list[pd.DataFrame] = []
    oos_equity_all: list[pd.DataFrame] = []

    for wi, w in enumerate(windows):
        train_ltf = _slice(ltf, w["train_start"], w["train_end"])
        test_ltf = _slice(ltf, w["test_start"], w["test_end"])
        train_htfs = {tf: _slice(h, w["train_start"] - pd.Timedelta(days=30), w["train_end"]) for tf, h in htfs.items()}
        # For test we extend HTF backwards too (need warmup), and forwards to test_end.
        test_htfs = {tf: _slice(h, w["test_start"] - pd.Timedelta(days=30), w["test_end"]) for tf, h in htfs.items()}

        if train_ltf.empty or test_ltf.empty:
            logger.warning(f"window {wi}: empty train/test slice, skipping")
            continue

        def _objective(trial: "optuna.Trial") -> float:
            params = sample_params(trial, param_space)
            try:
                result = strategy.generate_signals(ltf=train_ltf, htfs=train_htfs, params=params)
                sigs = result.signals
                bt = run_backtest(
                    train_ltf, sigs, symbol=symbol, ltf_tf=strategy.ltf,
                    params={
                        "initial_equity": wf_cfg.initial_equity,
                        "risk_pct": wf_cfg.risk_pct,
                        "compounding": wf_cfg.compounding,
                    },
                )
                stats = compute_stats(
                    trades=bt["trades"], equity_curve=bt["equity_curve"],
                    initial_equity=wf_cfg.initial_equity,
                    bars_per_year=opt_cfg.bars_per_year, include_force_eod=False,
                )
                score = composite_objective(stats, **{"min_trades": wf_cfg.min_trades_for_eval, **objective_kwargs})
                trial_rows.append({
                    "window": wi, **{f"p_{k}": v for k, v in params.items()},
                    "is_score": score,
                    "is_n_trades": stats["n_trades"],
                    "is_winrate": stats["winrate"],
                    "is_pf": stats["profit_factor"],
                    "is_maxdd": stats["max_drawdown_pct"],
                })
                # Optuna maximizes our objective (we negate -inf to make finite)
                return score if np.isfinite(score) else -1e9
            except Exception as e:
                logger.warning(f"trial failed in window {wi}: {e}")
                return -1e9

        sampler = optuna.samplers.TPESampler(seed=opt_cfg.sampler_seed + wi)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            _objective,
            n_trials=opt_cfg.n_trials,
            timeout=opt_cfg.timeout_per_window_sec,
            show_progress_bar=opt_cfg.show_progress_bar,
        )

        # ---- Robust param selection (default 'is_best' = original behavior) ----
        selector = str(opt_cfg.selector).lower()
        if selector == "is_best":
            best_params = study.best_params
            best_score = study.best_value
        elif selector == "kfold_robust":
            best_params, best_score, _diag = select_robust_params(
                study, strategy, train_ltf, train_htfs, symbol,
                wf_cfg, opt_cfg, objective_kwargs,
            )
        else:
            raise ValueError(f"unknown selector {selector!r}")
        logger.info(f"window {wi}: selector={selector} best score={best_score:.4f}, params={best_params}")

        # OOS: re-run on test window with best params.
        oos_result = strategy.generate_signals(ltf=test_ltf, htfs=test_htfs, params=best_params)
        oos_bt = run_backtest(
            test_ltf, oos_result.signals, symbol=symbol, ltf_tf=strategy.ltf,
            params={
                "initial_equity": wf_cfg.initial_equity,
                "risk_pct": wf_cfg.risk_pct,
                "compounding": wf_cfg.compounding,
            },
        )
        oos_stats = compute_stats(
            trades=oos_bt["trades"], equity_curve=oos_bt["equity_curve"],
            initial_equity=wf_cfg.initial_equity,
            bars_per_year=opt_cfg.bars_per_year, include_force_eod=False,
        )
        oos_trades_all.append(oos_bt["trades"].assign(window=wi))
        oos_equity_all.append(oos_bt["equity_curve"].assign(window=wi))

        window_rows.append({
            "window": wi,
            "train_start": w["train_start"], "train_end": w["train_end"],
            "test_start": w["test_start"], "test_end": w["test_end"],
            "is_best_score": best_score,
            **{f"best_{k}": v for k, v in best_params.items()},
            "oos_n_trades": oos_stats["n_trades"],
            "oos_winrate": oos_stats["winrate"],
            "oos_pf": oos_stats["profit_factor"],
            "oos_maxdd": oos_stats["max_drawdown_pct"],
            "oos_expectancy_R": oos_stats["expectancy_R"],
            "oos_sharpe": oos_stats["sharpe_annualized"],
            "oos_total_return_pct": oos_stats["total_return_pct"],
        })

    trials_df = pd.DataFrame(trial_rows)
    windows_df = pd.DataFrame(window_rows)
    oos_trades = pd.concat(oos_trades_all, ignore_index=True) if oos_trades_all else pd.DataFrame()
    oos_equity = pd.concat(oos_equity_all, ignore_index=True) if oos_equity_all else pd.DataFrame()

    # Aggregate OOS performance across all windows.
    if not oos_trades.empty:
        agg_stats = compute_stats(
            trades=oos_trades, equity_curve=oos_equity[["timestamp", "equity"]]
                  if "timestamp" in oos_equity.columns else pd.DataFrame({"timestamp": [], "equity": []}),
            initial_equity=wf_cfg.initial_equity,
            bars_per_year=opt_cfg.bars_per_year, include_force_eod=False,
        )
    else:
        agg_stats = {}

    # Persist
    if not trials_df.empty:
        trials_df.to_csv(output_dir / "trials.csv", index=False)
    if not windows_df.empty:
        windows_df.to_csv(output_dir / "windows.csv", index=False)
    if not oos_trades.empty:
        oos_trades.to_parquet(output_dir / "oos_trades.parquet", index=False)
    if not oos_equity.empty:
        oos_equity.to_parquet(output_dir / "oos_equity.parquet", index=False)

    parity_hash = _file_sha256(ROOT / "docs" / "decisions" / "backtest_live_parity.md")
    manifest = {
        "run_id": run_id,
        "strategy": strat_name,
        "symbol": symbol,
        "parity_doc_sha256": parity_hash,
        "n_windows": len(windows),
        "wf_cfg": wf_cfg.__dict__,
        "opt_cfg": opt_cfg.__dict__,
        "agg_oos_stats": agg_stats,
        "started_at": run_id,
        "param_space": param_space,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, default=_json_default, indent=2)

    return {
        "windows": windows_df,
        "trials": trials_df,
        "oos_trades": oos_trades,
        "oos_equity": oos_equity,
        "agg_stats": agg_stats,
        "output_dir": output_dir,
        "manifest": manifest,
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)
