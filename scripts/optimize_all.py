"""Run walk-forward optimization for every (symbol, strategy) in the registry.

Reads parquet data from data/btc/<symbol>/<tf>/ and data/xau/<symbol>/<tf>/,
runs walk-forward Optuna, and writes results under results/optimize/.

Usage:
  python -m scripts.optimize_all
  python -m scripts.optimize_all --symbols XAUUSD
  python -m scripts.optimize_all --strategies ema_pullback session_breakout
  python -m scripts.optimize_all --trials 30 --train-days 60 --test-days 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR
from src.data import binance_loader, histdata_loader
from src.optimize import run_walkforward, WalkForwardConfig, OptimizerConfig
from src.strategies.registry import REGISTRY


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("optimize_all")


# Map Binance interval strings to our TF names so the loader path matches the registry's HTF requests.
BINANCE_TF_MAP = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m", "H1": "1h", "H4": "4h"}


def _load_ohlcv(symbol: str, tf: str) -> pd.DataFrame:
    if symbol == "BTCUSDT":
        interval = BINANCE_TF_MAP[tf]
        return binance_loader.load(symbol, interval)
    if symbol == "XAUUSD":
        return histdata_loader.load(symbol, tf)
    raise ValueError(f"unknown symbol {symbol}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward optimize all strategies.")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Restrict to symbols (default: all in registry)")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Restrict to strategy names (default: all in registry)")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per window")
    parser.add_argument("--train-days", type=int, default=90)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--risk-pct", type=float, default=0.005)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--min-trades", type=int, default=20,
                        help="Min IS trades to be a valid candidate (else -inf score)")
    parser.add_argument("--selector", choices=["is_best", "kfold_robust"], default="is_best",
                        help="Param selection mode after Optuna study completes")
    parser.add_argument("--top-k-frac", type=float, default=0.20,
                        help="kfold_robust: top fraction of Optuna trials to re-evaluate on folds")
    parser.add_argument("--n-folds", type=int, default=3,
                        help="kfold_robust: number of IS sub-folds")
    parser.add_argument("--fold-min-trades", type=int, default=10,
                        help="kfold_robust: min trades for a fold to count")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    wf_cfg = WalkForwardConfig(
        train_days=args.train_days, test_days=args.test_days, step_days=args.step_days,
        min_trades_for_eval=args.min_trades,
        initial_equity=args.initial_equity, risk_pct=args.risk_pct, compounding=True,
    )
    opt_cfg = OptimizerConfig(
        n_trials=args.trials, show_progress_bar=False,
        selector=args.selector, top_k_frac=args.top_k_frac,
        n_folds=args.n_folds, fold_min_trades=args.fold_min_trades,
    )

    # Pick the subset of registry to run.
    entries = [(s, n, e) for (s, n), e in REGISTRY.items()
               if (args.symbols is None or s in args.symbols)
               and (args.strategies is None or n in args.strategies)]

    summary_rows = []
    for symbol, name, entry in entries:
        logger.info(f"========== {symbol} / {name} ==========")
        strategy = entry["strategy_cls"]()
        ltf_tf = strategy.ltf
        try:
            ltf = _load_ohlcv(symbol, ltf_tf)
        except Exception as e:
            logger.error(f"failed loading LTF {ltf_tf} for {symbol}: {e}")
            continue
        if ltf.empty:
            logger.warning(f"no LTF data for {symbol} {ltf_tf} — run scripts/download_all.py first")
            continue

        htfs: dict[str, pd.DataFrame] = {}
        for htf in strategy.required_htfs:
            try:
                h = _load_ohlcv(symbol, htf)
                if h.empty:
                    logger.warning(f"empty HTF {htf} for {symbol} — skipping strategy")
                    htfs = {}
                    break
                htfs[htf] = h
            except Exception as e:
                logger.error(f"failed loading HTF {htf} for {symbol}: {e}")
                htfs = {}
                break
        if strategy.required_htfs and not htfs:
            continue

        try:
            result = run_walkforward(
                strategy=strategy, ltf=ltf, htfs=htfs, symbol=symbol,
                param_space=entry["param_space"],
                wf_cfg=wf_cfg, opt_cfg=opt_cfg,
            )
        except Exception as e:
            logger.exception(f"walk-forward failed for {symbol}/{name}: {e}")
            continue

        agg = result.get("agg_stats", {})
        summary_rows.append({
            "symbol": symbol,
            "strategy": name,
            "n_windows": result["manifest"]["n_windows"],
            "oos_n_trades": agg.get("n_trades", 0),
            "oos_winrate": agg.get("winrate", 0.0),
            "oos_pf": agg.get("profit_factor", 0.0),
            "oos_expectancy_R": agg.get("expectancy_R", 0.0),
            "oos_maxdd": agg.get("max_drawdown_pct", 0.0),
            "oos_sharpe": agg.get("sharpe_annualized", 0.0),
            "oos_total_return_pct": agg.get("total_return_pct", 0.0),
            "output_dir": str(result["output_dir"]),
        })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        out_csv = RESULTS_DIR / "optimize" / "summary.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        logger.info(f"Wrote summary to {out_csv}")
        print("\n=== SUMMARY ===")
        print(df.to_string(index=False))
    else:
        logger.warning("No strategies produced results.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
