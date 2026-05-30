"""Replay OOS phase of a completed walk-forward run with signal modifications.

Skips the slow Optuna search entirely — loads `windows.csv` (best params per
window from a prior run), regenerates OOS signals, applies optional wrappers
(limit-entry, partial-TP), runs backtest, and aggregates stats.

~5 min replay vs 15-20 min full re-optimize. Good for sweeping execution-layer
configs like limit_offset_atr × expire_bars without re-searching strategy params.

Usage:
  python -m scripts.replay_oos \
      --windows-csv results/optimize/XauSmcConfluence_XAUUSD_20260527T074433/windows.csv \
      --symbol XAUUSD --strategy smc_confluence \
      --wrapper limit_entry --limit-offset-atr 0.3 --expire-bars 5

  python -m scripts.replay_oos --windows-csv <path> --symbol XAUUSD \
      --strategy smc_confluence --wrapper partial_tp --tp1-frac 0.5

  # Grid sweep — runs all combos and prints a comparison table
  python -m scripts.replay_oos --windows-csv <path> --symbol XAUUSD \
      --strategy smc_confluence \
      --sweep-limit-offset 0.1 0.2 0.3 0.5 \
      --sweep-expire 3 5 8
"""
from __future__ import annotations

import argparse
import ast
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.data import binance_loader, histdata_loader
from src.reports import compute_stats, composite_objective
from src.strategies.registry import REGISTRY
from src.strategies.signal_wrappers import limit_entry_wrapper, partial_tp_wrapper


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("replay")

BINANCE_TF_MAP = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m", "H1": "1h", "H4": "4h"}


def _load_ohlcv(symbol: str, tf: str) -> pd.DataFrame:
    if symbol == "BTCUSDT":
        return binance_loader.load(symbol, BINANCE_TF_MAP[tf])
    if symbol == "XAUUSD":
        return histdata_loader.load(symbol, tf)
    raise ValueError(f"unknown symbol {symbol}")


def _slice(ltf: pd.DataFrame, start, end) -> pd.DataFrame:
    return ltf[(ltf["timestamp"] >= start) & (ltf["timestamp"] < end)].reset_index(drop=True)


def _parse_param(v):
    """Convert a CSV cell value to its native Python type.

    windows.csv stores best_* params as strings; we need ints/floats/bools/strings.
    """
    if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v
    if isinstance(v, str):
        sv = v.strip()
        if sv.lower() in ("true", "false"):
            return sv.lower() == "true"
        try:
            return ast.literal_eval(sv)
        except (ValueError, SyntaxError):
            return sv
    return v


def replay_one_window(
    strategy,
    ltf: pd.DataFrame,
    htfs: dict[str, pd.DataFrame],
    window_row: pd.Series,
    symbol: str,
    wrapper: str | None,
    wrapper_kwargs: dict,
    initial_equity: float = 10_000.0,
    risk_pct: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Regenerate OOS signals using best_* params + optional wrapper. Return (trades, equity, stats)."""
    test_start = pd.Timestamp(window_row["test_start"], tz="UTC")
    test_end = pd.Timestamp(window_row["test_end"], tz="UTC")
    test_ltf = _slice(ltf, test_start, test_end)
    # HTF needs warmup window. Match the optimize code's 30-day backstep.
    test_htfs = {tf: _slice(h, test_start - pd.Timedelta(days=30), test_end) for tf, h in htfs.items()}
    if test_ltf.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    # Recover params from windows.csv columns prefixed 'best_'
    params = {}
    for col in window_row.index:
        if col.startswith("best_") and col != "best_score":
            name = col[len("best_"):]
            val = window_row[col]
            if pd.isna(val):
                continue
            params[name] = _parse_param(val)

    sigs = strategy.generate_signals(ltf=test_ltf, htfs=test_htfs, params=params).signals

    # Apply wrapper
    if wrapper == "limit_entry":
        sigs = limit_entry_wrapper(test_ltf, sigs, **wrapper_kwargs)
    elif wrapper == "partial_tp":
        sigs = partial_tp_wrapper(test_ltf, sigs, **wrapper_kwargs)
    elif wrapper is None or wrapper == "none":
        pass
    else:
        raise ValueError(f"unknown wrapper {wrapper!r}")

    bt = run_backtest(
        test_ltf, sigs, symbol=symbol, ltf_tf=strategy.ltf,
        params={"initial_equity": initial_equity, "risk_pct": risk_pct, "compounding": True},
    )
    stats = compute_stats(
        trades=bt["trades"], equity_curve=bt["equity_curve"],
        initial_equity=initial_equity, bars_per_year=252 * 24 * 12, include_force_eod=False,
    )
    return bt["trades"], bt["equity_curve"], stats


def aggregate(per_window_stats: list[dict], per_window_trades: list[pd.DataFrame],
              per_window_equity: list[pd.DataFrame], initial_equity: float = 10_000.0) -> dict:
    """Aggregate OOS across windows the same way walkforward.run_walkforward does."""
    if not per_window_trades:
        return {}
    all_trades = pd.concat(per_window_trades, ignore_index=True) if any(len(t) > 0 for t in per_window_trades) else pd.DataFrame()
    all_equity = pd.concat(per_window_equity, ignore_index=True) if any(len(e) > 0 for e in per_window_equity) else pd.DataFrame()
    if all_trades.empty:
        return {"n_trades": 0, "winrate": 0.0, "profit_factor": 0.0, "expectancy_R": 0.0,
                "max_drawdown_pct": 0.0, "total_return_pct": 0.0, "sharpe_annualized": 0.0,
                "n_pos_windows": 0, "n_windows": len(per_window_stats)}
    agg = compute_stats(
        trades=all_trades,
        equity_curve=all_equity[["timestamp", "equity"]] if "timestamp" in all_equity.columns else pd.DataFrame(),
        initial_equity=initial_equity, bars_per_year=252 * 24 * 12, include_force_eod=False,
    )
    agg["n_pos_windows"] = sum(1 for s in per_window_stats if s.get("expectancy_R", 0) > 0)
    agg["n_windows"] = len(per_window_stats)
    return agg


def run_replay(
    strategy_name: str, symbol: str, windows_csv: Path,
    wrapper: str | None, wrapper_kwargs: dict,
) -> dict:
    entry = REGISTRY[(symbol, strategy_name)]
    strategy = entry["strategy_cls"]()
    ltf = _load_ohlcv(symbol, strategy.ltf)
    htfs = {tf: _load_ohlcv(symbol, tf) for tf in strategy.required_htfs}

    windows = pd.read_csv(windows_csv)
    per_stats: list[dict] = []
    per_trades: list[pd.DataFrame] = []
    per_equity: list[pd.DataFrame] = []
    for _, w in windows.iterrows():
        try:
            tr, eq, st = replay_one_window(strategy, ltf, htfs, w, symbol, wrapper, wrapper_kwargs)
            per_stats.append(st)
            per_trades.append(tr.assign(window=int(w["window"])) if len(tr) > 0 else tr)
            per_equity.append(eq.assign(window=int(w["window"])) if len(eq) > 0 else eq)
        except Exception as e:
            logger.warning(f"window {int(w['window'])}: replay failed: {e}")
            per_stats.append({})
            per_trades.append(pd.DataFrame())
            per_equity.append(pd.DataFrame())
    agg = aggregate(per_stats, per_trades, per_equity)
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-csv", required=True, type=Path)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--strategy", required=True, help="Strategy short name (registry key)")
    ap.add_argument("--wrapper", default="none", choices=["none", "limit_entry", "partial_tp"])
    ap.add_argument("--limit-offset-atr", type=float, default=0.3)
    ap.add_argument("--expire-bars", type=int, default=5)
    ap.add_argument("--tp1-frac", type=float, default=0.5)
    # Grid sweep mode
    ap.add_argument("--sweep-limit-offset", nargs="+", type=float, default=None,
                    help="If set, sweeps limit_entry over these offset values (overrides --wrapper)")
    ap.add_argument("--sweep-expire", nargs="+", type=int, default=None,
                    help="If set, sweeps expire_bars values (used with --sweep-limit-offset)")
    args = ap.parse_args()

    # Sweep mode
    if args.sweep_limit_offset is not None:
        expires = args.sweep_expire if args.sweep_expire else [5]
        rows = []
        # Baseline (no wrapper)
        logger.info("=== baseline (no wrapper) ===")
        agg = run_replay(args.strategy, args.symbol, args.windows_csv, None, {})
        rows.append({"wrapper": "none", **{k: agg.get(k) for k in
                    ["n_trades", "winrate", "profit_factor", "expectancy_R",
                     "max_drawdown_pct", "total_return_pct", "n_pos_windows"]}})
        for off, exp in itertools.product(args.sweep_limit_offset, expires):
            logger.info(f"=== limit_entry offset={off} expire={exp} ===")
            agg = run_replay(args.strategy, args.symbol, args.windows_csv,
                             "limit_entry", {"offset_atr": off, "expire_bars": exp})
            rows.append({"wrapper": f"limit({off},{exp})",
                        **{k: agg.get(k) for k in
                           ["n_trades", "winrate", "profit_factor", "expectancy_R",
                            "max_drawdown_pct", "total_return_pct", "n_pos_windows"]}})
        df = pd.DataFrame(rows)
        print("\n=== SWEEP RESULTS ===")
        print(df.to_string(index=False))
        out_path = args.windows_csv.parent / f"replay_sweep_{args.strategy}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Wrote {out_path}")
        return 0

    # Single config
    if args.wrapper == "limit_entry":
        wk = {"offset_atr": args.limit_offset_atr, "expire_bars": args.expire_bars}
    elif args.wrapper == "partial_tp":
        wk = {"tp1_frac": args.tp1_frac}
    else:
        wk = {}
    agg = run_replay(args.strategy, args.symbol, args.windows_csv,
                     args.wrapper if args.wrapper != "none" else None, wk)
    print("\n=== REPLAY RESULT ===")
    print(json.dumps(agg, indent=2, default=str) if False else
          "\n".join(f"  {k}: {v}" for k, v in agg.items()))
    return 0


if __name__ == "__main__":
    import json  # late import only used in main()
    sys.exit(main())
