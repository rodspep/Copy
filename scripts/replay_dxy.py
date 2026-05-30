"""Apply DXY block-opposite wrapper on existing strategy OOS results.

Sweeps (slope_window, block_threshold_pct) to find sweet spot.

Usage:
  python -m scripts.replay_dxy --windows-csv <path> --strategy mtf_smc_entry --symbol XAUUSD
"""
from __future__ import annotations

import argparse, ast
from pathlib import Path

import pandas as pd
import numpy as np

from src.data.histdata_loader import load
from src.data.yahoo_loader import load as load_yahoo
from src.backtest import run_backtest
from src.reports import compute_stats
from src.strategies.registry import REGISTRY
from src.strategies.signal_wrappers import dxy_block_opposite_wrapper


def _parse_param(v):
    if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
        if isinstance(v, np.bool_): return bool(v)
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, np.floating): return float(v)
        return v
    if isinstance(v, str):
        sv = v.strip()
        if sv.lower() in ("true","false"): return sv.lower() == "true"
        try: return ast.literal_eval(sv)
        except: return sv
    return v


def replay_one_config(strategy, ltf, htfs, windows, symbol, dxy_daily,
                       slope_window, block_threshold_pct):
    """Replay all OOS windows with DXY filter applied. Return aggregated stats."""
    per_trades = []
    per_equity = []
    per_stats = []
    for _, w in windows.iterrows():
        ts = pd.Timestamp(w["test_start"], tz="UTC")
        te = pd.Timestamp(w["test_end"], tz="UTC")
        test_ltf = ltf[(ltf["timestamp"] >= ts) & (ltf["timestamp"] < te)].reset_index(drop=True)
        if test_ltf.empty: continue
        test_htfs = {tf: src_df[(src_df["timestamp"] >= ts - pd.Timedelta(days=30)) &
                                (src_df["timestamp"] < te)].reset_index(drop=True)
                     for tf, src_df in htfs.items()}
        params = {}
        for col in w.index:
            if col.startswith("best_") and col != "best_score":
                v = w[col]
                if pd.isna(v): continue
                params[col[5:]] = _parse_param(v)
        try:
            sigs = strategy.generate_signals(test_ltf, test_htfs, params).signals
            if slope_window is not None and block_threshold_pct is not None:
                sigs = dxy_block_opposite_wrapper(test_ltf, sigs, dxy_daily,
                                                  slope_window=slope_window,
                                                  block_threshold_pct=block_threshold_pct)
            bt = run_backtest(test_ltf, sigs, symbol=symbol, ltf_tf=strategy.ltf,
                              params={"initial_equity":10000.0,"risk_pct":0.005,"compounding":True})
            per_trades.append(bt["trades"])
            per_equity.append(bt["equity_curve"])
            st = compute_stats(trades=bt["trades"], equity_curve=bt["equity_curve"],
                              initial_equity=10000.0, bars_per_year=252*24*12,
                              include_force_eod=False)
            per_stats.append(st)
        except Exception as e:
            print(f"  window {int(w['window'])}: {e}")
            per_stats.append({})
    if not per_trades:
        return {"n_trades":0,"winrate":0,"profit_factor":0,"expectancy_R":0,
                "n_pos_windows":0}
    all_trades = pd.concat(per_trades, ignore_index=True)
    all_equity = pd.concat(per_equity, ignore_index=True)
    if all_trades.empty:
        return {"n_trades":0,"winrate":0,"profit_factor":0,"expectancy_R":0,
                "n_pos_windows":0}
    agg = compute_stats(trades=all_trades, equity_curve=all_equity[["timestamp","equity"]],
                       initial_equity=10000.0, bars_per_year=252*24*12, include_force_eod=False)
    agg["n_pos_windows"] = sum(1 for s in per_stats if s.get("expectancy_R",0) > 0)
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-csv", required=True, type=Path)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--symbol", default="XAUUSD")
    args = ap.parse_args()

    entry = REGISTRY[(args.symbol, args.strategy)]
    strat = entry["strategy_cls"]()
    ltf = load(args.symbol, strat.ltf) if args.symbol == "XAUUSD" else None
    if ltf is None:
        raise NotImplementedError("Only XAUUSD wired for replay_dxy")
    htfs = {tf: load(args.symbol, tf) for tf in strat.required_htfs}
    dxy = load_yahoo("DXY")
    windows = pd.read_csv(args.windows_csv)

    rows = []
    print("Baseline (no filter):")
    base = replay_one_config(strat, ltf, htfs, windows, args.symbol, dxy, None, None)
    rows.append({"config":"baseline",**{k:base.get(k) for k in ["n_trades","winrate","profit_factor","expectancy_R","max_drawdown_pct","total_return_pct","n_pos_windows"]}})

    for sw in [3, 5, 10]:
        for thr in [0.3, 0.5, 0.8, 1.2]:
            print(f"DXY filter slope_window={sw}, threshold={thr}%:")
            r = replay_one_config(strat, ltf, htfs, windows, args.symbol, dxy, sw, thr)
            rows.append({"config":f"sw={sw},thr={thr}",**{k:r.get(k) for k in
                ["n_trades","winrate","profit_factor","expectancy_R","max_drawdown_pct","total_return_pct","n_pos_windows"]}})

    df = pd.DataFrame(rows)
    print("\n=== DXY FILTER SWEEP ===")
    print(df.to_string(index=False))
    out = args.windows_csv.parent / f"replay_dxy_{args.strategy}.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
