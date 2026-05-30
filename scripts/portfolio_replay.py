"""Portfolio replay: run all XAU strategies in parallel with shared risk budget.

For each strategy:
  - Load its OOS-best params per window (from windows.csv)
  - Regenerate signals on each test window
  - Run backtest with risk_pct = TOTAL_RISK / N_STRATEGIES

Aggregate:
  - Concatenate all trades, all equity curves
  - Compute portfolio-level WR / exp_R / MaxDD
  - Per-window: sum PnL across all 7 strategies → "is window positive?"

This is the simplest portfolio model. It doesn't try to avoid overlapping
entries — each strategy operates independently. Real implementation would
need a risk router to avoid concentrated direction exposure.

Usage:
  python -m scripts.portfolio_replay
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.histdata_loader import load
from src.backtest import run_backtest
from src.reports import compute_stats
from src.strategies.registry import REGISTRY


STRATEGY_PATHS = {
    'mtf_smc_entry':      'results/optimize/XauMtfSmcEntry_XAUUSD_20260527T084922/windows.csv',
    'ema_pullback':       'results/optimize/XauEmaPullback_XAUUSD_20260527T061551/windows.csv',
    'htf_trend_reversal': 'results/optimize/XauHtfTrendReversal_XAUUSD_20260527T100852/windows.csv',
    'smc_confluence':     'results/optimize/XauSmcConfluence_XAUUSD_20260527T074433/windows.csv',
    'liquidity_sweep_reversal': 'results/optimize/XauLiquiditySweepReversal_XAUUSD_20260527T064007/windows.csv',
    'ma34_cascade':       'results/optimize/XauMa34Cascade_XAUUSD_20260527T093251/windows.csv',
    'zone_anticipator':   'results/optimize/XauZoneAnticipator_XAUUSD_20260527T083026/windows.csv',
}

TOTAL_RISK_PCT = 0.005     # same as single-strategy default
PER_STRAT_RISK = TOTAL_RISK_PCT / len(STRATEGY_PATHS)


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


def replay_strategy(name: str, windows_path: str, ltf: pd.DataFrame,
                    htf_cache: dict[str, dict[str, pd.DataFrame]],
                    symbol: str = "XAUUSD") -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Replay one strategy's OOS windows. Return (trades, equity, per-window stats)."""
    entry = REGISTRY[(symbol, name)]
    strat = entry["strategy_cls"]()
    htfs_full = htf_cache[name]
    windows = pd.read_csv(windows_path)
    all_trades = []
    all_equity = []
    per_stats = []
    for _, w in windows.iterrows():
        wi = int(w["window"])
        ts = pd.Timestamp(w["test_start"], tz="UTC")
        te = pd.Timestamp(w["test_end"], tz="UTC")
        test_ltf = ltf[(ltf["timestamp"] >= ts) & (ltf["timestamp"] < te)].reset_index(drop=True)
        if test_ltf.empty:
            per_stats.append({"window": wi, "n_trades": 0, "pnl_sum": 0.0})
            continue
        test_htfs = {tf: src_df[(src_df["timestamp"] >= ts - pd.Timedelta(days=30)) &
                                (src_df["timestamp"] < te)].reset_index(drop=True)
                     for tf, src_df in htfs_full.items()}
        params = {}
        for col in w.index:
            if col.startswith("best_") and col != "best_score":
                v = w[col]
                if pd.isna(v): continue
                params[col[5:]] = _parse_param(v)
        try:
            sigs = strat.generate_signals(test_ltf, test_htfs, params).signals
            bt = run_backtest(test_ltf, sigs, symbol=symbol, ltf_tf=strat.ltf,
                              params={"initial_equity": 10000.0,
                                     "risk_pct": PER_STRAT_RISK,
                                     "compounding": True})
            tr = bt["trades"].assign(window=wi, strategy=name)
            eq = bt["equity_curve"].assign(window=wi, strategy=name)
            all_trades.append(tr)
            all_equity.append(eq)
            per_stats.append({"window": wi, "n_trades": len(tr),
                             "pnl_sum": float(tr["pnl"].sum()) if len(tr) > 0 else 0.0})
        except Exception as e:
            print(f"  {name} window {wi}: {e}")
            per_stats.append({"window": wi, "n_trades": 0, "pnl_sum": 0.0})
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    return trades, equity, per_stats


def main() -> int:
    print(f"Loading XAU data...")
    ltf = load("XAUUSD", "M5")
    htf_data = {tf: load("XAUUSD", tf) for tf in ["M15","H1","H4"]}

    # Build per-strategy HTF cache to avoid reload
    htf_cache = {}
    for name in STRATEGY_PATHS:
        strat = REGISTRY[("XAUUSD", name)]["strategy_cls"]()
        htf_cache[name] = {tf: htf_data[tf] for tf in strat.required_htfs}

    # Replay each strategy
    print(f"\nPer-strategy risk: {PER_STRAT_RISK*100:.4f}% (total {TOTAL_RISK_PCT*100:.2f}%)")
    print(f"Replaying {len(STRATEGY_PATHS)} strategies...\n")

    all_trades = []
    all_per_window = {}   # strategy -> [per-window stats]
    for name, path in STRATEGY_PATHS.items():
        print(f"=== {name} ===")
        trades, _equity, per_stats = replay_strategy(name, path, ltf, htf_cache)
        print(f"  trades: {len(trades)}, total PnL: ${trades['pnl'].sum() if len(trades)>0 else 0:.2f}")
        all_trades.append(trades)
        all_per_window[name] = per_stats

    # Aggregate
    portfolio_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if portfolio_trades.empty:
        print("No portfolio trades.")
        return 0

    portfolio_trades = portfolio_trades.sort_values("entry_time").reset_index(drop=True)

    print(f"\n{'='*100}")
    print("PORTFOLIO AGGREGATE")
    print('='*100)
    n = len(portfolio_trades)
    n_win = (portfolio_trades["R_realized"] > 0).sum()
    wr = n_win / n
    exp_r = portfolio_trades["R_realized"].mean()
    total_pnl = portfolio_trades["pnl"].sum()
    pnl_pct = total_pnl / 10000.0

    print(f"  total trades:       {n}")
    print(f"  winrate:            {wr:.3f}")
    print(f"  expectancy_R:       {exp_r:+.4f}")
    print(f"  total PnL:          ${total_pnl:+.2f} ({pnl_pct:+.2%})")
    print(f"  trades/day:         {n / (3.5*365):.2f}")

    # Per-window aggregate: sum PnL across strategies → window positive?
    print(f"\n  PER-WINDOW PORTFOLIO PnL")
    window_pnls = {}
    for name, stats in all_per_window.items():
        for s in stats:
            wi = s["window"]
            window_pnls.setdefault(wi, 0.0)
            window_pnls[wi] += s["pnl_sum"]
    n_pos_windows = sum(1 for p in window_pnls.values() if p > 0)
    print(f"  positive-PnL windows: {n_pos_windows}/{len(window_pnls)} = {n_pos_windows/len(window_pnls):.1%}")

    # Show per-window
    wp = pd.DataFrame([{"window": k, "portfolio_pnl": v}
                      for k, v in sorted(window_pnls.items())])
    print(f"\n  Top 10 windows:")
    print(wp.sort_values("portfolio_pnl", ascending=False).head(10).to_string(index=False))
    print(f"\n  Bottom 10 windows:")
    print(wp.sort_values("portfolio_pnl").head(10).to_string(index=False))

    out = Path("results/optimize/portfolio_replay.csv")
    wp.to_csv(out, index=False)
    portfolio_trades.to_parquet(out.parent / "portfolio_trades.parquet", index=False)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
