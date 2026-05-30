"""Trade & equity-curve metrics for ranking strategies.

This module is consumed by both the backtest engine output (for one-shot reports)
and the walk-forward optimizer (for the composite objective). Metrics are
computed from the `trades` DataFrame and `equity_curve` DataFrame produced by
`src.backtest.run_backtest`.

Conventions:
- All percentages are returned as fractions (0.55 = 55%), not 55.0.
- All "R-multiples" use `R_realized = pnl / risk_amount` from the trade log (which
  already reflects costs — see parity ADR §4 "What R means").
- Drawdown is computed from the equity curve at bar-close mark-to-market.
- The composite score `composite_objective` is the function the optimizer maximizes;
  any change to its formula is a research decision and should be documented in
  `docs/decisions/`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Core stats
# -----------------------------------------------------------------------------

def compute_stats(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_equity: float,
    bars_per_year: float = 252 * 24 * 12,  # default: M5 24/7 trading (BTC). Override for XAU.
    risk_free_rate: float = 0.0,
    include_force_eod: bool = True,
) -> dict:
    """Compute summary stats from a backtest result.

    Args:
        trades         — DataFrame with at least columns: pnl, R_realized, exit_reason, side, bars_held.
        equity_curve   — DataFrame with columns ['timestamp','equity'].
        initial_equity — starting equity (denominator for total return).
        bars_per_year  — annualization factor for Sharpe/Sortino. Defaults to M5 24/7.
                         For M5 on XAU (5 sessions x 24h x 12 bars), use 252 * 24 * 12 too,
                         but for fewer trading hours adjust. Be explicit per use.
        risk_free_rate — annualized; subtracted from mean return for Sharpe.
        include_force_eod — if False, drops force-EOD trades from the stats (their
                            live counterpart doesn't exist).

    Returns dict of metrics:
      - n_trades, n_winners, n_losers, n_force_eod
      - winrate (fraction)
      - profit_factor
      - expectancy_per_trade, expectancy_R
      - avg_win, avg_loss, avg_RR (avg_win / |avg_loss|)
      - median_R, std_R
      - max_consecutive_losses, max_consecutive_wins
      - long_winrate, short_winrate, long_n, short_n
      - total_pnl, total_return_pct, final_equity
      - max_drawdown_pct, max_drawdown_abs
      - sharpe_annualized, sortino_annualized, calmar
      - exit_reason_counts (dict)
      - avg_bars_held
    """
    if trades is None or trades.empty:
        # Even with no trades we want a well-formed dict for the optimizer.
        return _empty_stats(initial_equity, equity_curve)

    df = trades.copy()
    if not include_force_eod:
        df = df[df["exit_reason"] != "force_eod"].copy()
    if df.empty:
        return _empty_stats(initial_equity, equity_curve)

    n_trades = len(df)
    winners_mask = df["pnl"] > 0
    losers_mask = df["pnl"] < 0
    n_winners = int(winners_mask.sum())
    n_losers = int(losers_mask.sum())
    n_force_eod = int((trades["exit_reason"] == "force_eod").sum())

    winrate = n_winners / n_trades if n_trades > 0 else 0.0

    gross_profit = float(df.loc[winners_mask, "pnl"].sum())
    gross_loss = float(-df.loc[losers_mask, "pnl"].sum())  # positive
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    expectancy = float(df["pnl"].mean())
    expectancy_R = float(df["R_realized"].mean()) if "R_realized" in df.columns else np.nan

    avg_win = float(df.loc[winners_mask, "pnl"].mean()) if n_winners > 0 else 0.0
    avg_loss = float(df.loc[losers_mask, "pnl"].mean()) if n_losers > 0 else 0.0
    avg_rr = (avg_win / abs(avg_loss)) if avg_loss < 0 else float("inf") if avg_win > 0 else 0.0

    median_R = float(df["R_realized"].median()) if "R_realized" in df.columns else np.nan
    std_R = float(df["R_realized"].std(ddof=1)) if "R_realized" in df.columns and len(df) > 1 else 0.0

    # Consecutive streaks
    is_win = (df["pnl"] > 0).to_numpy()
    max_streak_win = _max_consecutive(is_win, target=True)
    max_streak_loss = _max_consecutive(~is_win & (df["pnl"] < 0).to_numpy(), target=True)

    # Per-side stats
    long_df = df[df["side"] == 1]
    short_df = df[df["side"] == -1]
    long_n, short_n = len(long_df), len(short_df)
    long_wr = (long_df["pnl"] > 0).mean() if long_n > 0 else 0.0
    short_wr = (short_df["pnl"] > 0).mean() if short_n > 0 else 0.0

    # Equity-curve-derived
    eq = equity_curve["equity"].to_numpy(dtype="float64")
    ts = equity_curve["timestamp"].to_numpy() if "timestamp" in equity_curve.columns else None
    final_equity = float(eq[-1])
    total_pnl = final_equity - initial_equity
    total_return_pct = total_pnl / initial_equity if initial_equity > 0 else 0.0

    # Drawdown — running max of equity, then (equity - peak) / peak
    running_max = np.maximum.accumulate(eq)
    dd = eq - running_max  # ≤ 0
    dd_pct = np.where(running_max > 0, dd / running_max, 0.0)
    max_dd_abs = float(-dd.min())
    max_dd_pct = float(-dd_pct.min())

    # Sharpe / Sortino — from bar-level equity returns. Use log returns for additivity.
    rets = np.diff(np.log(np.maximum(eq, 1e-12)))
    if len(rets) > 1:
        mean_r = rets.mean()
        std_r = rets.std(ddof=1)
        # Per-bar excess return; risk-free is annualized so convert.
        rf_per_bar = (risk_free_rate / bars_per_year) if bars_per_year > 0 else 0.0
        excess = mean_r - rf_per_bar
        sharpe = (excess / std_r) * np.sqrt(bars_per_year) if std_r > 0 else 0.0
        downside = rets[rets < 0]
        downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
        sortino = (excess / downside_std) * np.sqrt(bars_per_year) if downside_std > 0 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    # Calmar: annualized return / max drawdown
    if ts is not None and len(ts) > 1:
        span_days = (pd.Timestamp(ts[-1]) - pd.Timestamp(ts[0])).total_seconds() / 86400.0
        ann_factor = 365.25 / span_days if span_days > 0 else 0.0
    else:
        ann_factor = 0.0
    annualized_return_pct = total_return_pct * ann_factor if ann_factor > 0 else 0.0
    calmar = (annualized_return_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

    exit_reason_counts = dict(df["exit_reason"].value_counts())

    avg_bars_held = float(df["bars_held"].mean()) if "bars_held" in df.columns else 0.0

    return {
        "n_trades": int(n_trades),
        "n_winners": n_winners,
        "n_losers": n_losers,
        "n_force_eod": n_force_eod,
        "winrate": float(winrate),
        "profit_factor": float(profit_factor),
        "expectancy_per_trade": float(expectancy),
        "expectancy_R": float(expectancy_R) if not np.isnan(expectancy_R) else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_RR": float(avg_rr),
        "median_R": float(median_R) if not np.isnan(median_R) else 0.0,
        "std_R": std_R,
        "max_consecutive_wins": int(max_streak_win),
        "max_consecutive_losses": int(max_streak_loss),
        "long_n": long_n,
        "short_n": short_n,
        "long_winrate": float(long_wr),
        "short_winrate": float(short_wr),
        "total_pnl": float(total_pnl),
        "total_return_pct": float(total_return_pct),
        "final_equity": final_equity,
        "max_drawdown_pct": float(max_dd_pct),
        "max_drawdown_abs": float(max_dd_abs),
        "sharpe_annualized": float(sharpe),
        "sortino_annualized": float(sortino),
        "calmar": float(calmar),
        "annualized_return_pct": float(annualized_return_pct),
        "exit_reason_counts": exit_reason_counts,
        "avg_bars_held": avg_bars_held,
    }


def _max_consecutive(boolean_arr: np.ndarray, target: bool = True) -> int:
    """Length of the longest run of `target` in `boolean_arr`."""
    if len(boolean_arr) == 0:
        return 0
    runs = 0
    best = 0
    for v in boolean_arr:
        if bool(v) is target:
            runs += 1
            if runs > best:
                best = runs
        else:
            runs = 0
    return best


def _empty_stats(initial_equity: float, equity_curve: pd.DataFrame) -> dict:
    """Stats dict for a run with zero trades — keeps the optimizer happy."""
    final_equity = (
        float(equity_curve["equity"].iloc[-1])
        if equity_curve is not None and len(equity_curve) > 0
        else initial_equity
    )
    return {
        "n_trades": 0, "n_winners": 0, "n_losers": 0, "n_force_eod": 0,
        "winrate": 0.0, "profit_factor": 0.0,
        "expectancy_per_trade": 0.0, "expectancy_R": 0.0,
        "avg_win": 0.0, "avg_loss": 0.0, "avg_RR": 0.0,
        "median_R": 0.0, "std_R": 0.0,
        "max_consecutive_wins": 0, "max_consecutive_losses": 0,
        "long_n": 0, "short_n": 0, "long_winrate": 0.0, "short_winrate": 0.0,
        "total_pnl": final_equity - initial_equity,
        "total_return_pct": (final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0,
        "final_equity": final_equity,
        "max_drawdown_pct": 0.0, "max_drawdown_abs": 0.0,
        "sharpe_annualized": 0.0, "sortino_annualized": 0.0, "calmar": 0.0,
        "annualized_return_pct": 0.0,
        "exit_reason_counts": {},
        "avg_bars_held": 0.0,
    }


# -----------------------------------------------------------------------------
# Composite objective — what the optimizer maximizes
# -----------------------------------------------------------------------------

def composite_objective(
    stats: dict,
    min_trades: int = 30,
    wr_weight: float = 1.0,
    pf_weight: float = 0.5,
    dd_penalty: float = 1.5,
    expectancy_floor_R: float = 0.0,
) -> float:
    """A scalar score the optimizer maximizes. Designed so degenerate solutions
    (huge WR with 1 trade, or PF=inf with one lucky win, etc.) score low.

    Formula:
      if n_trades < min_trades:                       → -inf
      if expectancy_R <= expectancy_floor_R:          → -inf (forces sustainably positive R)
      else:
        score = wr_weight * winrate
              + pf_weight * tanh(profit_factor - 1)   # diminishing return on PF
              - dd_penalty * max_drawdown_pct
              + 0.1 * tanh(sharpe_annualized)         # mild Sharpe nudge

    Per [[feedback-per-symbol-strategies]] the optimizer runs PER (symbol,
    strategy) so this objective doesn't need cross-symbol normalization.
    """
    n_trades = stats.get("n_trades", 0)
    if n_trades < min_trades:
        return float("-inf")
    if stats.get("expectancy_R", 0.0) <= expectancy_floor_R:
        return float("-inf")

    wr = stats.get("winrate", 0.0)
    pf = stats.get("profit_factor", 0.0)
    dd = stats.get("max_drawdown_pct", 0.0)
    sharpe = stats.get("sharpe_annualized", 0.0)

    score = (
        wr_weight * wr
        + pf_weight * float(np.tanh(pf - 1.0))
        - dd_penalty * dd
        + 0.1 * float(np.tanh(sharpe))
    )
    return float(score)
