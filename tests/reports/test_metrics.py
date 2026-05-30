"""Unit tests for the metrics module.

Run: python -m tests.reports.test_metrics
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.reports import compute_stats, composite_objective


def _eq_curve(equity_values: list[float], freq: str = "5min") -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=len(equity_values), freq=freq, tz="UTC")
    return pd.DataFrame({"timestamp": ts, "equity": equity_values})


def test_empty_trades_returns_well_formed_dict() -> None:
    eq = _eq_curve([10_000.0] * 10)
    s = compute_stats(trades=pd.DataFrame(), equity_curve=eq, initial_equity=10_000.0)
    assert s["n_trades"] == 0
    assert s["winrate"] == 0.0
    assert s["profit_factor"] == 0.0
    assert s["max_drawdown_pct"] == 0.0
    assert s["sharpe_annualized"] == 0.0
    print("PASS test_empty_trades_returns_well_formed_dict")


def test_compute_stats_basic_winrate_and_pf() -> None:
    # 3 winners (+10, +20, +30), 2 losers (-5, -15) → WR=3/5=0.6, PF=60/20=3.0
    trades = pd.DataFrame({
        "pnl":        [10.0, -5.0, 20.0, -15.0, 30.0],
        "R_realized": [1.0, -1.0,  2.0,  -1.5,  2.0],
        "exit_reason":["tp", "sl",  "tp", "sl",  "tp"],
        "side":       [1,    1,    -1,    1,    -1],
        "bars_held":  [3,    2,     5,    1,     4],
    })
    eq = _eq_curve([10_000.0, 10_010.0, 10_005.0, 10_025.0, 10_010.0, 10_040.0])
    s = compute_stats(trades=trades, equity_curve=eq, initial_equity=10_000.0)
    assert abs(s["winrate"] - 0.6) < 1e-9
    assert abs(s["profit_factor"] - 3.0) < 1e-9
    assert s["n_trades"] == 5
    assert s["n_winners"] == 3
    assert s["n_losers"] == 2
    assert s["long_n"] == 3
    assert s["short_n"] == 2
    assert s["max_consecutive_losses"] == 1  # losers don't appear back-to-back
    print("PASS test_compute_stats_basic_winrate_and_pf")


def test_compute_stats_drawdown() -> None:
    # Equity: 100 → 120 (peak) → 90 → 100 → 110. Max DD = (90-120)/120 = -25%.
    eq = _eq_curve([100.0, 120.0, 90.0, 100.0, 110.0])
    trades = pd.DataFrame({
        "pnl": [20.0, -30.0, 10.0, 10.0],
        "R_realized": [1.0, -1.5, 0.5, 0.5],
        "exit_reason": ["tp", "sl", "tp", "tp"],
        "side": [1, 1, 1, 1],
        "bars_held": [1, 1, 1, 1],
    })
    s = compute_stats(trades=trades, equity_curve=eq, initial_equity=100.0)
    assert abs(s["max_drawdown_pct"] - 0.25) < 1e-9
    assert abs(s["max_drawdown_abs"] - 30.0) < 1e-9
    print("PASS test_compute_stats_drawdown")


def test_include_force_eod_flag() -> None:
    trades = pd.DataFrame({
        "pnl":        [10.0, -5.0, 100.0],   # the +100 is a force_eod we want to exclude
        "R_realized": [1.0, -1.0,  10.0],
        "exit_reason":["tp", "sl",  "force_eod"],
        "side":       [1,    1,     1],
        "bars_held":  [3,    2,     1],
    })
    eq = _eq_curve([10_000.0, 10_010.0, 10_005.0, 10_105.0])
    with_eod = compute_stats(trades=trades, equity_curve=eq, initial_equity=10_000.0, include_force_eod=True)
    without_eod = compute_stats(trades=trades, equity_curve=eq, initial_equity=10_000.0, include_force_eod=False)
    assert with_eod["n_trades"] == 3
    assert without_eod["n_trades"] == 2
    # WR without the force_eod winner: 1/2 = 0.5
    assert abs(without_eod["winrate"] - 0.5) < 1e-9
    print("PASS test_include_force_eod_flag")


def test_composite_objective_rejects_low_trade_count() -> None:
    stats = {
        "n_trades": 10, "winrate": 0.9, "profit_factor": 5.0,
        "expectancy_R": 1.0, "max_drawdown_pct": 0.05, "sharpe_annualized": 2.0,
    }
    s = composite_objective(stats, min_trades=30)
    assert s == float("-inf"), f"low trade count should be -inf, got {s}"
    print("PASS test_composite_objective_rejects_low_trade_count")


def test_composite_objective_rejects_negative_expectancy() -> None:
    stats = {
        "n_trades": 100, "winrate": 0.9, "profit_factor": 0.5,
        "expectancy_R": -0.1, "max_drawdown_pct": 0.05, "sharpe_annualized": 0.5,
    }
    s = composite_objective(stats, min_trades=30, expectancy_floor_R=0.0)
    assert s == float("-inf")
    print("PASS test_composite_objective_rejects_negative_expectancy")


def test_composite_objective_rewards_better_combo() -> None:
    """Better WR + PF + lower DD → higher composite score."""
    weak = {
        "n_trades": 100, "winrate": 0.55, "profit_factor": 1.2,
        "expectancy_R": 0.1, "max_drawdown_pct": 0.10, "sharpe_annualized": 0.5,
    }
    strong = {
        "n_trades": 100, "winrate": 0.75, "profit_factor": 2.5,
        "expectancy_R": 0.5, "max_drawdown_pct": 0.05, "sharpe_annualized": 1.5,
    }
    s_weak = composite_objective(weak)
    s_strong = composite_objective(strong)
    assert s_strong > s_weak, f"strong ({s_strong}) should beat weak ({s_weak})"
    print(f"PASS test_composite_objective_rewards_better_combo — weak={s_weak:.3f} strong={s_strong:.3f}")


if __name__ == "__main__":
    test_empty_trades_returns_well_formed_dict()
    test_compute_stats_basic_winrate_and_pf()
    test_compute_stats_drawdown()
    test_include_force_eod_flag()
    test_composite_objective_rejects_low_trade_count()
    test_composite_objective_rejects_negative_expectancy()
    test_composite_objective_rewards_better_combo()
    print("\nAll metrics tests passed.")
