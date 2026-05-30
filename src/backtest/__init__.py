"""Backtest engine — vectorized simulator implementing the parity ADR.

Public API:
    from src.backtest import run_backtest
"""
from .engine import run_backtest, Position
from .fills import (
    entry_fill_price,
    round_price_to_tick,
    round_sl_tp,
    validate_sltp_after_entry,
    evaluate_sl_tp_on_bar,
    market_exit_price,
    commission,
    SlTpFill,
)
from .sizing import position_size

__all__ = [
    "run_backtest", "Position",
    "entry_fill_price", "round_price_to_tick", "round_sl_tp",
    "validate_sltp_after_entry", "evaluate_sl_tp_on_bar",
    "market_exit_price", "commission", "SlTpFill",
    "position_size",
]
