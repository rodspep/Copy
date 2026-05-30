"""Reporting & metrics for backtest results.

Public API:
    from src.reports import compute_stats, composite_objective
"""
from .metrics import compute_stats, composite_objective

__all__ = ["compute_stats", "composite_objective"]
