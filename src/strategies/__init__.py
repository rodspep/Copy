"""Strategy library.

Subpackages:
- `xau/` — XAU/USD-specific strategy candidates.
- `btc/` — BTC/USDT-specific strategy candidates.

Public API:
    from src.strategies import Strategy, StrategyResult, empty_signals, validate_signals
"""
from .base import Strategy, StrategyResult, empty_signals, validate_signals, VALID_ACTIONS

__all__ = ["Strategy", "StrategyResult", "empty_signals", "validate_signals", "VALID_ACTIONS"]
