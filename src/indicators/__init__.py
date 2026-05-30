"""Indicator library.

All indicators are pure, vectorized, lookahead-safe (explicit `min_periods`,
no implicit forward-fill). See `docs/decisions/backtest_live_parity.md` §9 for
the patterns that are forbidden.

Public surface — re-exported for short import paths:

    from src.indicators import ema, macd, adx, rsi, stochastic, atr, bollinger, keltner, vwap, align_htf_to_ltf
"""
from .trend import sma, ema, macd, adx, true_range
from .oscillators import rsi, stochastic
from .volatility import atr, bollinger, keltner
from .volume import vwap, obv, cvd
from .htf import align_htf_to_ltf
from .smc import swings, structure_breaks, fair_value_gaps, order_blocks, liquidity_sweeps

__all__ = [
    "sma", "ema", "macd", "adx", "true_range",
    "rsi", "stochastic",
    "atr", "bollinger", "keltner",
    "vwap", "obv", "cvd",
    "align_htf_to_ltf",
    "swings", "structure_breaks", "fair_value_gaps", "order_blocks", "liquidity_sweeps",
]
