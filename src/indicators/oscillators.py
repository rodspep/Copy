"""Oscillators: RSI, Stochastic.

Same conventions as `trend.py`: pure functions, explicit `min_periods`, NaN warmup,
no implicit forward-fill, no lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .trend import _wilder_smooth


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder).

    Uses Wilder smoothing of gains/losses (matching standard charting software,
    not the naive SMA variant). First `period` values are NaN.
    """
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)

    # If avg_loss == 0, RSI = 100 (no losses).
    # If both == 0 (flat market), RSI = 50 by convention (no momentum either way).
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    rsi_line = 100.0 - (100.0 / (1.0 + rs))

    # Cases where avg_loss == 0:
    flat = (avg_gain == 0) & (avg_loss == 0)
    rsi_line = rsi_line.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi_line = rsi_line.mask(flat, 50.0)
    return rsi_line


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """Stochastic oscillator — returns DataFrame ['k', 'd'].

    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k     = SMA(raw_k, smooth_k)            # "%K slow" — the line typically plotted
    d     = SMA(k, d_period)                # signal line

    If `smooth_k == 1`, %K equals raw_k (the "fast" stochastic).
    """
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("stochastic requires columns ['high','low','close']")
    if min(k_period, d_period, smooth_k) <= 0:
        raise ValueError("all periods must be > 0")

    highest_high = df["high"].rolling(window=k_period, min_periods=k_period).max()
    lowest_low = df["low"].rolling(window=k_period, min_periods=k_period).min()
    rng = highest_high - lowest_low
    # When range is zero (flat window), %K is undefined; set to 50 (neutral).
    raw_k = 100.0 * (df["close"] - lowest_low) / rng.where(rng > 0)
    raw_k = raw_k.mask(rng == 0, 50.0)

    if smooth_k == 1:
        k = raw_k
    else:
        k = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return pd.DataFrame({"k": k, "d": d}, index=df.index)
