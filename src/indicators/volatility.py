"""Volatility indicators: ATR, Bollinger Bands, Keltner Channels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .trend import ema, true_range, _wilder_smooth


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder). First `period` values are NaN.

    Strategies in this repo use ATR for SL/TP sizing (parity ADR §4), so this
    must be exactly the standard Wilder ATR — not the SMA variant.
    """
    tr = true_range(df)
    return _wilder_smooth(tr, period)


def bollinger(
    close: pd.Series,
    period: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands — DataFrame ['mid', 'upper', 'lower', 'bandwidth', 'percent_b'].

    mid       = SMA(close, period)
    upper     = mid + n_std * stdev(close, period)
    lower     = mid - n_std * stdev(close, period)
    bandwidth = (upper - lower) / mid    # relative width, useful for squeeze detection
    percent_b = (close - lower) / (upper - lower)   # 0 = at lower band, 1 = at upper band
    """
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    mid = close.rolling(window=period, min_periods=period).mean()
    # ddof=0 (population stdev) is what most charting platforms use for BB.
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std

    width = upper - lower
    bandwidth = width / mid.where(mid != 0)
    # When upper == lower (flat market), %B is undefined -> NaN (don't fabricate 0.5).
    percent_b = (close - lower) / width.where(width > 0)

    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "bandwidth": bandwidth, "percent_b": percent_b},
        index=close.index,
    )


def keltner(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    atr_mult: float = 2.0,
) -> pd.DataFrame:
    """Keltner Channels — DataFrame ['mid', 'upper', 'lower', 'width'].

    mid   = EMA(close, ema_period)
    upper = mid + atr_mult * ATR(df, atr_period)
    lower = mid - atr_mult * ATR(df, atr_period)
    """
    if "close" not in df.columns:
        raise ValueError("keltner requires column 'close'")
    mid = ema(df["close"], ema_period)
    a = atr(df, atr_period)
    upper = mid + atr_mult * a
    lower = mid - atr_mult * a
    width = upper - lower
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "width": width},
        index=df.index,
    )
