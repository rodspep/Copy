"""Trend indicators: SMA, EMA, MACD, ADX.

All functions are pure: they take pandas Series/DataFrames and return new Series/
DataFrames with the same index. All rolling/ewm operations specify `min_periods`
explicitly so the warmup period is NaN (per parity ADR §9 — no implicit warmup,
no future leakage through NaN backfill).

Inputs:
- For Series-input functions (sma, ema, macd): pass `close` price as a tz-aware Series.
- For OHLC-input functions (adx): pass a DataFrame with columns 'high', 'low', 'close'.

Outputs:
- Single-output indicators return a Series.
- Multi-output indicators return a DataFrame with named columns (so callers don't
  unpack tuples and lose self-documentation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average. First `period-1` values are NaN."""
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    return close.rolling(window=period, min_periods=period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (adjust=False, Wilder-compatible recursion).

    First `period-1` values are NaN. Uses pandas' `ewm(span=period, adjust=False)`
    which yields the standard EMA recursion `EMA_t = α * close_t + (1-α) * EMA_{t-1}`
    with `α = 2 / (period + 1)`.
    """
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def _wilder_smooth(s: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (used by ADX, RSI, ATR).

    Equivalent to `ewm(alpha=1/period, adjust=False)` with explicit min_periods,
    but seeded by an SMA over the first `period` values to match the canonical
    Wilder definition (avoids the small early-period bias of pure ewm).
    """
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    s = s.astype("float64")
    values = s.to_numpy()
    n = len(values)
    arr = np.full(n, np.nan, dtype="float64")
    if n < period:
        return pd.Series(arr, index=s.index)

    # Find the first non-NaN value. Inputs like close.diff() and true_range() begin
    # with one or more NaNs by construction; the seed window must start AFTER them.
    valid_positions = np.flatnonzero(~np.isnan(values))
    if len(valid_positions) == 0:
        return pd.Series(arr, index=s.index)

    first_valid = int(valid_positions[0])
    seed_pos = first_valid + period - 1
    if seed_pos >= n:
        return pd.Series(arr, index=s.index)

    first_window = values[first_valid : seed_pos + 1]
    if np.isnan(first_window).any():
        # NaNs interrupting the seed window — let Wilder remain undefined here.
        return pd.Series(arr, index=s.index)

    seed = float(first_window.mean())
    arr[seed_pos] = seed

    # Recursive Wilder update from the bar AFTER the seed.
    prev = seed
    for i in range(seed_pos + 1, n):
        v = values[i]
        if np.isnan(v):
            # Propagate NaN through Wilder smoothing without contaminating state.
            arr[i] = np.nan
            continue
        cur = (prev * (period - 1) + v) / period
        arr[i] = cur
        prev = cur
    return pd.Series(arr, index=s.index)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD: returns DataFrame with columns ['macd', 'signal', 'hist'].

    macd   = EMA(close, fast) - EMA(close, slow)
    signal = EMA(macd, signal)
    hist   = macd - signal
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    # Signal line: EMA of macd_line, also adjust=False, explicit min_periods.
    # Note: macd_line itself is NaN for the first slow-1 bars, so the signal
    # line will be NaN for slow-1 + signal-1 bars after combining.
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": hist},
        index=close.index,
    )


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range component used by ATR and ADX.

    TR_t = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)

    First value is NaN (no previous close).
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    tr = pd.concat([a, b, c], axis=1).max(axis=1)
    tr.iloc[0] = np.nan  # explicit: no previous close at t=0
    return tr


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index — returns DataFrame ['plus_di', 'minus_di', 'adx'].

    Standard Wilder definition:
      +DM_t = max(high_t - high_{t-1}, 0)   if  high_t-high_{t-1} > low_{t-1}-low_t  else 0
      -DM_t = max(low_{t-1} - low_t, 0)     if  low_{t-1}-low_t > high_t-high_{t-1}  else 0
      TR    = true range
      +DI   = 100 * Wilder(+DM, period) / Wilder(TR, period)
      -DI   = 100 * Wilder(-DM, period) / Wilder(TR, period)
      DX    = 100 * |+DI - -DI| / (+DI + -DI)
      ADX   = Wilder(DX, period)
    """
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("adx requires columns ['high','low','close']")
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    high = df["high"]
    low = df["low"]
    up_move = high.diff()
    down_move = -low.diff()  # positive when low decreases

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
        dtype="float64",
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
        dtype="float64",
    )
    # First row has no previous bar; force NaN so DM aligns with TR (which is also
    # NaN at t=0). Wilder smoothing now skips leading NaNs uniformly across DM and TR.
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan

    tr = true_range(df)
    tr_smooth = _wilder_smooth(tr, period)
    plus_dm_smooth = _wilder_smooth(plus_dm, period)
    minus_dm_smooth = _wilder_smooth(minus_dm, period)

    # Avoid divide-by-zero — replace 0 TR with NaN so DI becomes NaN there.
    tr_safe = tr_smooth.where(tr_smooth > 0)
    plus_di = 100.0 * (plus_dm_smooth / tr_safe)
    minus_di = 100.0 * (minus_dm_smooth / tr_safe)

    di_sum = plus_di + minus_di
    di_sum_safe = di_sum.where(di_sum > 0)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum_safe
    adx_line = _wilder_smooth(dx, period)

    return pd.DataFrame(
        {"plus_di": plus_di, "minus_di": minus_di, "adx": adx_line},
        index=df.index,
    )
