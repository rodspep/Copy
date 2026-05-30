"""Price-reaction levels — price bands where price has reversed many times.

User concept ("1 giá ở khung nhỏ mà điểm giá đó và quanh nó có phản ứng giá
nhiều"): a price level is *important* when price keeps reacting (reversing) at
it. We measure that by counting confirmed swing pivots (highs AND lows) that
cluster within a small band, over a trailing window. A high count = a
"magnetic" support/resistance level that price respects.

This is essentially a causal volume-profile-by-pivots: instead of weighting by
traded volume (which HistData lacks reliably), we weight by *rejection events*
(swing pivots), which directly encode "price reacted here".

----
No-lookahead guarantee:

A swing pivot is only counted from its CONFIRMATION bar onward — the bar where
`swings_df['new_swing_high'/'new_swing_low']` first becomes True (which is the
pivot bar + `right` bars, per smc.swings). The pivot's price value is historical
(it happened earlier) but its *existence as a confirmed pivot* is only known at
the confirmation bar. So at bar i we count only pivots whose confirmation bar
is <= i. This matches parity ADR §1/§9.

----
Returns per bar:
  - react_low_count  : # confirmed pivots within band of THIS bar's low
  - react_low_level  : mean price of those pivots (NaN if none) — the cluster center
  - react_high_count : # confirmed pivots within band of THIS bar's high
  - react_high_level : mean price of those pivots (NaN if none)

The strategy interprets:
  - low near a high react_low_count cluster  => strong support (long bounce zone)
  - high near a high react_high_count cluster => strong resistance (short zone)
"""
from __future__ import annotations

import bisect
from collections import deque

import numpy as np
import pandas as pd


def reaction_levels(
    df: pd.DataFrame,
    swings_df: pd.DataFrame,
    *,
    atr_series: pd.Series,
    lookback: int = 500,
    band_atr_mult: float = 0.25,
) -> pd.DataFrame:
    """Count confirmed swing pivots clustered near each bar's low/high.

    Args:
        df:            OHLCV with 'low','high'.
        swings_df:     output of smc.swings() — shares index with df.
        atr_series:    ATR series (for adaptive band width).
        lookback:      trailing window in bars to consider pivots.
        band_atr_mult: band half-width = band_atr_mult × ATR. A pivot counts if
                       within ±band of the bar's low/high.

    Returns DataFrame indexed like df with react_low_count/level, react_high_count/level.
    """
    if len(df) != len(swings_df):
        raise ValueError("df and swings_df must share length")

    n = len(df)
    low = df["low"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    atr_v = atr_series.to_numpy(dtype="float64")
    band = float(band_atr_mult) * atr_v

    new_high = swings_df["new_swing_high"].to_numpy()
    new_low = swings_df["new_swing_low"].to_numpy()
    sh_price = swings_df["swing_high_price"].to_numpy(dtype="float64")
    sl_price = swings_df["swing_low_price"].to_numpy(dtype="float64")

    # Collect (confirm_bar, price) for every confirmed pivot, in bar order.
    cb_list: list[int] = []
    pr_list: list[float] = []
    for i in range(n):
        if new_high[i] and not np.isnan(sh_price[i]):
            cb_list.append(i)
            pr_list.append(float(sh_price[i]))
        if new_low[i] and not np.isnan(sl_price[i]):
            cb_list.append(i)
            pr_list.append(float(sl_price[i]))
    m = len(cb_list)

    react_low_count = np.zeros(n, dtype=np.int32)
    react_high_count = np.zeros(n, dtype=np.int32)
    react_low_level = np.full(n, np.nan, dtype="float64")
    react_high_level = np.full(n, np.nan, dtype="float64")

    if m == 0:
        return pd.DataFrame({
            "react_low_count": react_low_count, "react_low_level": react_low_level,
            "react_high_count": react_high_count, "react_high_level": react_high_level,
        }, index=df.index)

    active_prices: list[float] = []     # sorted list of active pivot prices
    active_queue: deque = deque()       # (confirm_bar, price) in ascending confirm_bar
    ptr = 0

    for i in range(n):
        # Activate pivots confirmed at or before bar i (causal — known by bar i close).
        while ptr < m and cb_list[ptr] <= i:
            pr = pr_list[ptr]
            bisect.insort(active_prices, pr)
            active_queue.append((cb_list[ptr], pr))
            ptr += 1
        # Expire pivots older than lookback.
        cutoff = i - lookback
        while active_queue and active_queue[0][0] <= cutoff:
            _, old_pr = active_queue.popleft()
            idx = bisect.bisect_left(active_prices, old_pr)
            if idx < len(active_prices) and active_prices[idx] == old_pr:
                active_prices.pop(idx)

        b = band[i]
        if not np.isnan(b) and active_prices:
            # Near low
            lo0 = bisect.bisect_left(active_prices, low[i] - b)
            lo1 = bisect.bisect_right(active_prices, low[i] + b)
            c_lo = lo1 - lo0
            react_low_count[i] = c_lo
            if c_lo > 0:
                react_low_level[i] = sum(active_prices[lo0:lo1]) / c_lo
            # Near high
            hi0 = bisect.bisect_left(active_prices, high[i] - b)
            hi1 = bisect.bisect_right(active_prices, high[i] + b)
            c_hi = hi1 - hi0
            react_high_count[i] = c_hi
            if c_hi > 0:
                react_high_level[i] = sum(active_prices[hi0:hi1]) / c_hi

    return pd.DataFrame({
        "react_low_count": react_low_count,
        "react_low_level": react_low_level,
        "react_high_count": react_high_count,
        "react_high_level": react_high_level,
    }, index=df.index)
