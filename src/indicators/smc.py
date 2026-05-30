"""Smart Money Concepts (SMC) primitives.

Provides: swing high/low detection, BOS / CHoCH (break-of-structure / change-of-
character), order blocks (OB), fair value gaps (FVG), and liquidity sweeps.

All functions are lookahead-safe in the following sense: the value reported at
bar i is "what would have been confirmed by bar i's close" — swing pivots are
confirmed only AFTER `swing_lookforward` bars have closed (so a swing detected
at bar i becomes known at bar i + swing_lookforward), and the output series for
those primitives is shifted to bar `i + lookforward` so callers reading at bar
j see only swings confirmed by bar j.

This matches the parity ADR §1/§9: an indicator at bar j may not use any bar
> j to compute its value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Swing pivots (fractal-style)
# -----------------------------------------------------------------------------

def swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Detect swing highs/lows using `left` bars on each side.

    A bar at position p is a **swing high** iff `high[p]` is strictly greater
    than `high[p-left..p-1]` and strictly greater-than-or-equal to `high[p+1..p+right]`.
    Equivalent for swing low using `low`.

    **Lookahead handling:** A swing at position p requires `right` future bars
    to confirm. The output series therefore mark a swing at index `p + right`
    (the confirmation bar), not at p itself. A caller reading the result at
    bar j sees only swings whose `right`-bar confirmation has completed by j.

    Returns DataFrame with columns:
      - 'swing_high_price' : price level of the most recently confirmed swing high
                              (NaN until any swing confirmed); known as of the
                              CONFIRMATION bar.
      - 'swing_high_idx'   : integer position of that swing high.
      - 'swing_low_price', 'swing_low_idx' : same for swing lows.
      - 'new_swing_high'   : bool — True at the confirmation bar of a new swing high.
      - 'new_swing_low'    : bool — True at the confirmation bar of a new swing low.
    """
    if left <= 0 or right <= 0:
        raise ValueError("left and right must be > 0")
    required = {"high", "low"}
    if not required.issubset(df.columns):
        raise ValueError(f"swings requires columns {required}")

    n = len(df)
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")

    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)

    # Detect at position p (which requires p-left..p+right available).
    for p in range(left, n - right):
        h_left = high[p - left : p]
        h_right = high[p + 1 : p + right + 1]
        if high[p] > h_left.max() and high[p] >= h_right.max():
            is_swing_high[p] = True
        l_left = low[p - left : p]
        l_right = low[p + 1 : p + right + 1]
        if low[p] < l_left.min() and low[p] <= l_right.min():
            is_swing_low[p] = True

    # CONFIRMATION shift: a swing at position p is only known at p + right.
    # We record at the confirmation bar (p + right) but store the PIVOT's price
    # and original position so downstream code can reason about levels.
    new_high = np.zeros(n, dtype=bool)
    new_low = np.zeros(n, dtype=bool)
    high_price = np.full(n, np.nan)
    high_idx = np.full(n, -1, dtype=np.int64)
    low_price = np.full(n, np.nan)
    low_idx = np.full(n, -1, dtype=np.int64)

    last_high_price = np.nan
    last_high_idx = -1
    last_low_price = np.nan
    last_low_idx = -1

    for p in range(n):
        confirm_pos_for_high = p - right  # the swing position whose confirmation lands at p
        if 0 <= confirm_pos_for_high < n and is_swing_high[confirm_pos_for_high]:
            last_high_price = high[confirm_pos_for_high]
            last_high_idx = confirm_pos_for_high
            new_high[p] = True
        if 0 <= confirm_pos_for_high < n and is_swing_low[confirm_pos_for_high]:
            last_low_price = low[confirm_pos_for_high]
            last_low_idx = confirm_pos_for_high
            new_low[p] = True
        high_price[p] = last_high_price
        high_idx[p] = last_high_idx
        low_price[p] = last_low_price
        low_idx[p] = last_low_idx

    return pd.DataFrame({
        "swing_high_price": high_price,
        "swing_high_idx": high_idx,
        "swing_low_price": low_price,
        "swing_low_idx": low_idx,
        "new_swing_high": new_high,
        "new_swing_low": new_low,
    }, index=df.index)


# -----------------------------------------------------------------------------
# Break of Structure (BOS) / Change of Character (CHoCH)
# -----------------------------------------------------------------------------

def structure_breaks(df: pd.DataFrame, swings_df: pd.DataFrame) -> pd.DataFrame:
    """Detect BOS (continuation) and CHoCH (reversal) using close-through-swing rule.

    Convention: track current internal trend state {+1 up, -1 down, 0 unknown}.
    - In UPTREND: a CLOSE above the latest swing high = BOS (continuation up).
    - In UPTREND: a CLOSE below the latest swing low = CHoCH (reversal to down).
    - In DOWNTREND: a CLOSE below latest swing low = BOS (continuation down).
    - In DOWNTREND: a CLOSE above latest swing high = CHoCH (reversal to up).
    - From UNKNOWN: first close-through fixes the initial trend (treat as BOS).

    Inputs:
      df       — OHLCV with 'close'.
      swings_df — output of `swings()` (must share index).

    Returns DataFrame with columns:
      - 'trend'   : int (-1/0/+1) — current SMC trend state.
      - 'bos'     : bool — True at bar where a BOS just printed.
      - 'choch'   : bool — True at bar where a CHoCH just printed.
    """
    if "close" not in df.columns:
        raise ValueError("structure_breaks requires column 'close'")
    if len(df) != len(swings_df):
        raise ValueError("df and swings_df must have the same length")

    n = len(df)
    close = df["close"].to_numpy(dtype="float64")
    sh = swings_df["swing_high_price"].to_numpy(dtype="float64")
    sl = swings_df["swing_low_price"].to_numpy(dtype="float64")

    trend = np.zeros(n, dtype=np.int8)
    bos = np.zeros(n, dtype=bool)
    choch = np.zeros(n, dtype=bool)

    cur = 0  # -1, 0, +1
    for i in range(n):
        broke_up = (not np.isnan(sh[i])) and close[i] > sh[i]
        broke_down = (not np.isnan(sl[i])) and close[i] < sl[i]
        if cur == 0:
            if broke_up:
                cur = 1; bos[i] = True
            elif broke_down:
                cur = -1; bos[i] = True
        elif cur == 1:
            if broke_down:
                cur = -1; choch[i] = True
            elif broke_up:
                bos[i] = True
        elif cur == -1:
            if broke_up:
                cur = 1; choch[i] = True
            elif broke_down:
                bos[i] = True
        trend[i] = cur

    return pd.DataFrame({"trend": trend, "bos": bos, "choch": choch}, index=df.index)


# -----------------------------------------------------------------------------
# Fair Value Gaps (FVG)
# -----------------------------------------------------------------------------

def fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Detect 3-bar fair value gaps (ICT definition).

    Bullish FVG at bar i: `low[i] > high[i-2]`. The gap is between `high[i-2]`
    (top of FVG = lower edge of remaining gap?) — by ICT convention the gap zone
    is [high[i-2], low[i]] (bullish) or [high[i], low[i-2]] (bearish).

    **Confirmation timing:** The middle bar is i-1, so the FVG is detectable at
    bar i's close. We report at bar i.

    Returns DataFrame with columns:
      - 'bull_fvg'       : bool — True at bar i where a bullish FVG just printed.
      - 'bull_fvg_top'   : top of the most recent bullish FVG (NaN if none).
      - 'bull_fvg_bot'   : bottom of the most recent bullish FVG.
      - 'bear_fvg', 'bear_fvg_top', 'bear_fvg_bot' : mirrored.
    """
    if not {"high", "low"}.issubset(df.columns):
        raise ValueError("fair_value_gaps requires columns ['high','low']")
    n = len(df)
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")

    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    bull_top = np.full(n, np.nan)
    bull_bot = np.full(n, np.nan)
    bear_top = np.full(n, np.nan)
    bear_bot = np.full(n, np.nan)

    last_bull_top = last_bull_bot = np.nan
    last_bear_top = last_bear_bot = np.nan
    for i in range(2, n):
        if low[i] > high[i - 2]:  # bullish FVG
            bull[i] = True
            last_bull_bot = high[i - 2]
            last_bull_top = low[i]
        if high[i] < low[i - 2]:  # bearish FVG
            bear[i] = True
            last_bear_bot = high[i]
            last_bear_top = low[i - 2]
        bull_top[i] = last_bull_top
        bull_bot[i] = last_bull_bot
        bear_top[i] = last_bear_top
        bear_bot[i] = last_bear_bot

    return pd.DataFrame({
        "bull_fvg": bull, "bull_fvg_top": bull_top, "bull_fvg_bot": bull_bot,
        "bear_fvg": bear, "bear_fvg_top": bear_top, "bear_fvg_bot": bear_bot,
    }, index=df.index)


# -----------------------------------------------------------------------------
# Order Blocks (last opposing candle before BOS)
# -----------------------------------------------------------------------------

def order_blocks(df: pd.DataFrame, structure: pd.DataFrame) -> pd.DataFrame:
    """Detect bullish/bearish order blocks at BOS events.

    Bullish OB: last DOWN candle before a bullish BOS. OB zone = that down candle's
    [low, high]. Detected at the BOS bar (so the OB level is "known" then).

    Bearish OB: last UP candle before a bearish BOS.

    Inputs:
      df         — OHLCV with 'open','close','high','low'.
      structure  — output of `structure_breaks` (shares index).

    Returns DataFrame:
      - 'bull_ob_top', 'bull_ob_bot' : most recent bullish OB zone (NaN until any).
      - 'bear_ob_top', 'bear_ob_bot' : same for bearish.
      - 'bull_ob_idx', 'bear_ob_idx' : positional index of the OB origin bar.
    """
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"order_blocks requires columns {required}")
    if len(df) != len(structure):
        raise ValueError("df and structure must have same length")

    n = len(df)
    open_ = df["open"].to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    trend = structure["trend"].to_numpy(dtype=np.int8)
    bos = structure["bos"].to_numpy(dtype=bool)

    bull_top = np.full(n, np.nan)
    bull_bot = np.full(n, np.nan)
    bull_idx = np.full(n, -1, dtype=np.int64)
    bear_top = np.full(n, np.nan)
    bear_bot = np.full(n, np.nan)
    bear_idx = np.full(n, -1, dtype=np.int64)

    last_bull_top = last_bull_bot = np.nan
    last_bull_idx = -1
    last_bear_top = last_bear_bot = np.nan
    last_bear_idx = -1

    for i in range(n):
        if bos[i] and trend[i] == 1:
            # Find last DOWN candle BEFORE i
            for j in range(i - 1, -1, -1):
                if close[j] < open_[j]:
                    last_bull_top = high[j]
                    last_bull_bot = low[j]
                    last_bull_idx = j
                    break
        if bos[i] and trend[i] == -1:
            for j in range(i - 1, -1, -1):
                if close[j] > open_[j]:
                    last_bear_top = high[j]
                    last_bear_bot = low[j]
                    last_bear_idx = j
                    break
        bull_top[i] = last_bull_top
        bull_bot[i] = last_bull_bot
        bull_idx[i] = last_bull_idx
        bear_top[i] = last_bear_top
        bear_bot[i] = last_bear_bot
        bear_idx[i] = last_bear_idx

    return pd.DataFrame({
        "bull_ob_top": bull_top, "bull_ob_bot": bull_bot, "bull_ob_idx": bull_idx,
        "bear_ob_top": bear_top, "bear_ob_bot": bear_bot, "bear_ob_idx": bear_idx,
    }, index=df.index)


# -----------------------------------------------------------------------------
# Liquidity sweeps
# -----------------------------------------------------------------------------

def liquidity_sweeps(df: pd.DataFrame, swings_df: pd.DataFrame) -> pd.DataFrame:
    """Detect liquidity sweeps: wick past a swing extreme followed by a close back inside.

    Bullish sweep (sells liquidity grabbed): low[i] < swing_low_price[i] (taken
    out the recent swing low intra-bar) BUT close[i] > swing_low_price[i] (closed
    back above it). Used as a long reversal cue.

    Bearish sweep: mirror — high above swing high, close back below.

    Returns DataFrame:
      - 'bull_sweep' : bool
      - 'bear_sweep' : bool
    """
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("liquidity_sweeps requires ['high','low','close']")

    n = len(df)
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")
    sh = swings_df["swing_high_price"].to_numpy(dtype="float64")
    sl = swings_df["swing_low_price"].to_numpy(dtype="float64")

    bull = (~np.isnan(sl)) & (low < sl) & (close > sl)
    bear = (~np.isnan(sh)) & (high > sh) & (close < sh)
    return pd.DataFrame({"bull_sweep": bull, "bear_sweep": bear}, index=df.index)
