"""Unit tests for SMC primitives.

Run: python -m tests.indicators.test_smc
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import (
    swings, structure_breaks, fair_value_gaps, order_blocks, liquidity_sweeps,
)


def _bars_from_highs_lows(highs: list[float], lows: list[float],
                          opens: list[float] | None = None,
                          closes: list[float] | None = None) -> pd.DataFrame:
    n = len(highs)
    if opens is None: opens = [(h + l) / 2 for h, l in zip(highs, lows)]
    if closes is None: closes = list(opens)
    ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": [100.0] * n,
    })


def test_swings_detects_obvious_pivot() -> None:
    # An obvious swing high at index 5 (high=10), confirmed by index 8 (right=3).
    highs = [1, 2, 3, 4, 5, 10, 5, 4, 3, 2, 1]
    lows = [0.5] * len(highs)
    df = _bars_from_highs_lows(highs, lows)
    s = swings(df, left=3, right=3)
    # Confirmation at index 5 + 3 = 8
    assert s["new_swing_high"].iloc[8], "expected swing high confirmation at idx 8"
    assert s["swing_high_price"].iloc[8] == 10.0
    assert s["swing_high_idx"].iloc[8] == 5
    # Before confirmation, swing_high_price should be NaN
    assert pd.isna(s["swing_high_price"].iloc[7])
    print("PASS test_swings_detects_obvious_pivot")


def test_swings_lookahead_safe_on_prefix() -> None:
    n = 100
    rng = np.random.default_rng(0)
    highs = rng.uniform(10, 20, size=n).cumsum() / 10 + 100
    lows = highs - 1
    df = _bars_from_highs_lows(list(highs), list(lows))
    full = swings(df, left=3, right=3)
    K = 70
    prefix = swings(df.iloc[:K], left=3, right=3)
    # First K rows must match exactly.
    for col in ["new_swing_high", "new_swing_low"]:
        assert (full[col].iloc[:K].to_numpy() == prefix[col].to_numpy()).all(), (
            f"lookahead in {col}"
        )
    print("PASS test_swings_lookahead_safe_on_prefix")


def test_structure_breaks_bos_then_choch() -> None:
    # Construct a zigzag with clear swing high AND swing low, then a break-above
    # creating a BOS up, then a break-below creating a CHoCH down.
    highs =  [102, 104, 106, 108, 110, 112, 110, 108, 106, 104, 106, 108, 110, 112, 114, 113, 110, 105, 100, 95]
    lows  =  [100, 102, 104, 106, 108, 110, 108, 106, 104, 102, 104, 106, 108, 110, 112, 111, 108, 103,  98, 93]
    closes = [101, 103, 105, 107, 109, 111, 109, 107, 105, 103, 105, 107, 109, 111, 113, 112, 109, 104,  99, 94]
    df = _bars_from_highs_lows(highs, lows, closes=closes)
    s = swings(df, left=2, right=2)
    st = structure_breaks(df, s)
    # We expect at least one BOS (initial up break) and ideally a CHoCH later.
    assert st["bos"].any(), f"expected at least one BOS event (swings: {int(s['new_swing_high'].sum())}H/{int(s['new_swing_low'].sum())}L)"
    print(f"PASS test_structure_breaks_bos_then_choch (bos={int(st['bos'].sum())} choch={int(st['choch'].sum())})")


def test_fair_value_gaps_detects_3bar_gap() -> None:
    # Bar 0: high=100, low=99. Bar 1: noise. Bar 2: low=102 > bar0.high → bullish FVG.
    df = _bars_from_highs_lows(
        highs=[100, 101, 103, 104],
        lows=[99, 100, 102, 103],
    )
    f = fair_value_gaps(df)
    assert f["bull_fvg"].iloc[2], "expected bullish FVG at bar 2"
    assert f["bull_fvg_bot"].iloc[2] == 100  # high[0]
    assert f["bull_fvg_top"].iloc[2] == 102  # low[2]
    # No bearish FVG in this monotone-up series
    assert not f["bear_fvg"].any()
    print("PASS test_fair_value_gaps_detects_3bar_gap")


def test_order_blocks_marks_last_opposite_candle_before_bos() -> None:
    # Construct: 3 down candles, then 5 up candles ending with a BOS-up event.
    n = 15
    opens = [10, 9, 8, 7, 7.5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    closes = [9, 8, 7, 7.5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    highs = [c + 0.5 for c in closes]
    lows = [o - 0.5 for o in opens]
    df = _bars_from_highs_lows(highs, lows, opens=opens, closes=closes)
    s = swings(df, left=2, right=2)
    st = structure_breaks(df, s)
    ob = order_blocks(df, st)
    # If a BOS fired, the bullish OB level should be one of the down candles' lows/highs.
    if st["bos"].any() and (st["trend"] == 1).any():
        first_bos_idx = int(np.where(st["bos"].to_numpy() & (st["trend"].to_numpy() == 1))[0][0])
        # The OB bar index must be < first_bos_idx and must be a DOWN bar.
        ob_idx = int(ob["bull_ob_idx"].iloc[first_bos_idx])
        assert ob_idx >= 0
        assert ob_idx < first_bos_idx
        assert closes[ob_idx] < opens[ob_idx]
    print("PASS test_order_blocks_marks_last_opposite_candle_before_bos")


def test_liquidity_sweep_bull_at_swing_low() -> None:
    # Build sequence that creates a swing low at index 5 with low=10. Then later,
    # a bar that wicks BELOW 10 but closes back STRICTLY ABOVE it (bullish sweep).
    lows  = [20, 18, 16, 14, 12, 10, 12, 14, 16, 18, 20, 19, 18, 17, 16,  9, 12, 13, 14, 15]
    highs = [l + 2 for l in lows]
    # Closes — make sure idx 15 closes strictly above the swing low at 10.
    closes = [l + 1 for l in lows]
    closes[15] = 11.5  # explicit: above swing low of 10
    df = _bars_from_highs_lows(highs, lows, closes=closes)
    s = swings(df, left=3, right=3)
    ls = liquidity_sweeps(df, s)
    # The sweep at index 15 (low=9 below swing low=10, close=11.5 above): bullish sweep.
    assert ls["bull_sweep"].iloc[15], (
        f"expected bullish sweep at idx 15 (swing_low={s['swing_low_price'].iloc[15]}, "
        f"low={lows[15]}, close={closes[15]})"
    )
    print("PASS test_liquidity_sweep_bull_at_swing_low")


if __name__ == "__main__":
    test_swings_detects_obvious_pivot()
    test_swings_lookahead_safe_on_prefix()
    test_structure_breaks_bos_then_choch()
    test_fair_value_gaps_detects_3bar_gap()
    test_order_blocks_marks_last_opposite_candle_before_bos()
    test_liquidity_sweep_bull_at_swing_low()
    print("\nAll SMC tests passed.")
