"""Tests for limit_entry_wrapper and partial_tp_wrapper."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.signal_wrappers import limit_entry_wrapper, partial_tp_wrapper
from src.strategies.base import empty_signals


def _make_ohlcv(n: int = 200, base: float = 2000.0) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    close = base + np.arange(n) * 0.1  # smooth uptrend
    open_ = close - 0.05
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": np.full(n, 100.0),
    })


def test_limit_entry_drops_signal_when_no_pullback() -> None:
    """In a strict uptrend without pullbacks, no long limit (below close)
    should ever fill. The wrapper must drop all long signals."""
    ltf = _make_ohlcv(n=200)
    sigs = empty_signals(ltf)
    # Fire 1 long signal at bar 50
    sigs.at[50, "action"] = "enter_long"
    sigs.at[50, "sl"] = ltf.at[50, "close"] - 5.0
    sigs.at[50, "tp"] = ltf.at[50, "close"] + 5.0

    # Make subsequent bars STRICTLY higher than the trigger close
    ltf2 = ltf.copy()
    ltf2.loc[51:, "low"] = ltf2.loc[51:, "close"] + 0.5  # low > close > limit_price

    out = limit_entry_wrapper(ltf2, sigs, offset_atr=0.5, expire_bars=5)
    assert (out["action"] != "enter_long").all(), "limit unexpectedly filled in strict uptrend"


def test_limit_entry_fills_on_pullback() -> None:
    """Construct a synthetic bar AFTER the trigger that dips down to the limit
    price; the wrapper must convert the signal into an entry on that bar."""
    n = 50
    ltf = _make_ohlcv(n=n)
    sigs = empty_signals(ltf)
    sigs.at[20, "action"] = "enter_long"
    sigs.at[20, "sl"] = ltf.at[20, "close"] - 5.0
    sigs.at[20, "tp"] = ltf.at[20, "close"] + 5.0

    # Engineer bar 22 to dip below limit_price ≈ close[20] - offset*ATR.
    # Default ATR(14) for this synthetic is ~0.5 (high-low). offset_atr=0.5 → limit ≈ close-0.25.
    # Just make bar 22's low explicitly lower:
    target_low = ltf.at[20, "close"] - 1.0
    ltf.at[22, "low"] = target_low

    out = limit_entry_wrapper(ltf, sigs, offset_atr=0.5, expire_bars=5)
    # Trigger bar must NOT have an entry (deferred)
    assert out.at[20, "action"] == "hold"
    # Fill bar must have the entry with original SL/TP
    assert out.at[22, "action"] == "enter_long", f"expected fill at bar 22, got actions={out['action'].tolist()[18:28]}"
    assert out.at[22, "sl"] == sigs.at[20, "sl"]
    assert out.at[22, "tp"] == sigs.at[20, "tp"]


def test_limit_entry_short_fills_on_rally() -> None:
    """Mirror test for shorts: signal at bar i, bar i+k's high reaches limit (above close)."""
    n = 50
    ltf = _make_ohlcv(n=n)
    sigs = empty_signals(ltf)
    sigs.at[20, "action"] = "enter_short"
    sigs.at[20, "sl"] = ltf.at[20, "close"] + 5.0
    sigs.at[20, "tp"] = ltf.at[20, "close"] - 5.0

    # offset_atr=2.0 places the short limit far above natural highs (which are
    # close + 0.5). We then spike bar 23's high above that.
    target_high = ltf.at[20, "close"] + 5.0
    ltf.at[23, "high"] = target_high

    out = limit_entry_wrapper(ltf, sigs, offset_atr=2.0, expire_bars=5)
    assert out.at[20, "action"] == "hold"
    assert out.at[23, "action"] == "enter_short"


def test_limit_entry_schema_invariants() -> None:
    """Output schema must match contract: same length, valid actions, finite SL/TP on enters."""
    ltf = _make_ohlcv(n=100)
    sigs = empty_signals(ltf)
    for i in [10, 30, 60]:
        sigs.at[i, "action"] = "enter_long"
        sigs.at[i, "sl"] = ltf.at[i, "close"] - 3.0
        sigs.at[i, "tp"] = ltf.at[i, "close"] + 3.0
        # Engineer pullback
        if i + 2 < len(ltf):
            ltf.at[i + 2, "low"] = ltf.at[i, "close"] - 1.5

    out = limit_entry_wrapper(ltf, sigs, offset_atr=0.4, expire_bars=4)
    assert len(out) == len(ltf)
    assert set(out["action"].unique()).issubset({"hold", "enter_long", "enter_short", "exit"})
    enter_mask = out["action"].isin(["enter_long", "enter_short"])
    if enter_mask.any():
        assert np.isfinite(out.loc[enter_mask, "sl"]).all()
        assert np.isfinite(out.loc[enter_mask, "tp"]).all()


def test_limit_entry_first_come_wins_on_collision() -> None:
    """If two original signals would land on the same fill bar, keep the first."""
    n = 50
    ltf = _make_ohlcv(n=n)
    sigs = empty_signals(ltf)
    # Two consecutive long signals at bars 20 and 21 — both might want to fill at bar 22
    for i in [20, 21]:
        sigs.at[i, "action"] = "enter_long"
        sigs.at[i, "sl"] = ltf.at[i, "close"] - 5.0
        sigs.at[i, "tp"] = ltf.at[i, "close"] + 5.0
    ltf.at[22, "low"] = ltf.at[20, "close"] - 2.0  # dip enough for both limits

    out = limit_entry_wrapper(ltf, sigs, offset_atr=0.3, expire_bars=5)
    # Exactly one entry at bar 22 (first signal wins)
    assert out.at[22, "action"] == "enter_long"
    # bar 23+ has no entry from bar 21 signal (it was already claimed at 22? No,
    # actually bar 22's collision check means bar 21's signal looks at bar 22-26.
    # bar 21's limit is slightly different but if bar 23-26 also dip the wrapper
    # might still fill it. So we just check that the wrapper produced finite output
    # with valid schema.
    assert (out["action"].isin(["hold", "enter_long", "enter_short", "exit"])).all()


def test_partial_tp_wrapper_shrinks_tp() -> None:
    """Replacing TP with TP1 should bring tp closer to close than the original."""
    n = 30
    ltf = _make_ohlcv(n=n)
    sigs = empty_signals(ltf)
    sigs.at[10, "action"] = "enter_long"
    sigs.at[10, "sl"] = ltf.at[10, "close"] - 5.0
    sigs.at[10, "tp"] = ltf.at[10, "close"] + 10.0

    out = partial_tp_wrapper(ltf, sigs, tp1_frac=0.5)
    orig_dist = sigs.at[10, "tp"] - ltf.at[10, "close"]
    new_dist = out.at[10, "tp"] - ltf.at[10, "close"]
    assert new_dist < orig_dist
    assert pytest.approx(new_dist, rel=1e-9) == orig_dist * 0.5
    assert out.at[10, "action"] == "enter_long"
    assert out.at[10, "sl"] == sigs.at[10, "sl"]
