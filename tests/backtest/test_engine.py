"""Backtest engine tests — every parity-ADR fill rule has a dedicated case.

Each test constructs hand-built OHLC and signal data with known outcomes, then
asserts the engine produced exactly the expected entry_price, exit_price,
exit_reason, and PnL. Float comparisons use small tolerances (1e-9) because the
math is exact arithmetic over float64 without compounding error in these short
sequences.

Run: python -m tests.backtest.test_engine
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.backtest.fills import (
    entry_fill_price,
    round_sl_tp,
    validate_sltp_after_entry,
    evaluate_sl_tp_on_bar,
    market_exit_price,
    commission,
)
from src.backtest.sizing import position_size
from src.config import SYMBOLS


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _bars(ohlc: list[tuple[float, float, float, float]], start: str = "2025-01-01 00:00", freq: str = "5min") -> pd.DataFrame:
    """Build a tz-aware OHLCV frame from a list of (open, high, low, close) tuples.

    Volume is set to a constant 100; trades column omitted (engine doesn't need it).
    """
    n = len(ohlc)
    ts = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    rows = []
    for i, (o, h, l, c) in enumerate(ohlc):
        # Defend the OHLC invariant
        h = max(h, o, c)
        l = min(l, o, c)
        rows.append((ts[i], o, h, l, c, 100.0))
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _signals(actions: list[str], sls: list[float], tps: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "action": actions,
        "sl": sls,
        "tp": tps,
    })


# -----------------------------------------------------------------------------
# Unit tests for fills.py
# -----------------------------------------------------------------------------

def test_entry_fill_long_pays_more_short_pays_less() -> None:
    pip = 1.0
    long_price = entry_fill_price(bar_open=100.0, side=1, spread_pips=1.0, slippage_pips=1.0, pip=pip)
    short_price = entry_fill_price(bar_open=100.0, side=-1, spread_pips=1.0, slippage_pips=1.0, pip=pip)
    assert long_price == 102.0
    assert short_price == 98.0
    print("PASS test_entry_fill_long_pays_more_short_pays_less")


def test_round_sl_tp_long_rounds_down_short_rounds_up() -> None:
    sl_l, tp_l = round_sl_tp(side=1, sl=99.07, tp=101.93, min_tick=0.1)
    assert sl_l == 99.0
    assert tp_l == 101.9
    sl_s, tp_s = round_sl_tp(side=-1, sl=101.07, tp=99.93, min_tick=0.1)
    # short: both round UP
    # 101.07 -> 101.1 ; 99.93 -> 100.0
    assert abs(sl_s - 101.1) < 1e-9
    assert abs(tp_s - 100.0) < 1e-9
    print("PASS test_round_sl_tp_long_rounds_down_short_rounds_up")


def test_validate_sltp() -> None:
    assert validate_sltp_after_entry(1, entry_price=100.0, sl=99.0, tp=101.0) is True
    assert validate_sltp_after_entry(1, entry_price=100.0, sl=100.5, tp=101.0) is False  # SL > entry
    assert validate_sltp_after_entry(-1, entry_price=100.0, sl=101.0, tp=99.0) is True
    assert validate_sltp_after_entry(-1, entry_price=100.0, sl=100.0, tp=99.0) is False  # SL == entry
    print("PASS test_validate_sltp")


def test_eval_sltp_long_only_tp_hit() -> None:
    # Long: SL=99, TP=102. Bar open 100, high 102.5, low 99.5 → TP hit at 102.
    fill = evaluate_sl_tp_on_bar(
        side=1, bar_open=100.0, bar_high=102.5, bar_low=99.5,
        sl=99.0, tp=102.0, slippage_pips=1.0, pip=0.1,
    )
    assert fill.reason == "tp"
    assert fill.price == 102.0
    print("PASS test_eval_sltp_long_only_tp_hit")


def test_eval_sltp_long_only_sl_hit_with_slippage() -> None:
    # Long: SL=99, TP=102. Bar open 100, high 100.5, low 98.5 → SL hit at 99 - slip.
    # slip = slippage_pips * pip = 1.0 * 0.1 = 0.1
    fill = evaluate_sl_tp_on_bar(
        side=1, bar_open=100.0, bar_high=100.5, bar_low=98.5,
        sl=99.0, tp=102.0, slippage_pips=1.0, pip=0.1,
    )
    assert fill.reason == "sl"
    assert abs(fill.price - 98.9) < 1e-9
    print("PASS test_eval_sltp_long_only_sl_hit_with_slippage")


def test_eval_sltp_long_both_in_range_sl_first() -> None:
    # Long: SL=99, TP=102. Bar open 100.5, high 102.5, low 98.5 → both touched → SL first.
    fill = evaluate_sl_tp_on_bar(
        side=1, bar_open=100.5, bar_high=102.5, bar_low=98.5,
        sl=99.0, tp=102.0, slippage_pips=1.0, pip=0.1,
    )
    assert fill.reason == "sl"
    assert abs(fill.price - 98.9) < 1e-9  # 99 - 0.1
    print("PASS test_eval_sltp_long_both_in_range_sl_first")


def test_eval_sltp_long_adverse_open_gap() -> None:
    # Long position, bar opens BELOW the stop → fill at open - slip, not at SL.
    fill = evaluate_sl_tp_on_bar(
        side=1, bar_open=97.0, bar_high=98.0, bar_low=96.5,
        sl=99.0, tp=102.0, slippage_pips=1.0, pip=0.1,
    )
    assert fill.reason == "sl"
    assert abs(fill.price - (97.0 - 0.1)) < 1e-9
    print("PASS test_eval_sltp_long_adverse_open_gap")


def test_eval_sltp_long_favorable_open_gap() -> None:
    # Long position, bar opens ABOVE the TP → fill at TP exactly (no improvement).
    fill = evaluate_sl_tp_on_bar(
        side=1, bar_open=103.0, bar_high=104.0, bar_low=102.5,
        sl=99.0, tp=102.0, slippage_pips=1.0, pip=0.1,
    )
    assert fill.reason == "tp"
    assert fill.price == 102.0
    print("PASS test_eval_sltp_long_favorable_open_gap")


def test_eval_sltp_short_symmetric() -> None:
    # Short: SL=101, TP=98. Bar open 100, high 100.5, low 97.5 → TP hit at 98.
    fill = evaluate_sl_tp_on_bar(
        side=-1, bar_open=100.0, bar_high=100.5, bar_low=97.5,
        sl=101.0, tp=98.0, slippage_pips=1.0, pip=0.1,
    )
    assert fill.reason == "tp"
    assert fill.price == 98.0
    # SL hit: bar open 100, high 101.5, low 99.5 → SL at 101 + slip
    fill2 = evaluate_sl_tp_on_bar(
        side=-1, bar_open=100.0, bar_high=101.5, bar_low=99.5,
        sl=101.0, tp=98.0, slippage_pips=1.0, pip=0.1,
    )
    assert fill2.reason == "sl"
    assert abs(fill2.price - 101.1) < 1e-9
    # Short adverse open gap (gap UP past SL): bar open 102 → fill at 102 + slip
    fill3 = evaluate_sl_tp_on_bar(
        side=-1, bar_open=102.0, bar_high=102.5, bar_low=101.5,
        sl=101.0, tp=98.0, slippage_pips=1.0, pip=0.1,
    )
    assert fill3.reason == "sl"
    assert abs(fill3.price - 102.1) < 1e-9
    print("PASS test_eval_sltp_short_symmetric")


def test_position_size_basic_and_skip_below_min() -> None:
    # XAU-like: pip=0.1, qty_step=0.01, min_qty=0.01, contract_multiplier=1
    qty = position_size(
        equity=10_000.0, risk_pct=0.005,
        entry_price=2000.0, sl_price=1995.0,  # 5.0 USD distance
        contract_multiplier=1.0, qty_step=0.01, min_qty=0.01,
    )
    # risk = 50, distance = 5, qty = 10 → rounds to 10.00
    assert abs(qty - 10.00) < 1e-9

    # Below min: tiny equity, tight risk
    qty_zero = position_size(
        equity=1.0, risk_pct=0.001,
        entry_price=2000.0, sl_price=1990.0,
        contract_multiplier=1.0, qty_step=0.01, min_qty=0.01,
    )
    # raw_qty = 0.001 / 10 = 0.0001, rounds to 0.00 < min_qty → 0.0
    assert qty_zero == 0.0
    print("PASS test_position_size_basic_and_skip_below_min")


# -----------------------------------------------------------------------------
# Integration tests: engine end-to-end
# -----------------------------------------------------------------------------

def test_engine_long_tp_hit_simple() -> None:
    """One long entry, TP hit two bars later. Verify entry/exit prices and PnL sign."""
    # BTC-like symbol so pip=1, spread=1, slip=1 → entry markup = 2 USD
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),   # bar 0
        (100.0, 100.5, 99.5, 100.0),   # bar 1 — entry happens here (signal at bar 0 close)
        (100.0, 105.0, 99.5, 104.0),   # bar 2 — TP=105 NOT hit (high is 105 not 105.1) — fix below
        (104.0, 110.0, 103.0, 109.0),  # bar 3 — TP hit
    ]
    df = _bars(ohlc)
    # Strategy: at bar 0 close, enter long. SL=95, TP=110.
    sig = _signals(
        ["enter_long", "hold", "hold", "hold"],
        [95.0, np.nan, np.nan, np.nan],
        [110.0, np.nan, np.nan, np.nan],
    )
    res = run_backtest(df, sig, symbol="BTCUSDT", ltf_tf="M5", params={"initial_equity": 10_000.0, "risk_pct": 0.005})
    trades = res["trades"]
    assert len(trades) == 1
    t = trades.iloc[0]
    # Entry at bar 1's open=100, side=1, +2 (spread+slip) → 102
    assert abs(t["entry_price"] - 102.0) < 1e-9
    # Exit: bar 3 high=110 ≥ TP=110, low=103 > SL=95 → TP fill at 110
    assert t["exit_reason"] == "tp"
    assert abs(t["exit_price"] - 110.0) < 1e-9
    # PnL: side=1, qty * (110 - 102) - 2 * commission
    # qty = 10_000 * 0.005 / (102 - 95) = 50/7 ≈ 7.1428... — rounds DOWN to qty_step 0.00001
    qty = t["qty"]
    expected_qty_raw = 50.0 / 7.0
    assert qty <= expected_qty_raw
    assert qty > 0
    # Commission on BTC: 0.04% on notional
    assert t["entry_commission"] > 0
    assert t["exit_commission"] > 0
    assert t["pnl"] > 0
    print(f"PASS test_engine_long_tp_hit_simple — pnl={t['pnl']:.2f} qty={qty:.6f}")


def test_engine_short_tp_hit_symmetric() -> None:
    """Short entry, TP hit. Mirror of long test."""
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 89.0, 90.0),   # bar 2: SHORT TP at 90 → hit
    ]
    df = _bars(ohlc)
    sig = _signals(
        ["enter_short", "hold", "hold"],
        [105.0, np.nan, np.nan],
        [90.0, np.nan, np.nan],
    )
    res = run_backtest(df, sig, symbol="BTCUSDT", ltf_tf="M5")
    trades = res["trades"]
    assert len(trades) == 1
    t = trades.iloc[0]
    # Short entry: 100 - 2 = 98
    assert abs(t["entry_price"] - 98.0) < 1e-9
    # TP at 90
    assert t["exit_reason"] == "tp"
    assert abs(t["exit_price"] - 90.0) < 1e-9
    assert t["pnl"] > 0
    assert t["side"] == -1
    print(f"PASS test_engine_short_tp_hit_symmetric — pnl={t['pnl']:.2f}")


def test_engine_sl_first_on_ambiguous_bar() -> None:
    """Both SL and TP within bar [low,high]. SL must fill first (pessimistic)."""
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 115.0, 90.0, 100.0),  # both SL=95 and TP=110 within range
    ]
    df = _bars(ohlc)
    sig = _signals(
        ["enter_long", "hold", "hold"],
        [95.0, np.nan, np.nan],
        [110.0, np.nan, np.nan],
    )
    res = run_backtest(df, sig, symbol="BTCUSDT", ltf_tf="M5")
    t = res["trades"].iloc[0]
    assert t["exit_reason"] == "sl"
    # SL = 95, slip = 1 pip * 1 pip-size = 1 → exit at 94
    assert abs(t["exit_price"] - 94.0) < 1e-9
    assert t["pnl"] < 0  # loser
    print(f"PASS test_engine_sl_first_on_ambiguous_bar — pnl={t['pnl']:.2f}")


def test_engine_adverse_open_gap_long() -> None:
    """Long position; next bar opens below SL → fill at open-slip, not at SL."""
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),  # entry bar
        (90.0, 91.0, 88.0, 89.0),     # gap down through SL=95
    ]
    df = _bars(ohlc)
    sig = _signals(
        ["enter_long", "hold", "hold"],
        [95.0, np.nan, np.nan],
        [110.0, np.nan, np.nan],
    )
    res = run_backtest(df, sig, symbol="BTCUSDT", ltf_tf="M5")
    t = res["trades"].iloc[0]
    assert t["exit_reason"] == "sl"
    # bar 2 open = 90, slip = 1 → exit at 89
    assert abs(t["exit_price"] - 89.0) < 1e-9
    assert t["pnl"] < 0
    print(f"PASS test_engine_adverse_open_gap_long — pnl={t['pnl']:.2f}")


def test_engine_same_bar_entry_and_sl() -> None:
    """Entry bar itself contains SL → trade closes on the same bar (§3.2)."""
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 94.0, 95.5),  # entry bar; low=94 < SL=95 → SL same bar
    ]
    df = _bars(ohlc)
    sig = _signals(
        ["enter_long", "hold"],
        [95.0, np.nan],
        [110.0, np.nan],
    )
    res = run_backtest(df, sig, symbol="BTCUSDT", ltf_tf="M5")
    t = res["trades"].iloc[0]
    assert t["exit_reason"] == "sl"
    assert t["bars_held"] == 0  # same bar
    print(f"PASS test_engine_same_bar_entry_and_sl — bars_held=0")


def test_engine_manual_exit_before_intrabar_sltp() -> None:
    """Strategy emits 'exit' — must fill at next bar open BEFORE intra-bar SL/TP eval."""
    # Setup: long entered bar 1. Bar 2 has SL hit intra-bar BUT strategy emits 'exit' at bar 1 close.
    # Per §3.7, manual exit takes precedence → exit at bar 2 open, not SL.
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),   # entry bar
        (100.0, 100.5, 90.0, 95.0),    # SL=95 IS within range, but manual exit fires first
    ]
    df = _bars(ohlc)
    sig = _signals(
        ["enter_long", "exit", "hold"],
        [95.0, np.nan, np.nan],
        [110.0, np.nan, np.nan],
    )
    res = run_backtest(df, sig, symbol="BTCUSDT", ltf_tf="M5")
    t = res["trades"].iloc[0]
    assert t["exit_reason"] == "manual"
    # Manual exit price: long → open - slip = 100 - 1 = 99
    assert abs(t["exit_price"] - 99.0) < 1e-9
    print(f"PASS test_engine_manual_exit_before_intrabar_sltp — exit_price={t['exit_price']}")


def test_engine_stale_signal_skipped_after_gap() -> None:
    """Signal at bar i-1 close is stale if bar i is not the expected next bar."""
    # Build LTF M5 series with a missing bar (e.g. 0:00, 0:05, then jumps to 0:15)
    ts = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:05", "2025-01-01 00:15"], utc=True)
    df = pd.DataFrame({
        "timestamp": ts,
        "open": [100.0, 100.0, 100.0],
        "high": [100.5, 100.5, 110.0],
        "low": [99.5, 99.5, 99.5],
        "close": [100.0, 100.0, 100.0],
        "volume": [100.0, 100.0, 100.0],
    })
    # Strategy enters long at bar 1 close. Bar 2 is AFTER a gap → stale, no entry.
    sig = _signals(
        ["hold", "enter_long", "hold"],
        [np.nan, 95.0, np.nan],
        [np.nan, 110.0, np.nan],
    )
    res = run_backtest(df, sig, symbol="BTCUSDT", ltf_tf="M5")
    assert len(res["trades"]) == 0
    assert res["meta"]["stale_signal_bars"] >= 1
    print("PASS test_engine_stale_signal_skipped_after_gap")


def test_engine_force_close_at_eod() -> None:
    """Open position at last bar is force-closed (no resting state across runs)."""
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),  # entry
        (100.0, 100.5, 99.5, 100.0),  # no SL/TP hit; last bar
    ]
    df = _bars(ohlc)
    sig = _signals(
        ["enter_long", "hold", "hold"],
        [95.0, np.nan, np.nan],
        [110.0, np.nan, np.nan],
    )
    res = run_backtest(df, sig, symbol="BTCUSDT", ltf_tf="M5")
    t = res["trades"].iloc[0]
    assert t["exit_reason"] == "force_eod"
    print(f"PASS test_engine_force_close_at_eod — exit_price={t['exit_price']}")


def test_engine_no_entry_after_stop_when_signal_was_emitted_during_position() -> None:
    """Regression: an enter_* signal emitted while a position is open MUST be ignored,
    even if the old position closes via SL/TP during the next bar (ADR §6)."""
    # Bar 0: hold (just to set baseline).
    # Bar 1: enter_long (signal at bar 0 close); position opens at bar 1 open=100, +2 → 102.
    # Bar 2: strategy emits enter_long AGAIN while position is still open (this is the
    #        ignored signal). At bar 2's close, position is still open.
    # Bar 3: bar opens normally but high gaps to 95→ SL at 95.
    #        The signal from bar 2 (enter_long) had been emitted while position was open
    #        → engine MUST ignore it even though the old position closes during bar 3.
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),  # entry bar
        (100.0, 100.5, 99.5, 100.0),  # position still open; ignored "enter_long" signal here
        (100.0, 100.5, 90.0, 91.0),   # SL hit; ALSO carries the ignored entry signal
    ]
    df = _bars(ohlc)
    sig = _signals(
        ["hold", "enter_long", "enter_long", "hold"],
        [np.nan, 95.0, 95.0, np.nan],
        [np.nan, 110.0, 110.0, np.nan],
    )
    res = run_backtest(df, sig, symbol="BTCUSDT", ltf_tf="M5")
    trades = res["trades"]
    # Exactly ONE trade — the original. The ignored re-entry must NOT produce a second.
    assert len(trades) == 1, f"expected 1 trade, got {len(trades)} (ignored-signal bug)"
    t = trades.iloc[0]
    assert t["exit_reason"] == "sl"
    print("PASS test_engine_no_entry_after_stop_when_signal_was_emitted_during_position")


def test_engine_entry_price_rounded_to_tick() -> None:
    """Entry price must be rounded to min_tick after spread+slip — long rounds UP, short DOWN."""
    # XAU: pip=0.1, spread=2.0 pips, slip=1.0 pips → markup = 0.3
    # If bar_open is exactly on a 0.01 tick, entry = open + 0.3 (already on tick).
    # If bar_open = 2000.005 (sub-tick — unlikely in practice but the test must still pass),
    # entry_raw = 2000.305; with min_tick=0.01, long rounds UP to 2000.31.
    # We use BTC for simpler arithmetic: pip=1, spread=1, slip=1 → markup=2, min_tick=0.01.
    # Construct bar_open with sub-tick fractional part to exercise rounding.
    ohlc = [
        (100.005, 100.5, 99.5, 100.005),
        (100.005, 100.5, 99.5, 100.005),
        (100.005, 102.5, 99.5, 102.0),
    ]
    df = _bars(ohlc)
    sig = _signals(
        ["enter_long", "hold", "hold"],
        [95.0, np.nan, np.nan],
        [102.5, np.nan, np.nan],
    )
    res = run_backtest(df, sig, symbol="BTCUSDT", ltf_tf="M5")
    t = res["trades"].iloc[0]
    # bar 1 open = 100.005, side=+1, markup=2 → raw = 102.005, round UP to 0.01 tick → 102.01
    assert abs(t["entry_price"] - 102.01) < 1e-9, f"entry_price={t['entry_price']}"
    print("PASS test_engine_entry_price_rounded_to_tick")


def test_engine_rejects_non_utc_tz() -> None:
    """ADR §1 requires UTC; non-UTC tz-aware timestamps must be rejected."""
    ts_ny = pd.date_range("2025-01-01", periods=3, freq="5min", tz="America/New_York")
    df = pd.DataFrame({
        "timestamp": ts_ny, "open": [100.0]*3, "high": [100.5]*3,
        "low": [99.5]*3, "close": [100.0]*3, "volume": [100.0]*3,
    })
    sig = _signals(["hold"]*3, [np.nan]*3, [np.nan]*3)
    try:
        run_backtest(df, sig, symbol="BTCUSDT", ltf_tf="M5")
    except ValueError as e:
        assert "UTC" in str(e)
        print("PASS test_engine_rejects_non_utc_tz")
        return
    raise AssertionError("engine should have rejected non-UTC timestamps")


def test_engine_no_pyramiding() -> None:
    """Second enter_long while position is open → ignored, only one trade."""
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),  # entry
        (100.0, 100.5, 99.5, 100.0),  # ignored second-entry signal here
        (100.0, 115.0, 99.5, 110.0),  # TP hit
    ]
    df = _bars(ohlc)
    sig = _signals(
        ["enter_long", "enter_long", "hold", "hold"],
        [95.0, 95.0, np.nan, np.nan],
        [110.0, 110.0, np.nan, np.nan],
    )
    res = run_backtest(df, sig, symbol="BTCUSDT", ltf_tf="M5")
    assert len(res["trades"]) == 1
    print("PASS test_engine_no_pyramiding")


def test_engine_parity_doc_hash_in_meta() -> None:
    """Manifest records the parity-doc hash — gates the live bot from running on stale spec."""
    ohlc = [(100.0, 100.5, 99.5, 100.0)] * 3
    df = _bars(ohlc)
    sig = _signals(["hold"] * 3, [np.nan] * 3, [np.nan] * 3)
    res = run_backtest(df, sig, symbol="BTCUSDT", ltf_tf="M5")
    h = res["meta"]["parity_doc_sha256"]
    assert isinstance(h, str) and len(h) == 64
    print(f"PASS test_engine_parity_doc_hash_in_meta — hash={h[:16]}...")


if __name__ == "__main__":
    test_entry_fill_long_pays_more_short_pays_less()
    test_round_sl_tp_long_rounds_down_short_rounds_up()
    test_validate_sltp()
    test_eval_sltp_long_only_tp_hit()
    test_eval_sltp_long_only_sl_hit_with_slippage()
    test_eval_sltp_long_both_in_range_sl_first()
    test_eval_sltp_long_adverse_open_gap()
    test_eval_sltp_long_favorable_open_gap()
    test_eval_sltp_short_symmetric()
    test_position_size_basic_and_skip_below_min()
    test_engine_long_tp_hit_simple()
    test_engine_short_tp_hit_symmetric()
    test_engine_sl_first_on_ambiguous_bar()
    test_engine_adverse_open_gap_long()
    test_engine_same_bar_entry_and_sl()
    test_engine_manual_exit_before_intrabar_sltp()
    test_engine_stale_signal_skipped_after_gap()
    test_engine_force_close_at_eod()
    test_engine_no_pyramiding()
    test_engine_no_entry_after_stop_when_signal_was_emitted_during_position()
    test_engine_entry_price_rounded_to_tick()
    test_engine_rejects_non_utc_tz()
    test_engine_parity_doc_hash_in_meta()
    print("\nAll backtest engine tests passed.")
