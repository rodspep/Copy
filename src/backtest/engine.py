"""Backtest engine — vectorized state machine that consumes a pre-computed signal
DataFrame and an OHLCV LTF DataFrame.

This module is the only place in the codebase that simulates fills. It implements
the parity ADR (`docs/decisions/backtest_live_parity.md`) exactly. Live execution
adapters MUST call `src.backtest.fills.*` and `src.backtest.sizing.*` to match.

----
Algorithm (per ADR §2, §3):

For each LTF bar i = 1..N-1 (bar 0 has no preceding signal):

  Let `prev_signal = signals.iloc[i-1]`  ← signal emitted at close of bar i-1.
  Let `bar = ltf.iloc[i]`                 ← bar i, whose OPEN is the candidate fill price.

  1. Stale-signal guard (§1): if `bar.timestamp != ltf.timestamp[i-1] + Δ_l`,
     mark `stale = True`. A stale signal is logged but NOT actioned. Existing
     positions still process SL/TP on this bar under the gap rules.

  2. If position is open AND prev_signal.action == 'exit' AND not stale:
       Manual exit at bar.open (§3.7) BEFORE any SL/TP eval. Close position.
     Else if position is open:
       Evaluate SL/TP against bar [open, high, low] (§3.2). If fill, close.

  3. If position is now None AND prev_signal.action in (enter_long, enter_short)
     AND not stale:
       a) Compute entry_price (§3.1, §3.3).
       b) Round SL/TP (§3.1).
       c) Validate SL/TP geometry (§3.2). If invalid, skip.
       d) Size qty (§4). If qty == 0 (below min_qty), skip.
       e) Open position. Charge entry commission (§3.5).
       f) IMMEDIATELY evaluate SL/TP against bar i's [high, low] — §3.2
          requires the entry bar to be included in SL/TP evaluation.

  4. Equity update: mark-to-market on bar.close, recorded into equity_curve.

Closed trades are recorded into `trades` with side, entry/exit price, qty, R, etc.

----
Inputs:

- `ltf`     : pd.DataFrame with columns ['timestamp','open','high','low','close','volume'].
              UTC tz-aware, sorted, deduplicated. (Loaders enforce this.)
- `signals` : pd.DataFrame with same length as ltf, columns:
              ['action','sl','tp']
              action ∈ {'enter_long','enter_short','exit','hold'}
              sl, tp: float (required for enter_*; ignored otherwise)
- `symbol`  : str, key into SYMBOLS dict in src/config.py.
- `ltf_tf`  : str, e.g. 'M5' — used for stale-signal Δ_l comparison.
- `params`  : dict with at least:
              {'initial_equity': float, 'risk_pct': float, 'compounding': bool}

Outputs (returned as a dict to avoid tuple positional confusion):
- 'trades'       : pd.DataFrame of closed trades.
- 'equity_curve' : pd.DataFrame indexed by ltf.timestamp with column 'equity'.
- 'stats'        : dict of summary metrics (computed in src.reports, not here).
- 'meta'         : dict with parity_doc_sha256, symbol, ltf_tf, run_started_at.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.config import SYMBOLS, ROOT
from src.indicators.htf import TF_DURATION
from src.backtest.fills import (
    entry_fill_price,
    round_price_to_tick,
    round_sl_tp,
    validate_sltp_after_entry,
    evaluate_sl_tp_on_bar,
    market_exit_price,
    commission,
)
from src.backtest.sizing import position_size


VALID_ACTIONS = {"enter_long", "enter_short", "exit", "hold"}


@dataclass
class Position:
    side: int  # +1 long, -1 short
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    sl: float
    tp: float
    risk_amount: float
    entry_commission: float
    meta: dict = field(default_factory=dict)
    # Partial-close state (added 2026-05-27 — see parity ADR §10).
    # If `partial_tp_price` is finite, the engine will close `partial_close_frac`
    # of the position when that price is touched, then optionally move SL to BE.
    # `partial_done` flips True after the partial fires; second close uses tp/sl
    # for the remaining qty only.
    partial_tp_price: float = float("nan")
    partial_close_frac: float = 0.5
    partial_done: bool = False
    original_qty: float = 0.0
    # When True, after partial fill the SL is moved to entry_price.
    move_sl_to_be: bool = True


def _parity_doc_sha256() -> str:
    path = ROOT / "docs" / "decisions" / "backtest_live_parity.md"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _close_position(
    position: "Position",
    *,
    exit_idx: int,
    exit_time: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
    exit_commission: float,
    contract_multiplier: float,
) -> tuple[dict, float]:
    """Build the trade record + realized PnL for one closed position.

    Returns (trade_dict, pnl). The 16-field schema and PnL formula live in ONE
    place so a future schema change (e.g. adding MFE/MAE) or PnL-math change
    cannot diverge between the four close paths (manual exit, intra-bar SL/TP,
    same-bar entry+SL, force-EOD).
    """
    pnl = (
        position.side * position.qty * contract_multiplier
        * (exit_price - position.entry_price)
        - position.entry_commission - exit_commission
    )
    record = {
        "entry_idx": position.entry_idx,
        "entry_time": position.entry_time,
        "exit_idx": exit_idx,
        "exit_time": exit_time,
        "side": position.side,
        "qty": position.qty,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "sl": position.sl,
        "tp": position.tp,
        "exit_reason": exit_reason,
        "entry_commission": position.entry_commission,
        "exit_commission": exit_commission,
        "pnl": pnl,
        "R_realized": (pnl / position.risk_amount) if position.risk_amount > 0 else np.nan,
        "bars_held": exit_idx - position.entry_idx,
    }
    return record, pnl


def _validate_inputs(ltf: pd.DataFrame, signals: pd.DataFrame, ltf_tf: str) -> None:
    required_ltf = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required_ltf - set(ltf.columns)
    if missing:
        raise ValueError(f"ltf missing required columns: {missing}")
    tz = ltf["timestamp"].dt.tz
    if tz is None or str(tz) != "UTC":
        raise ValueError(f"ltf timestamps must be tz-aware UTC, got tz={tz}")
    if not ltf["timestamp"].is_monotonic_increasing:
        raise ValueError("ltf timestamps must be sorted ascending")
    if ltf["timestamp"].duplicated().any():
        raise ValueError("ltf has duplicate timestamps")

    if len(signals) != len(ltf):
        raise ValueError(f"signals length {len(signals)} != ltf length {len(ltf)}")
    required_sig = {"action", "sl", "tp"}
    missing_sig = required_sig - set(signals.columns)
    if missing_sig:
        raise ValueError(f"signals missing required columns: {missing_sig}")
    bad = set(signals["action"].unique()) - VALID_ACTIONS
    if bad:
        raise ValueError(f"signals contains invalid actions: {bad}")

    if ltf_tf not in TF_DURATION:
        raise ValueError(f"unknown ltf_tf {ltf_tf}; must be one of {list(TF_DURATION)}")


def run_backtest(
    ltf: pd.DataFrame,
    signals: pd.DataFrame,
    symbol: str,
    ltf_tf: str,
    params: Optional[dict] = None,
) -> dict:
    """Run the backtest. See module docstring for full semantics.

    This function does **not** mutate inputs. It builds `trades` and `equity_curve`
    by walking bar-by-bar and applying the parity-ADR fill rules.
    """
    if symbol not in SYMBOLS:
        raise ValueError(f"unknown symbol {symbol}; must be one of {list(SYMBOLS)}")
    _validate_inputs(ltf, signals, ltf_tf)

    params = dict(params or {})
    initial_equity: float = float(params.get("initial_equity", 10_000.0))
    risk_pct: float = float(params.get("risk_pct", 0.005))
    compounding: bool = bool(params.get("compounding", True))

    sym = SYMBOLS[symbol]
    pip = float(sym["pip"])
    min_tick = float(sym["min_tick"])
    spread_pips = float(sym["spread_pips"])
    slippage_pips = float(sym["slippage_pips"])
    qty_step = float(sym["qty_step"])
    min_qty = float(sym["min_qty"])
    contract_multiplier = float(sym["contract_multiplier"])
    commission_pct = float(sym["commission_pct"])
    commission_usd = float(sym["commission_usd"])

    ltf_dur = TF_DURATION[ltf_tf]

    n = len(ltf)
    ts = ltf["timestamp"].to_numpy()
    open_ = ltf["open"].to_numpy(dtype="float64")
    high = ltf["high"].to_numpy(dtype="float64")
    low = ltf["low"].to_numpy(dtype="float64")
    close = ltf["close"].to_numpy(dtype="float64")

    action = signals["action"].to_numpy()
    sig_sl = signals["sl"].to_numpy(dtype="float64")
    sig_tp = signals["tp"].to_numpy(dtype="float64")

    # Pre-compute stale-signal flags: bar i is "after a gap" iff ts[i] != ts[i-1] + Δ_l.
    # Bar 0 has no predecessor; we treat its signal slot as always non-actionable
    # (no preceding signal can act on bar 0).
    is_stale = np.zeros(n, dtype=bool)
    if n > 1:
        prev_ts = pd.Series(ts[:-1])
        expected_next = prev_ts + ltf_dur
        actual_next = pd.Series(ts[1:])
        is_stale[1:] = (expected_next.to_numpy() != actual_next.to_numpy())

    trades: list[dict] = []
    equity_arr = np.full(n, initial_equity, dtype="float64")
    equity = initial_equity
    position: Optional[Position] = None

    for i in range(1, n):
        prev_action = action[i - 1]
        bar_open = float(open_[i])
        bar_high = float(high[i])
        bar_low = float(low[i])
        bar_ts = pd.Timestamp(ts[i])
        stale = bool(is_stale[i])

        # Snapshot whether we were already in a position at the START of bar i.
        # Per ADR §6: a strategy that emits `enter_*` while a position is open
        # has that signal ignored. In live, the signal at bar i-1 close was
        # emitted with an open position, so it never reaches the broker. The
        # backtest must respect the same gate even if the old position is
        # closed via SL/TP during bar i — otherwise we'd retroactively act on
        # a signal that live would have ignored.
        had_position_at_bar_start = position is not None

        # ----- 2. Handle existing position -----
        if position is not None:
            # 2a. Manual exit (precedence over SL/TP — §3.7)
            if prev_action == "exit" and not stale:
                exit_price = market_exit_price(bar_open, position.side, slippage_pips, pip)
                exit_commission = commission(
                    exit_price, position.qty, contract_multiplier, commission_pct, commission_usd
                )
                record, pnl = _close_position(
                    position, exit_idx=i, exit_time=bar_ts,
                    exit_price=exit_price, exit_reason="manual",
                    exit_commission=exit_commission, contract_multiplier=contract_multiplier,
                )
                trades.append(record)
                equity += pnl
                position = None
            else:
                # 2b. SL/TP evaluation against bar i (§3.2)
                fill = evaluate_sl_tp_on_bar(
                    side=position.side,
                    bar_open=bar_open,
                    bar_high=bar_high,
                    bar_low=bar_low,
                    sl=position.sl,
                    tp=position.tp,
                    slippage_pips=slippage_pips,
                    pip=pip,
                )
                if fill.filled:
                    exit_price = fill.price
                    exit_commission = commission(
                        exit_price, position.qty, contract_multiplier, commission_pct, commission_usd
                    )
                    record, pnl = _close_position(
                        position, exit_idx=i, exit_time=bar_ts,
                        exit_price=exit_price, exit_reason=fill.reason,
                        exit_commission=exit_commission, contract_multiplier=contract_multiplier,
                    )
                    trades.append(record)
                    equity += pnl
                    position = None

        # ----- 3. Handle new entry (only if flat AT THE START of this bar) -----
        # `had_position_at_bar_start` enforces ADR §6: a signal emitted while a
        # position was open is ignored even if the old position closes mid-bar.
        if (
            not had_position_at_bar_start
            and position is None
            and prev_action in ("enter_long", "enter_short")
            and not stale
        ):
            side: int = 1 if prev_action == "enter_long" else -1
            planned_sl = float(sig_sl[i - 1])
            planned_tp = float(sig_tp[i - 1])

            if not (np.isfinite(planned_sl) and np.isfinite(planned_tp)):
                # Strategy emitted an enter_* without valid SL/TP — skip.
                pass
            else:
                # 3a. Entry price (§3.1) — round to min_tick conservatively:
                #     long: round UP (we pay no less than modeled)
                #     short: round DOWN (we receive no more than modeled)
                entry_price_raw = entry_fill_price(bar_open, side, spread_pips, slippage_pips, pip)
                entry_price = round_price_to_tick(
                    entry_price_raw,
                    min_tick,
                    "up" if side == 1 else "down",
                )

                # 3b. Round SL/TP (§3.1). NOTE: rounding may move SL/TP a fraction
                # against us, which is checked again by 3c.
                sl_r, tp_r = round_sl_tp(side, planned_sl, planned_tp, min_tick)

                # 3c. Validate post-entry geometry (§3.2). If gap moved price past
                # SL or TP after rounding, cancel.
                if not validate_sltp_after_entry(side, entry_price, sl_r, tp_r):
                    pass
                else:
                    # 3d. Size (§4). Use current equity (compounding) or initial (fixed).
                    sizing_equity = equity if compounding else initial_equity
                    risk_amount = sizing_equity * risk_pct
                    qty = position_size(
                        equity=sizing_equity,
                        risk_pct=risk_pct,
                        entry_price=entry_price,
                        sl_price=sl_r,
                        contract_multiplier=contract_multiplier,
                        qty_step=qty_step,
                        min_qty=min_qty,
                    )
                    if qty > 0:
                        # 3e. Open position
                        entry_commission = commission(
                            entry_price, qty, contract_multiplier, commission_pct, commission_usd
                        )
                        position = Position(
                            side=side,
                            entry_idx=i,
                            entry_time=bar_ts,
                            entry_price=entry_price,
                            qty=qty,
                            sl=sl_r,
                            tp=tp_r,
                            risk_amount=risk_amount,
                            entry_commission=entry_commission,
                        )

                        # 3f. Evaluate SL/TP on the ENTRY BAR (§3.2: "starting with
                        # the entry bar itself"). This lets scalps that enter and
                        # exit on the same M1/M5 candle settle correctly.
                        fill = evaluate_sl_tp_on_bar(
                            side=position.side,
                            bar_open=bar_open,
                            bar_high=bar_high,
                            bar_low=bar_low,
                            sl=position.sl,
                            tp=position.tp,
                            slippage_pips=slippage_pips,
                            pip=pip,
                        )
                        if fill.filled:
                            exit_price = fill.price
                            exit_commission = commission(
                                exit_price, position.qty, contract_multiplier,
                                commission_pct, commission_usd
                            )
                            record, pnl = _close_position(
                                position, exit_idx=i, exit_time=bar_ts,
                                exit_price=exit_price, exit_reason=fill.reason,
                                exit_commission=exit_commission,
                                contract_multiplier=contract_multiplier,
                            )
                            trades.append(record)
                            equity += pnl
                            position = None

        # ----- 4. Equity mark-to-market on bar close -----
        if position is not None:
            unrealized = (
                position.side * position.qty * contract_multiplier *
                (close[i] - position.entry_price)
                - position.entry_commission
            )
            equity_arr[i] = equity + unrealized
        else:
            equity_arr[i] = equity

    # Force-EOD close (ADR §3.6.1): synthetic market exit at the last bar's CLOSE,
    # slippage applied, no synthetic spread. Backtest-only — live has no end-of-data.
    if position is not None:
        last_i = n - 1
        exit_price = market_exit_price(float(close[last_i]), position.side, slippage_pips, pip)
        exit_commission = commission(
            exit_price, position.qty, contract_multiplier, commission_pct, commission_usd
        )
        record, pnl = _close_position(
            position, exit_idx=last_i, exit_time=pd.Timestamp(ts[last_i]),
            exit_price=exit_price, exit_reason="force_eod",
            exit_commission=exit_commission, contract_multiplier=contract_multiplier,
        )
        trades.append(record)
        equity += pnl
        equity_arr[last_i] = equity
        position = None

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame({
        "timestamp": ltf["timestamp"].to_numpy(),
        "equity": equity_arr,
    })

    meta = {
        "parity_doc_sha256": _parity_doc_sha256(),
        "symbol": symbol,
        "ltf_tf": ltf_tf,
        "n_bars": int(n),
        "n_trades": int(len(trades_df)),
        "stale_signal_bars": int(is_stale.sum()),
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "initial_equity": initial_equity,
            "risk_pct": risk_pct,
            "compounding": compounding,
        },
        "symbol_spec": dict(sym),
    }

    return {
        "trades": trades_df,
        "equity_curve": equity_df,
        "meta": meta,
    }
