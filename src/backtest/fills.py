"""Pure fill-price arithmetic.

Every formula here implements a specific clause of the parity ADR
(`docs/decisions/backtest_live_parity.md`). Citations of the form §X.Y point to
that document. **Live execution adapters must use these same functions** — never
inline the math.

All functions take primitives (floats, dicts, ints), not DataFrames, so they are
trivially unit-testable in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import math


Side = Literal[1, -1]
"""+1 = long, -1 = short. We use ints, not strings, so sign math works directly."""


# -----------------------------------------------------------------------------
# Entry fill (ADR §3.1)
# -----------------------------------------------------------------------------

def entry_fill_price(
    bar_open: float,
    side: Side,
    spread_pips: float,
    slippage_pips: float,
    pip: float,
) -> float:
    """Synthetic entry price: open + side * (spread + slip) * pip.

    Loads the full round-trip spread onto the entry (§3.3) plus one slippage_pips
    of adverse entry slippage (§3.1, §3.4).

    Long: pays MORE than the open.
    Short: receives LESS than the open.
    """
    return bar_open + side * (spread_pips + slippage_pips) * pip


def round_price_to_tick(price: float, min_tick: float, direction: Literal["down", "up"]) -> float:
    """Round a price to the broker's min_tick.

    `direction='down'` → toward negative infinity (`math.floor`).
    `direction='up'`   → toward positive infinity (`math.ceil`).

    Per §3.1: long SL/TP round down, short SL/TP round up — always in the
    direction the broker will accept without tightening risk.
    """
    if min_tick <= 0:
        raise ValueError(f"min_tick must be > 0, got {min_tick}")
    n = price / min_tick
    if direction == "down":
        return math.floor(n) * min_tick
    if direction == "up":
        return math.ceil(n) * min_tick
    raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")


def round_sl_tp(side: Side, sl: float, tp: float, min_tick: float) -> tuple[float, float]:
    """Round SL/TP per §3.1 (long: both round down; short: both round up).

    Returns (sl_rounded, tp_rounded).
    """
    direction: Literal["down", "up"] = "down" if side == 1 else "up"
    return (
        round_price_to_tick(sl, min_tick, direction),
        round_price_to_tick(tp, min_tick, direction),
    )


def validate_sltp_after_entry(side: Side, entry_price: float, sl: float, tp: float) -> bool:
    """Per §3.2: after entry price is known, long requires sl < entry < tp;
    short requires tp < entry < sl. If invalid, the engine cancels the entry.

    Returns True iff the SL/TP geometry is valid for the given side.
    """
    if side == 1:
        return sl < entry_price < tp
    if side == -1:
        return tp < entry_price < sl
    raise ValueError(f"side must be +1 or -1, got {side}")


# -----------------------------------------------------------------------------
# Intra-bar SL/TP evaluation (ADR §3.2)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SlTpFill:
    """Result of evaluating one bar against an open position's SL/TP."""

    reason: Literal["sl", "tp", "none"]
    price: float  # exit price (after slippage where applicable); NaN if reason=='none'

    @property
    def filled(self) -> bool:
        return self.reason in ("sl", "tp")


def evaluate_sl_tp_on_bar(
    side: Side,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    sl: float,
    tp: float,
    slippage_pips: float,
    pip: float,
) -> SlTpFill:
    """Evaluate one bar's [open, high, low] against a position's SL/TP.

    Implements §3.2 exactly:

    1. Adverse open gap (long: open ≤ sl; short: open ≥ sl) → SL fills at
       `bar_open - side * slippage_pips * pip` (worse than the stop).
    2. Favorable open gap (long: open ≥ tp; short: open ≤ tp) → TP fills at
       `tp` exactly (limits never improve).
    3. Both SL and TP in range, no gap → SL first (pessimistic).
    4. Only SL touched intra-bar → SL fills at `sl - side * slippage_pips * pip`.
    5. Only TP touched intra-bar → TP fills at `tp` exactly.
    6. Neither touched → reason='none'.

    Notes:
    - "Adverse open" is checked first; if true, we do NOT also check TP for the
      same bar — the gap-through-stop is the decisive event.
    - "Favorable open gap" is also checked before intra-bar TP evaluation, but
      ONLY if the adverse-open check did not already fire. If both adverse and
      favorable opens are possible (only happens with absurdly tight SL/TP
      relative to bar range), the adverse case wins — pessimistic.
    """
    slip = slippage_pips * pip

    if side == 1:
        # --- Long ---
        # Adverse open gap: opens at or below SL
        if bar_open <= sl:
            return SlTpFill(reason="sl", price=bar_open - slip)
        # Favorable open gap: opens at or above TP
        if bar_open >= tp:
            return SlTpFill(reason="tp", price=tp)
        # Both touched intra-bar → SL first
        sl_touched = bar_low <= sl
        tp_touched = bar_high >= tp
        if sl_touched and tp_touched:
            return SlTpFill(reason="sl", price=sl - slip)
        if sl_touched:
            return SlTpFill(reason="sl", price=sl - slip)
        if tp_touched:
            return SlTpFill(reason="tp", price=tp)
        return SlTpFill(reason="none", price=float("nan"))

    if side == -1:
        # --- Short ---
        # Adverse open gap: opens at or above SL
        if bar_open >= sl:
            return SlTpFill(reason="sl", price=bar_open + slip)
        # Favorable open gap: opens at or below TP
        if bar_open <= tp:
            return SlTpFill(reason="tp", price=tp)
        # Both touched intra-bar → SL first
        sl_touched = bar_high >= sl
        tp_touched = bar_low <= tp
        if sl_touched and tp_touched:
            return SlTpFill(reason="sl", price=sl + slip)
        if sl_touched:
            return SlTpFill(reason="sl", price=sl + slip)
        if tp_touched:
            return SlTpFill(reason="tp", price=tp)
        return SlTpFill(reason="none", price=float("nan"))

    raise ValueError(f"side must be +1 or -1, got {side}")


# -----------------------------------------------------------------------------
# Manual strategy exit (ADR §3.7)
# -----------------------------------------------------------------------------

def market_exit_price(
    reference_price: float,
    side: Side,
    slippage_pips: float,
    pip: float,
) -> float:
    """Market exit price with slippage applied; NO synthetic spread (§3.3 already
    loaded full round-trip spread onto entry; adding it again here would
    double-charge — §3.7).

    Two callers in the engine:
      - **Manual strategy exit** (§3.7): `reference_price` = bar i+1 open.
      - **Force-EOD close** (§3.6.1): `reference_price` = last bar's close.

    Long exit: receives LESS than reference (we pay slippage selling).
    Short exit: pays MORE than reference (we pay slippage covering).
    """
    return reference_price - side * slippage_pips * pip


# -----------------------------------------------------------------------------
# Commission (ADR §3.5)
# -----------------------------------------------------------------------------

def commission(
    fill_price: float,
    qty: float,
    contract_multiplier: float,
    commission_pct: float,
    commission_usd: float,
) -> float:
    """Commission for a single executed fill (entry OR exit).

    `commission_pct` is a fraction of notional (e.g. 0.0004 = 0.04%).
    `commission_usd` is a fixed per-fill fee in account currency.

    Per §3.5: applied on BOTH entry and exit. The caller is responsible for
    invoking this function twice per trade.
    """
    notional = abs(fill_price * qty * contract_multiplier)
    return notional * commission_pct + commission_usd
