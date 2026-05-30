"""Position sizing per parity ADR §4.

`qty = risk_amount / (stop_distance_in_price * contract_multiplier)`
Round DOWN to `qty_step`. Never round up. If below `min_qty`, return 0 (skip).
"""
from __future__ import annotations

import math


def position_size(
    equity: float,
    risk_pct: float,
    entry_price: float,
    sl_price: float,
    contract_multiplier: float,
    qty_step: float,
    min_qty: float,
) -> float:
    """Returns qty (in the symbol's native unit — ounces for XAU, BTC for BTCUSDT).

    Returns 0.0 if the computed qty rounds below `min_qty` (skip the trade per §4).
    """
    if equity <= 0 or risk_pct <= 0:
        return 0.0
    if contract_multiplier <= 0:
        raise ValueError(f"contract_multiplier must be > 0, got {contract_multiplier}")
    stop_distance = abs(entry_price - sl_price)
    if stop_distance <= 0:
        # SL at entry → infinite size; reject. The ADR §3.2 validity check should
        # have already rejected this, but be defensive.
        return 0.0

    risk_amount = equity * risk_pct
    raw_qty = risk_amount / (stop_distance * contract_multiplier)

    if qty_step <= 0:
        raise ValueError(f"qty_step must be > 0, got {qty_step}")
    rounded_qty = math.floor(raw_qty / qty_step) * qty_step

    if rounded_qty < min_qty:
        return 0.0
    return rounded_qty
