"""Pure decision logic for the UG copier — no MT5, no I/O, fully unit-testable.

Given a parsed UG signal + the current price + config, decide whether to place a
pending LIMIT order and with what parameters, or skip (with a reason). This is the
safety-critical core, kept pure so it can be tested exhaustively before any real
order is ever placed.

UG signal geometry (decoded — see docs/decisions/ug_logic_decode.md):
  - Entry is a ZONE "A - B"; SL is a fixed price; TP1..TP4 in pip (1 pip = 0.1 px).
  - BEST entry = the favourable (deep) edge of the zone: lowest price for a BUY,
    highest for a SELL → tightest risk to the fixed SL, best RR.
  - Only take the wider methods: TP1 ∈ {100, 150} pip (skip the PP2 scalp TP1=50).
  - If price has ALREADY run to/past TP1, the move is done → skip (UG's own
    "nếu giá đã chạy đến TP1, KHÔNG vào").
"""
from __future__ import annotations

from dataclasses import dataclass

PIP = 0.1                       # XAU: 1 pip = 0.1 price
ALLOWED_TP1_PIP = (100.0, 150.0)


@dataclass(frozen=True)
class Order:
    side: str                   # 'long' | 'short'
    order_type: str             # 'buy_limit' | 'sell_limit'
    entry: float                # limit price (deep edge of the zone)
    sl: float
    tp: float                   # TP1 from entry
    volume: float
    tp1_pip: float


@dataclass(frozen=True)
class Decision:
    action: str                 # 'place' | 'skip'
    reason: str
    order: Order | None = None


def _f(v):
    return None if v in (None, "") else float(v)


def decide(sig: dict, current_price: float, volume: float = 0.01) -> Decision:
    """Decide what to do with a parsed UG signal at the current market price."""
    direction = sig.get("direction")
    if direction not in ("long", "short"):
        return Decision("skip", f"bad direction {direction!r}")

    tp1_pip = _f((sig.get("tps_pip") or {}).get(1) or (sig.get("tps_pip") or {}).get("1"))
    if tp1_pip is None:
        return Decision("skip", "no TP1")
    if tp1_pip not in ALLOWED_TP1_PIP:
        return Decision("skip", f"filtered: TP1={tp1_pip:g}pip (only {ALLOWED_TP1_PIP})")

    lo, hi = _f(sig.get("entry_low")), _f(sig.get("entry_high"))
    sl = _f(sig.get("sl"))
    if lo is None or hi is None or sl is None:
        return Decision("skip", "missing entry zone / SL")

    # Deep (favourable) edge of the zone.
    if direction == "long":
        entry = min(lo, hi)
        otype, sign = "buy_limit", 1.0
    else:
        entry = max(lo, hi)
        otype, sign = "sell_limit", -1.0

    tp = entry + sign * tp1_pip * PIP

    # SL must sit on the correct (loss) side of entry; else the signal is malformed.
    if (direction == "long" and not (sl < entry < tp)) or \
       (direction == "short" and not (sl > entry > tp)):
        return Decision("skip", f"bad geometry: entry={entry} sl={sl} tp={tp}")

    # Price must sit BETWEEN the limit entry and TP1 for a valid pull-back limit:
    #  - long: entry < price < tp  (buy-limit rests below market; TP1 not yet hit)
    #  - short: tp < price < entry
    # price past TP1 → move done (don't chase); price already beyond entry → the
    # pull-back already triggered / limit would be on the wrong side of market.
    if direction == "long":
        if current_price >= tp:
            return Decision("skip", f"price {current_price} at/through TP1 {tp}")
        if current_price <= entry:
            return Decision("skip", f"price {current_price} already at/below entry {entry}")
    else:
        if current_price <= tp:
            return Decision("skip", f"price {current_price} at/through TP1 {tp}")
        if current_price >= entry:
            return Decision("skip", f"price {current_price} already at/above entry {entry}")

    return Decision("place", "ok", Order(
        side=direction, order_type=otype, entry=round(entry, 3),
        sl=round(sl, 3), tp=round(tp, 3), volume=volume, tp1_pip=tp1_pip))


def should_cancel_pending(order: Order, current_price: float) -> bool:
    """Cancel a still-unfilled pending limit if price reached TP1 without us (the
    move happened) — same 'don't chase' rule, applied while the order waits."""
    if order.side == "long":
        return current_price >= order.tp
    return current_price <= order.tp
