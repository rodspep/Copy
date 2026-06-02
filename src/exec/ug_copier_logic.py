"""Pure decision logic for the UG copier — no MT5, no I/O, fully unit-testable.

Given a parsed UG signal + the current price + config, decide whether to place a
pending LIMIT order and with what parameters, or skip (with a reason). This is the
safety-critical core, kept pure so it can be tested exhaustively before any real
order is ever placed.

UG signal geometry (decoded — see docs/decisions/ug_logic_decode.md):
  - Entry is a ZONE "A - B"; SL is a fixed price; TP1..TP4 in pip (1 pip = 0.1 px).
  - Entry within the zone is chosen PER METHOD (ENTRY_MODE_BY_TP1) — all MID for now.
  - TP1 = 50 pip (PP2 scalp), entry MID, is the ONLY proven edge. TP1 ∈ {100, 150}
    are kept ON DEMO purely for OBSERVATION (gather comparison data) — NOT promoted
    to real money until they show an edge. See ENTRY_MODE_BY_TP1 below.
  - If price has ALREADY run to/past TP1, the move is done → skip (UG's own
    "nếu giá đã chạy đến TP1, KHÔNG vào").
"""
from __future__ import annotations

import math
from dataclasses import dataclass

PIP = 0.1                       # XAU: 1 pip = 0.1 price
# Entry edge per method, from a full-history backtest of every collected UG signal
# on real M1 (scripts/ug_method_pnl.py, 93 signals, 26/5–1/6, limit fill + 3pip
# cost, 0.01 lot):
#   TP1=50  (PP2 scalp), MID : +$30.90, WR 78%, +0.116R — the ONLY proven edge.
#   TP1=100                  : every entry mode net-negative (n=6, thin).
#   TP1=150 (PRI-GOLD)       : when filled, WR ~40% → net-negative (n=16). 'deep'
#                              never fills (no data), so we observe it at MID.
# DECISION (user): trade 50-mid for real; keep 100 & 150 ON DEMO at MID purely to
# OBSERVE / accumulate comparison data via /stats by_method. They are NOT edges yet
# — do NOT enable them for real money until the demo data says otherwise. Entry held
# at MID across all three so the only variable compared is TP1 size.
# near=entry_low (entry-side); deep=entry_high (SL-side); mid=midpoint.
ENTRY_MODE_BY_TP1 = {50.0: "mid", 100.0: "mid", 150.0: "mid"}
OBSERVE_ONLY_TP1 = (100.0, 150.0)      # not real-money edges; demo observation only
ALLOWED_TP1_PIP = tuple(ENTRY_MODE_BY_TP1)

# EXIT strategy (scripts/ug_exit_strategy.py, Codex-reviewed): instead of taking the
# whole position at TP1, split into TWO equal legs at the SAME entry+SL —
#   leg 'tp1' : TP = TP1  (locks the high-probability scalp win)
#   leg 'tp3' : TP = TP3  (runner; its SL is moved to break-even once leg 'tp1' wins)
# On the collected week this nearly doubled net (50pip bucket: +$53.40 vs +$30.90)
# at no extra risk on the runner after TP1. The broker closes each leg at its own TP
# natively; the copier only moves the runner's SL to BE after the TP1 leg closes.
# Kill-switch: set RUNNER_TP3=False to revert to a single TP1-full order.
RUNNER_TP3 = True


@dataclass(frozen=True)
class Order:
    side: str                   # 'long' | 'short'
    order_type: str             # 'buy_limit' | 'sell_limit'
    entry: float                # limit price
    sl: float
    tp: float                   # this leg's TP price
    volume: float
    tp1_pip: float              # method identity (TP1 pip) — same for both legs
    leg: str = "tp1"            # 'tp1' (scalp) | 'tp3' (runner, SL→BE after tp1 wins)
    tp_pip: float = 0.0         # this leg's TP distance in pip from entry


@dataclass(frozen=True)
class Decision:
    action: str                 # 'place' | 'skip'
    reason: str
    orders: tuple = ()          # bracket legs to place (1 = TP1-only; 2 = TP1+TP3)

    @property
    def order(self) -> Order | None:
        """First leg (TP1) — convenience for callers/tests that want a single order."""
        return self.orders[0] if self.orders else None


def _f(v):
    """Coerce to float; None for empty/None/non-numeric (never raises, so a malformed
    signal field degrades to a safe skip instead of crashing the poll loop)."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return None


def decide(sig: dict, current_price: float, volume: float = 0.01,
           real_mode: bool = False) -> Decision:
    """Decide what to do with a parsed UG signal at the current market price.

    real_mode=True (trading real money) restricts to the proven edge only: the
    observation-only methods (OBSERVE_ONLY_TP1) are skipped so demo-only data
    gathering never risks real funds."""
    if not isinstance(sig, dict):
        return Decision("skip", f"bad signal {sig!r}")
    direction = sig.get("direction")
    if direction not in ("long", "short"):
        return Decision("skip", f"bad direction {direction!r}")
    if not isinstance(current_price, (int, float)) or not math.isfinite(current_price):
        return Decision("skip", f"invalid current_price {current_price!r}")
    if not isinstance(volume, (int, float)) or not math.isfinite(volume) or volume <= 0:
        return Decision("skip", f"invalid volume {volume!r}")

    tps = sig.get("tps_pip")
    if not isinstance(tps, dict):
        tps = {}
    tp1_raw = _f(tps.get(1) or tps.get("1"))
    if tp1_raw is None or not math.isfinite(tp1_raw):
        return Decision("skip", "no/invalid TP1")
    # Canonicalize to an allowed key with tolerance so a computed float (e.g.
    # 150.0000001) still matches the dict/tuple keys robustly.
    tp1_pip = min(ALLOWED_TP1_PIP, key=lambda x: abs(x - tp1_raw))
    if abs(tp1_pip - tp1_raw) > 1e-6:
        return Decision("skip", f"filtered: TP1={tp1_raw:g}pip (only {ALLOWED_TP1_PIP})")
    if real_mode and tp1_pip in OBSERVE_ONLY_TP1:
        return Decision("skip", f"observe-only TP1={tp1_pip:g}pip not traded on real money")

    lo, hi = _f(sig.get("entry_low")), _f(sig.get("entry_high"))
    sl = _f(sig.get("sl"))
    if lo is None or hi is None or sl is None:
        return Decision("skip", "missing entry zone / SL")
    if not all(math.isfinite(v) for v in (lo, hi, sl)):
        return Decision("skip", "non-finite entry zone / SL")

    # Entry within the zone, per the method's chosen mode.
    # UG writes the zone "near - deep": entry_low(lo)=near (entry-side),
    # entry_high(hi)=deep (SL-side, = the fixed-SL anchor). mid = average.
    entry = {"near": lo, "mid": (lo + hi) / 2.0, "deep": hi}[ENTRY_MODE_BY_TP1[tp1_pip]]
    sign = 1.0 if direction == "long" else -1.0
    otype = "buy_limit" if direction == "long" else "sell_limit"

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

    entry_r, sl_r = round(entry, 3), round(sl, 3)
    leg_tp1 = Order(side=direction, order_type=otype, entry=entry_r, sl=sl_r,
                    tp=round(tp, 3), volume=volume, tp1_pip=tp1_pip,
                    leg="tp1", tp_pip=tp1_pip)

    # Runner leg to TP3 (only if enabled AND the signal carries a valid, further TP3).
    # If TP3 is missing/malformed, degrade safely to a single TP1-full order.
    if RUNNER_TP3:
        tp3_raw = _f(tps.get(3) or tps.get("3"))
        if tp3_raw is not None and math.isfinite(tp3_raw) and tp3_raw > tp1_pip + 1e-9:
            tp3_price = entry + sign * tp3_raw * PIP
            # tp3 must be strictly beyond tp1 on the profit side (sign already ensures
            # direction; tp3_raw>tp1_pip ensures further). Belt-and-suspenders check:
            beyond = (tp3_price > tp > entry) if direction == "long" else (tp3_price < tp < entry)
            if beyond:
                leg_tp3 = Order(side=direction, order_type=otype, entry=entry_r, sl=sl_r,
                                tp=round(tp3_price, 3), volume=volume, tp1_pip=tp1_pip,
                                leg="tp3", tp_pip=tp3_raw)
                return Decision("place", "ok", orders=(leg_tp1, leg_tp3))

    return Decision("place", "ok", orders=(leg_tp1,))


def should_cancel_pending(order: Order, current_price: float) -> bool:
    """Cancel a still-unfilled pending limit if price reached TP1 without us (the
    move happened) — same 'don't chase' rule, applied while the order waits."""
    if order.side == "long":
        return current_price >= order.tp
    return current_price <= order.tp
