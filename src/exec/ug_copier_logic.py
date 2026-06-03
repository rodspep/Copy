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
  - Placement is a pull-back LIMIT: place when price is on the fillable side of entry
    and WAIT for the pull-back. With DEEP_LIMIT=True we place even if price already
    passed TP1 (it's a limit, not a chase); DEEP_LIMIT=False restores the old
    "skip if price ran to TP1" chase behaviour. See DEEP_LIMIT below.
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

# DEEP_LIMIT: place a pull-back LIMIT whenever price is on the fillable side of entry
# and WAIT for the pull-back — do NOT skip just because price already passed TP1.
# Verified on the full UG history (scripts/ug_method_pnl.py, Codex-reviewed): the old
# 'don't chase / skip if past TP1' rule discarded ~60% of the 50pip edge (38%→83% fill,
# +$30.90→+$115.30 in-sample). UG's 'don't chase' is for MARKET entries; we use limits.
# Forward-testing on demo. Set False to revert to the conservative chase behaviour.
DEEP_LIMIT = True


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

    # 'mid' = the published-entry anchor (TP1/TP3 prices are measured from it, matching
    # UG's stated levels). The entry ZONE is [zlo, zhi].
    mid = {"near": lo, "mid": (lo + hi) / 2.0, "deep": hi}[ENTRY_MODE_BY_TP1[tp1_pip]]
    long = direction == "long"
    sign = 1.0 if long else -1.0
    zlo, zhi = (lo, hi) if lo <= hi else (hi, lo)
    tp1_price = mid + sign * tp1_pip * PIP

    # SL must sit on the correct (loss) side, TP1 on the profit side, of the anchor.
    if (long and not (sl < mid < tp1_price)) or (not long and not (sl > mid > tp1_price)):
        return Decision("skip", f"bad geometry: mid={mid} sl={sl} tp1={tp1_price}")

    # ZONE-AWARE entry (DEEP_LIMIT). Decide entry style by where price sits vs the zone:
    #   - in the zone            → enter NOW at market (we're at an acceptable entry).
    #   - beyond zone, wait side → rest a LIMIT at the anchor, wait for the pull-back.
    #     (long: price above zone; short: price below zone)
    #   - past the zone the wrong way → setup voided → skip.
    #     (long: price below zone; short: price above zone)
    # DEEP_LIMIT=False restores the conservative legacy chase (limit only, between
    # entry and TP1, never market).
    if DEEP_LIMIT:
        # MARKET only when it gives an entry AT-OR-BETTER than the anchor (never chase):
        #   long  → market only if price <= mid (buy at/below planned entry);
        #           price > mid  → LIMIT at mid, wait for the pull-back DOWN (don't buy high).
        #   short → market only if price >= mid (sell at/above planned entry);
        #           price < mid  → LIMIT at mid, wait for the rally UP (don't sell low).
        # Past the zone the wrong way (long: price < zlo; short: price > zhi) → voided.
        if long:
            if current_price < zlo:
                return Decision("skip", f"price {current_price} below entry zone {zlo}-{zhi} — voided")
            entry_used, market = (current_price, True) if current_price <= mid else (mid, False)
        else:
            if current_price > zhi:
                return Decision("skip", f"price {current_price} above entry zone {zlo}-{zhi} — voided")
            entry_used, market = (current_price, True) if current_price >= mid else (mid, False)
    else:
        in_window = (mid < current_price < tp1_price) if long else (tp1_price < current_price < mid)
        if not in_window:
            return Decision("skip", f"chase: price {current_price} not between entry {mid} and TP1 {tp1_price}")
        entry_used, market = mid, False

    # Final entry-side sanity for the ACTUAL entry used.
    if (long and not (sl < entry_used < tp1_price)) or (not long and not (sl > entry_used > tp1_price)):
        return Decision("skip", f"entry {entry_used} not between SL {sl} and TP1 {tp1_price}")

    suffix = "market" if market else "limit"
    otype = f"{'buy' if long else 'sell'}_{suffix}"
    entry_r, sl_r = round(entry_used, 3), round(sl, 3)
    legs = [Order(side=direction, order_type=otype, entry=entry_r, sl=sl_r,
                  tp=round(tp1_price, 3), volume=volume, tp1_pip=tp1_pip, leg="tp1", tp_pip=tp1_pip)]

    # Runner leg to TP3 (UG's published TP3 level), if enabled + present + further than TP1.
    if RUNNER_TP3:
        tp3_raw = _f(tps.get(3) or tps.get("3"))
        if tp3_raw is not None and math.isfinite(tp3_raw) and tp3_raw > tp1_pip + 1e-9:
            tp3_price = mid + sign * tp3_raw * PIP
            beyond = (tp3_price > tp1_price) if long else (tp3_price < tp1_price)
            if beyond:
                legs.append(Order(side=direction, order_type=otype, entry=entry_r, sl=sl_r,
                                  tp=round(tp3_price, 3), volume=volume, tp1_pip=tp1_pip,
                                  leg="tp3", tp_pip=tp3_raw))
    return Decision("place", "ok", orders=tuple(legs))


def should_cancel_pending(order: Order, current_price: float) -> bool:
    """Cancel a still-unfilled pending limit if price reached TP1 without us (the
    move happened) — same 'don't chase' rule, applied while the order waits."""
    if order.side == "long":
        return current_price >= order.tp
    return current_price <= order.tp
