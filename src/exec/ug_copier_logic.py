"""Pure decision logic for the UG copier — no MT5, no I/O, fully unit-testable.

Given a parsed UG signal + the current price + config, decide whether to place a
pending LIMIT order and with what parameters, or skip (with a reason). This is the
safety-critical core, kept pure so it can be tested exhaustively before any real
order is ever placed.

UG signal geometry (decoded — see docs/decisions/ug_logic_decode.md):
  - Entry is a ZONE "A - B"; SL is a fixed price; TP1..TP4 in pip (1 pip = 0.1 px).
  - Entry within the zone is chosen PER METHOD (ENTRY_MODE_BY_TP1) — all MID for now.
  - TP1 ∈ {50, 100, 150} all trade on real money, entry MID, with a PER-METHOD 2-leg exit
    (near + runner, SL→BE). Near/runner distances are per method (TP1_PIP_BY_METHOD,
    RUNNER_PIP_BY_TP1 below), backtest-tuned per method. OBSERVE_ONLY_TP1 is empty.
  - Placement is a pull-back LIMIT: place when price is on the fillable side of entry
    and WAIT for the pull-back. With DEEP_LIMIT=True we place even if price already
    passed TP1 (it's a limit, not a chase); DEEP_LIMIT=False restores the old
    "skip if price ran to TP1" chase behaviour. See DEEP_LIMIT below.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

PIP = 0.1                       # XAU: 1 pip = 0.1 price
# All methods (50/100/150) trade on real money at MID entry with the per-method 2-leg exit
# (TP1_PIP_BY_METHOD near leg + RUNNER_PIP_BY_TP1 runner, SL→BE after the near leg books).
# Distances are backtested per method (scripts/tcu_edge, scripts/tcu_legs): the 50-method
# books quick (near 50pip, tight SL); the 150-method runs far (near 100pip, runner 200pip);
# the 100-method has no live history → safe defaults (near 50, runner 150). Entry held at
# MID for all. OBSERVE_ONLY_TP1 is empty (no method is demo-only anymore).
# near=entry_low (entry-side); deep=entry_high (SL-side); mid=midpoint.
ENTRY_MODE_BY_TP1 = {50.0: "mid", 100.0: "mid", 150.0: "mid"}
OBSERVE_ONLY_TP1 = ()                   # (was 100/150) — now ALL methods use the same exit
# UNIFIED EXIT (user decision, applies to demo AND real): every UG signal — regardless of
# its published TP template — is traded as a 2-leg bracket (near TP leg + runner leg,
# SL→BE after the near leg books). Both leg distances are PER METHOD (TP1_PIP_BY_METHOD,
# RUNNER_PIP_BY_TP1): the tight-SL 50-method books quick (near 50pip); the wide-SL 150-method
# books the near leg at 100pip (it runs far). tp1_pip carries the signal's method (50/100/150)
# for /stats labelling + dedup; the FIXED_* values below are only the fallback defaults.
FIXED_TP1_PIP = 50.0
FIXED_TP3_PIP = 150.0
# Runner (tp3) distance PER METHOD. Backtest (scripts/tcu_edge, recency-weighted): the
# 150-method (Ai Signals) RUNS far (median MFE ~200pip, ~50% reach +200) → runner 200pip;
# the 50-method doesn't run as far → keeps the 150 runner.
RUNNER_PIP_BY_TP1 = {50.0: 150.0, 100.0: 150.0, 150.0: 200.0}
# TP1 (near leg) distance PER METHOD. The 150-method has a WIDE ~150pip SL (it targets far),
# so booking the near leg at only 50pip wastes it AND leaves a 0.33 TP1:SL ratio. Backtest
# (scripts/tcu_edge, ALL windows full/28d/15d) shows the 150-method's near leg at 100pip
# earns +35-89% per signal vs 50pip (WR 85%->~74%, but profit ~doubles on recent; TP1:SL
# improves to 0.67). The 50-method has a tighter ~115pip SL — booking QUICK at 50 is optimal
# (raising it LOSES on every window). 100-method has no live history → keep 50 (conservative).
TP1_PIP_BY_METHOD = {50.0: 50.0, 100.0: 50.0, 150.0: 100.0}
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
    anchor: float = 0.0         # signal anchor (mid); a MARKET fill must be at-or-better


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

    real_mode=True (trading real money) skips any method still listed in OBSERVE_ONLY_TP1
    (currently EMPTY — all of 50/100/150 trade real with the per-method 2-leg exit). The
    hook is kept so a method can be demoted to demo-only again without code changes."""
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
    # Near-leg TP distance is PER METHOD (TP1_PIP_BY_METHOD): 50pip for the tight-SL 50/100
    # methods, 100pip for the wide-SL (~150pip) 150-method which runs far. tp1_pip stays the
    # method identity (50/100/150) for /stats + dedup; tp1_leg_pip is the actual TP distance.
    tp1_leg_pip = TP1_PIP_BY_METHOD.get(tp1_pip, FIXED_TP1_PIP)
    tp1_price = mid + sign * tp1_leg_pip * PIP

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
    anchor_r = round(mid, 3)
    legs = [Order(side=direction, order_type=otype, entry=entry_r, sl=sl_r,
                  tp=round(tp1_price, 3), volume=volume, tp1_pip=tp1_pip, leg="tp1",
                  tp_pip=tp1_leg_pip, anchor=anchor_r)]

    # Runner leg: TP distance is per-method (RUNNER_PIP_BY_TP1) — 150-method runs farther so
    # its runner is 200pip; others keep 150pip. SL→BE after the TP1 leg books (managed live).
    if RUNNER_TP3:
        tp3_pip = RUNNER_PIP_BY_TP1.get(tp1_pip, FIXED_TP3_PIP)
        tp3_price = mid + sign * tp3_pip * PIP
        legs.append(Order(side=direction, order_type=otype, entry=entry_r, sl=sl_r,
                          tp=round(tp3_price, 3), volume=volume, tp1_pip=tp1_pip,
                          leg="tp3", tp_pip=tp3_pip, anchor=anchor_r))
    return Decision("place", "ok", orders=tuple(legs))


def should_cancel_pending(order: Order, current_price: float) -> bool:
    """Cancel a still-unfilled pending limit if price reached TP1 without us (the
    move happened) — same 'don't chase' rule, applied while the order waits."""
    if order.side == "long":
        return current_price >= order.tp
    return current_price <= order.tp
