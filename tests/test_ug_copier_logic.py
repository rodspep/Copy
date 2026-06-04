"""Tests for the UG copier decision logic — must be airtight (it places real orders).

Zone-aware entry (DEEP_LIMIT=True): price IN the entry zone → MARKET now; price beyond
the zone on the wait side → LIMIT at the anchor (mid); price past the zone the wrong way
→ skip. TP1/TP3 prices are measured from the anchor (UG's published levels).
"""
from __future__ import annotations

from src.exec.ug_copier_logic import decide, should_cancel_pending, Order


def _buy(tp1=150, lo=4468, hi=4458, sl=4448):
    # zone [4458,4468], mid 4463
    return {"direction": "long", "entry_low": lo, "entry_high": hi, "sl": sl,
            "tps_pip": {1: tp1, 2: 200, 3: 300, 4: 400}}


def _sell(tp1=150, lo=4552, hi=4555, sl=4565):
    # zone [4552,4555], mid 4553.5
    return {"direction": "short", "entry_low": lo, "entry_high": hi, "sl": sl,
            "tps_pip": {1: tp1, 2: 200, 3: 300}}


# ---- zone-aware entry style: MARKET only at-or-better than anchor mid ----
def test_buy_at_or_below_mid_enters_market():
    d = decide(_buy(tp1=150), current_price=4460)        # 4460 <= mid 4463, >= zlo 4458
    assert d.action == "place"
    o = d.order
    assert o.order_type == "buy_market"
    assert o.entry == 4460                                # market = current (good) price
    assert o.tp == round(4463 + 100 * 0.1, 3)            # 4473 — 150-method near leg = 100pip
    assert o.tp_pip == 100 and o.tp1_pip == 150          # actual TP 100pip; method id still 150
    assert o.sl == 4448 and o.volume == 0.01


def test_buy_above_mid_in_zone_uses_limit_not_market():
    # REGRESSION (the +$4.10 bug): price ABOVE the anchor but still in-zone must NOT
    # market-buy high ("đu đỉnh") — rest a LIMIT at mid and wait for the pull-back.
    d = decide(_buy(tp1=150), current_price=4465)        # 4465 > mid 4463, in zone
    assert d.action == "place"
    assert d.order.order_type == "buy_limit"
    assert d.order.entry == 4463                          # limit at anchor, not 4465


def test_buy_above_zone_rests_limit():
    d = decide(_buy(tp1=150), current_price=4475)        # 4475 > zhi → wait for pullback
    assert d.order.order_type == "buy_limit" and d.order.entry == 4463


def test_sell_at_or_above_mid_enters_market():
    d = decide(_sell(tp1=150), current_price=4554)        # 4554 >= mid 4553.5, <= zhi 4555
    assert d.action == "place"
    assert d.order.order_type == "sell_market"
    assert d.order.entry == 4554


def test_sell_below_mid_in_zone_uses_limit_not_market():
    # REGRESSION mirror: price below anchor (in zone) must rest a limit, not sell low.
    d = decide(_sell(tp1=150), current_price=4553)        # 4553 < mid 4553.5, in zone
    assert d.order.order_type == "sell_limit" and d.order.entry == 4553.5


def test_sell_below_zone_rests_limit():
    d = decide(_sell(tp1=150), current_price=4550)        # 4550 < zlo → wait for rally
    assert d.order.order_type == "sell_limit" and d.order.entry == 4553.5


def test_skip_past_zone_wrong_way():
    assert decide(_buy(tp1=150), current_price=4450).action == "skip"   # below buy zone
    assert decide(_sell(tp1=150), current_price=4560).action == "skip"  # above sell zone


# ---- method filter / observe-only ----
def test_filter_rejects_other_tp1():
    assert decide(_buy(tp1=200), 4465).action == "skip"
    assert decide(_buy(tp1=300), 4465).action == "skip"


def test_filter_accepts_50_100_150():
    assert decide(_buy(tp1=50), 4465).action == "place"
    assert decide(_buy(tp1=100), 4465).action == "place"
    assert decide(_buy(tp1=150), 4465).action == "place"


def test_real_mode_trades_all_methods_unified():
    # UNIFIED: 100/150 are no longer observe-only — they trade on real too (same exit).
    assert decide(_buy(tp1=50), 4465, real_mode=True).action == "place"
    assert decide(_buy(tp1=100), 4465, real_mode=True).action == "place"
    assert decide(_buy(tp1=150), 4465, real_mode=True).action == "place"


# ---- DEEP_LIMIT kill-switch (legacy chase: limit only, no market, no in-zone-below-mid) ----
def test_chase_mode_limit_and_skips(monkeypatch):
    import src.exec.ug_copier_logic as L
    monkeypatch.setattr(L, "DEEP_LIMIT", False)
    # between mid(4463) and TP1(4468=mid+50pip) → legacy limit at mid
    d = L.decide(_buy(tp1=150), current_price=4465)
    assert d.action == "place" and d.order.order_type == "buy_limit" and d.order.entry == 4463
    # past TP1 (150-method near leg now 100pip → TP1=4473) → skip; in-zone-below-mid → skip
    assert L.decide(_buy(tp1=150), 4475).action == "skip"
    assert L.decide(_buy(tp1=150), 4460).action == "skip"


# ---- bracket (TP1 + TP3 runner) ----
def test_bracket_two_legs_market_in_zone():
    d = decide(_buy(tp1=50, lo=4468, hi=4458, sl=4448), current_price=4461)  # <= mid 4463
    assert d.action == "place" and len(d.orders) == 2
    a, b = d.orders
    assert a.leg == "tp1" and b.leg == "tp3"
    assert a.order_type == b.order_type == "buy_market"
    assert a.entry == b.entry == 4461 and a.sl == b.sl == 4448
    assert a.tp == round(4463 + 50 * 0.1, 3)             # 4468 (anchor + FIXED TP1 50pip)
    assert b.tp == round(4463 + 150 * 0.1, 3)            # 4478 (anchor + FIXED TP3 150pip)
    assert b.tp_pip == 150 and a.tp_pip == 50
    assert a.tp1_pip == b.tp1_pip == 50                  # method id (here PP2=50)
    assert b.tp > a.tp                                    # runner strictly further


def test_bracket_short_limit_below_zone():
    d = decide(_sell(tp1=50, lo=4552, hi=4555, sl=4565), current_price=4550)
    assert d.action == "place" and len(d.orders) == 2
    a, b = d.orders
    assert a.order_type == b.order_type == "sell_limit" and a.entry == 4553.5
    assert b.tp < a.tp < a.entry                          # both TPs below entry for a short


def test_runner_always_added_fixed_150_even_without_published_tp3():
    # UNIFIED: the runner TP is a FIXED 150pip — added even if the signal lists no TP3.
    sig = {"direction": "long", "entry_low": 4468, "entry_high": 4458, "sl": 4448,
           "tps_pip": {1: 50, 2: 100}}                   # no TP3 published
    d = decide(sig, current_price=4461)
    assert d.action == "place" and len(d.orders) == 2
    assert d.orders[1].leg == "tp3" and d.orders[1].tp == round(4463 + 150 * 0.1, 3)


def test_runner_200pip_for_150_method():
    # 150-method (Ai Signals) runs far → near leg books at 100pip, runner TP at 200pip.
    d = decide(_buy(tp1=150), current_price=4460)        # mid 4463, market entry 4460
    assert len(d.orders) == 2
    a, b = d.orders
    assert a.leg == "tp1" and a.tp == round(4463 + 100 * 0.1, 3) and a.tp_pip == 100  # near 100pip
    assert b.leg == "tp3" and b.tp == round(4463 + 200 * 0.1, 3)   # 4483 — runner 200pip
    assert b.tp_pip == 200 and b.tp1_pip == 150                     # distance 200; method id 150


def test_tp1_distance_per_method():
    # Near-leg TP is per-method: 50-method books quick at 50pip (tight SL); 150-method at
    # 100pip (wide ~150pip SL, runs far). tp1_pip stays the method identity.
    d50 = decide(_buy(tp1=50), current_price=4460)
    assert d50.order.tp == round(4463 + 50 * 0.1, 3) and d50.order.tp_pip == 50 and d50.order.tp1_pip == 50
    d150 = decide(_buy(tp1=150), current_price=4460)
    assert d150.order.tp == round(4463 + 100 * 0.1, 3) and d150.order.tp_pip == 100


def test_runner_stays_150_for_50_and_100():
    for m in (50, 100):
        d = decide(_buy(tp1=m), current_price=4460)
        assert d.orders[1].tp_pip == 150                           # 50/100 keep the 150 runner


def test_runner_kill_switch(monkeypatch):
    import src.exec.ug_copier_logic as L
    monkeypatch.setattr(L, "RUNNER_TP3", False)
    d = L.decide(_buy(tp1=50, lo=4468, hi=4458, sl=4448), current_price=4461)
    assert d.action == "place" and len(d.orders) == 1 and d.orders[0].leg == "tp1"


def test_skip_propagates_to_no_orders():
    d = decide(_buy(tp1=200), 4465)
    assert d.action == "skip" and d.orders == () and d.order is None


# ---- malformed input guards (must never raise; must skip) ----
def test_bad_geometry_skipped():
    assert decide(_buy(tp1=150, sl=4480), 4465).action == "skip"   # SL above entry for a long


def test_missing_and_nonfinite_fields_skipped():
    nan = float("nan")
    assert decide({"direction": "long"}, 4465).action == "skip"
    assert decide(None, 4465).action == "skip"
    assert decide(_buy(tp1=nan), 4465).action == "skip"
    assert decide(_buy(), nan).action == "skip"
    assert decide(_buy(sl=nan), 4465).action == "skip"
    assert decide(_buy(), 4465, volume=0).action == "skip"
    assert decide({"direction": "long", "entry_low": 1, "entry_high": 2, "sl": 0.5,
                   "tps_pip": "garbage"}, 4465).action == "skip"


def test_should_cancel_pending():
    o = Order("long", "buy_limit", 4458, 4448, 4473, 0.01, 150)
    assert should_cancel_pending(o, 4474) is True
    assert should_cancel_pending(o, 4465) is False
    os = Order("short", "sell_limit", 4555, 4565, 4545, 0.01, 100)
    assert should_cancel_pending(os, 4544) is True
    assert should_cancel_pending(os, 4550) is False
