"""Tests for the UG copier decision logic — must be airtight (it places real orders)."""
from __future__ import annotations

from src.exec.ug_copier_logic import decide, should_cancel_pending, Order


def _buy(tp1=150, lo=4468, hi=4458, sl=4448):
    return {"direction": "long", "entry_low": lo, "entry_high": hi, "sl": sl,
            "tps_pip": {1: tp1, 2: 200, 3: 300, 4: 400}}


def _sell(tp1=150, lo=4552, hi=4555, sl=4565):
    return {"direction": "short", "entry_low": lo, "entry_high": hi, "sl": sl,
            "tps_pip": {1: tp1, 2: 200, 3: 300}}


def test_buy_150_picks_deep_edge_tp_from_entry():
    d = decide(_buy(tp1=150), current_price=4465)
    assert d.action == "place"
    o = d.order
    assert o.order_type == "buy_limit"
    assert o.entry == 4458            # 150 → deep (low) edge
    assert o.tp == 4458 + 150 * 0.1   # 4473, TP1 from entry
    assert o.sl == 4448
    assert o.volume == 0.01


def test_sell_150_picks_high_edge():
    d = decide(_sell(tp1=150), current_price=4550)
    assert d.action == "place"
    assert d.order.order_type == "sell_limit"
    assert d.order.entry == 4555      # 150 → deep (high) edge for a short
    assert d.order.tp == 4555 - 150 * 0.1   # 4540


def test_pp2_uses_mid_entry():
    d = decide(_buy(tp1=50, lo=4468, hi=4458), current_price=4465)
    assert d.action == "place", d.reason
    assert d.order.entry == 4463                      # mid of 4468/4458
    assert d.order.tp == round(4463 + 50 * 0.1, 3)    # TP from mid


def test_100_uses_near_entry():
    # TP1=100 → NEAR (1st number = entry_low). price must be > near to rest a buy-limit.
    d = decide(_buy(tp1=100, lo=4468, hi=4458), current_price=4470)
    assert d.action == "place", d.reason
    assert d.order.entry == 4468                      # near (1st number / entry_low)
    assert d.order.tp == round(4468 + 100 * 0.1, 3)   # 4478, TP from near
    assert d.order.sl == 4448                         # fixed stated SL


def test_filter_rejects_other_tp1():
    assert decide(_buy(tp1=200), 4465).action == "skip"   # not in {50,100,150}
    assert decide(_buy(tp1=300), 4465).action == "skip"


def test_filter_accepts_50_100_150():
    # price chosen above each method's entry edge so the buy-limit is valid:
    # 50→mid(4463), 100→near(4468), 150→deep(4458).
    assert decide(_buy(tp1=50), 4465).action == "place"
    assert decide(_buy(tp1=100), 4470).action == "place"
    assert decide(_buy(tp1=150), 4465).action == "place"


def test_skip_if_price_past_tp1():
    # BUY (150→deep) entry 4458, TP1 4473. Price already 4474 → move done → skip.
    assert decide(_buy(tp1=150), current_price=4474).action == "skip"
    # SELL (150→deep) entry 4555, TP1 4540. Price already 4539 → skip.
    assert decide(_sell(tp1=150), current_price=4539).action == "skip"


def test_place_if_price_between_entry_and_tp1():
    # BUY: price 4465 (between entry 4458 and TP1 4473) → still place the limit.
    assert decide(_buy(tp1=150), 4465).action == "place"


def test_skip_if_price_below_entry_for_long():
    # BUY entry 4458; price already 4450 (below the deep edge) → can't rest a
    # buy-limit above market / pullback already triggered → skip.
    assert decide(_buy(tp1=150), current_price=4450).action == "skip"
    # SELL (150→deep) entry 4555; price already 4560 (above) → skip.
    assert decide(_sell(tp1=150), current_price=4560).action == "skip"


def test_bad_geometry_skipped():
    bad = _buy(tp1=150, sl=4480)       # SL above entry for a long → malformed
    assert decide(bad, 4465).action == "skip"


def test_missing_fields_skipped():
    assert decide({"direction": "long"}, 4465).action == "skip"
    assert decide({"direction": "long", "entry_low": 1, "entry_high": 2,
                   "sl": 0.5, "tps_pip": {}}, 4465).action == "skip"


def test_should_cancel_pending():
    o = Order("long", "buy_limit", 4458, 4448, 4473, 0.01, 150)
    assert should_cancel_pending(o, 4474) is True     # reached TP1 unfilled
    assert should_cancel_pending(o, 4465) is False
    os = Order("short", "sell_limit", 4555, 4565, 4545, 0.01, 100)
    assert should_cancel_pending(os, 4544) is True
    assert should_cancel_pending(os, 4550) is False
