"""Tests for the UG copier decision logic — must be airtight (it places real orders)."""
from __future__ import annotations

from src.exec.ug_copier_logic import decide, should_cancel_pending, Order


def _buy(tp1=150, lo=4468, hi=4458, sl=4448):
    return {"direction": "long", "entry_low": lo, "entry_high": hi, "sl": sl,
            "tps_pip": {1: tp1, 2: 200, 3: 300, 4: 400}}


def _sell(tp1=150, lo=4552, hi=4555, sl=4565):
    return {"direction": "short", "entry_low": lo, "entry_high": hi, "sl": sl,
            "tps_pip": {1: tp1, 2: 200, 3: 300}}


def test_buy_picks_deep_edge_and_tp_from_entry():
    d = decide(_buy(tp1=150), current_price=4465)
    assert d.action == "place"
    o = d.order
    assert o.order_type == "buy_limit"
    assert o.entry == 4458            # deep (low) edge
    assert o.tp == 4458 + 150 * 0.1   # 4473, TP1 from entry
    assert o.sl == 4448
    assert o.volume == 0.01


def test_sell_picks_high_edge():
    d = decide(_sell(tp1=100), current_price=4550)
    assert d.action == "place"
    assert d.order.order_type == "sell_limit"
    assert d.order.entry == 4555      # high edge for a short
    assert d.order.tp == 4555 - 100 * 0.1   # 4545


def test_filter_rejects_pp2_tp1_50():
    assert decide(_buy(tp1=50), 4465).action == "skip"
    assert decide(_buy(tp1=200), 4465).action == "skip"   # not in {100,150}


def test_filter_accepts_100_and_150():
    assert decide(_buy(tp1=100), 4465).action == "place"
    assert decide(_buy(tp1=150), 4465).action == "place"


def test_skip_if_price_past_tp1():
    # BUY entry 4458, TP1 4473. Price already 4474 → move done → skip.
    assert decide(_buy(tp1=150), current_price=4474).action == "skip"
    # SELL entry 4555, TP1 4545. Price already 4544 → skip.
    assert decide(_sell(tp1=100), current_price=4544).action == "skip"


def test_place_if_price_between_entry_and_tp1():
    # BUY: price 4465 (between entry 4458 and TP1 4473) → still place the limit.
    assert decide(_buy(tp1=150), 4465).action == "place"


def test_skip_if_price_below_entry_for_long():
    # BUY entry 4458; price already 4450 (below the deep edge) → can't rest a
    # buy-limit above market / pullback already triggered → skip.
    assert decide(_buy(tp1=150), current_price=4450).action == "skip"
    # SELL entry 4555; price already 4560 (above) → skip.
    assert decide(_sell(tp1=100), current_price=4560).action == "skip"


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
