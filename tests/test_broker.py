"""Stateful broker tests against a controllable FakeMt5 — covers the real-world
interactions/confusions that bit us (and that static review can miss). No real terminal.
"""
from __future__ import annotations

import threading

from src.exec.broker import Mt5Broker, MAGIC
from src.exec.ug_copier_logic import Order
from tests.fake_mt5 import FakeMt5, ORDER_TYPE_BUY_LIMIT, TRADE_RETCODE_DONE


def make_broker(fake=None, login=433674415, require_demo=True):
    """Build an Mt5Broker bypassing the real MT5 __init__, wired to a FakeMt5."""
    fake = fake or FakeMt5(login=login)
    b = Mt5Broker.__new__(Mt5Broker)
    b.mt5 = fake
    b._lock = threading.RLock()
    b._login = login
    b.require_demo = require_demo
    return b, fake


def _lim(side="long", entry=4460.0, sl=4450.0, tp=4465.0, leg="tp1", anchor=0.0):
    ot = "buy_limit" if side == "long" else "sell_limit"
    return Order(side=side, order_type=ot, entry=entry, sl=sl, tp=tp, volume=0.01,
                 tp1_pip=50, leg=leg, tp_pip=50, anchor=anchor)


def _mkt(side="long", entry=4460.0, sl=4450.0, tp=4465.0, anchor=4461.0):
    ot = "buy_market" if side == "long" else "sell_market"
    return Order(side=side, order_type=ot, entry=entry, sl=sl, tp=tp, volume=0.01,
                 tp1_pip=50, leg="tp1", tp_pip=50, anchor=anchor)


# ---- place_limit ----
def test_place_limit_done_returns_ticket():
    b, f = make_broker()
    tk = b.place_limit("XAUUSDm", _lim())
    assert tk is not None and len(f._orders) == 1 and f._orders[0].ticket == tk


def test_place_limit_unclear_reconciles_only_matching_leg():
    # Two resting pendings, same entry/vol/type, DIFFERENT tp (the two bracket legs).
    # An unclear send for the tp3 leg must reconcile to the tp3 ticket, NOT the tp1 one.
    b, f = make_broker()
    f._tk = 5000
    b.place_limit("XAUUSDm", _lim(tp=4465.0))   # tp1 leg
    b.place_limit("XAUUSDm", _lim(tp=4475.0))   # tp3 leg (further)
    tp1_tk = f._orders[0].ticket
    tp3_tk = f._orders[1].ticket
    f.next_retcode = 10004                       # unclear/fail on the next send
    got = b.place_limit("XAUUSDm", _lim(tp=4475.0))   # re-attempt tp3 → reconcile
    assert got == tp3_tk and got != tp1_tk


def test_place_limit_account_changed_blocked():
    b, f = make_broker()
    f.acc.login = 999999                          # account switched
    assert b.place_limit("XAUUSDm", _lim()) is None and not f._orders


# ---- place_market ----
def test_place_market_resolves_position_id_and_fill_from_deal():
    b, f = make_broker()
    f.market_position_id = 777                    # position id != order ticket
    f.market_fill_price = 4460.28                 # actual fill != requested
    res = b.place_market("XAUUSDm", _mkt(anchor=4461.0))
    assert res is not None
    assert res["position_id"] == 777              # resolved from the DEAL, not r.order
    assert abs(res["fill_price"] - 4460.28) < 1e-9


def test_place_market_rejects_worse_than_anchor():
    # buy anchor 4460; ask 4460.3 > anchor → must NOT chase → None, no position.
    b, f = make_broker(FakeMt5(bid=4460.0, ask=4460.3))
    assert b.place_market("XAUUSDm", _mkt(anchor=4460.0)) is None
    assert not f._positions


def test_place_market_rejects_wrong_side_of_sl_tp():
    b, f = make_broker(FakeMt5(bid=4466.0, ask=4466.2))   # ask above tp 4465
    assert b.place_market("XAUUSDm", _mkt(anchor=4470.0)) is None


# ---- account-lock on management sends ----
def test_cancel_blocked_on_account_change():
    b, f = make_broker()
    b.place_limit("XAUUSDm", _lim())
    tk = f._orders[0].ticket
    f.acc.login = 999999
    assert b.cancel(tk) is False and len(f._orders) == 1   # not removed


def test_modify_sl_blocked_on_account_change_else_ok():
    b, f = make_broker()
    res = b.place_market("XAUUSDm", _mkt(anchor=4461.0))
    pid = res["position_id"]
    f.acc.login = 999999
    assert b.modify_sl(pid, 4460.0) is False               # blocked
    f.acc.login = 433674415
    assert b.modify_sl(pid, 4460.0) is True                # restored → works
    assert next(p for p in f._positions if p.ticket == pid).sl == 4460.0


# ---- fail-closed reads on a definite login change ----
def test_reads_fail_closed_on_account_change():
    b, f = make_broker()
    b.place_limit("XAUUSDm", _lim())
    f.acc.login = 999999
    assert b.pending_tickets("XAUUSDm") is None
    assert b.list_magic("XAUUSDm") is None
    assert b.closed_info(777) is None
    assert b.fill_info(123) == "unknown"


def test_login_changed_semantics():
    b, f = make_broker(login=111)
    assert b.login_changed() is False             # same
    f.acc.login = 222
    assert b.login_changed() is True              # definite change
    f.acc.login, f.acc_none = 222, True           # transient None
    assert b.login_changed() is False             # transient is NOT a change


# ---- closed_info: the +10010 deposit-deal trap + partial close ----
def test_closed_info_ignores_balance_deal():
    b, f = make_broker()
    res = b.place_market("XAUUSDm", _mkt(anchor=4461.0))
    pid = res["position_id"]
    f.add_balance_deal(10000.0)                   # deposit deal (position_id 0) — must NOT sum
    f.close_position(pid, out_price=4465.0, profit=5.0)
    ci = b.closed_info(pid)
    assert ci is not None and abs(ci["profit"] - 5.0) < 1e-9   # 5, not 10005
    assert abs(ci["close_price"] - 4465.0) < 1e-9


def test_closed_info_partial_returns_none():
    b, f = make_broker()
    res = b.place_market("XAUUSDm", _mkt(anchor=4461.0))   # IN volume 0.01
    pid = res["position_id"]
    f.close_position(pid, out_price=4465.0, profit=2.5, volume=0.005)  # only half closed
    assert b.closed_info(pid) is None             # still partially open → keep tracking


def test_closed_info_open_returns_none():
    b, f = make_broker()
    res = b.place_market("XAUUSDm", _mkt(anchor=4461.0))
    assert b.closed_info(res["position_id"]) is None   # no OUT deal yet


# ---- fill_info: transient vs confirmed ----
def test_fill_info_states():
    b, f = make_broker()
    res = b.place_market("XAUUSDm", _mkt(anchor=4461.0))
    pid = res["position_id"]
    # the IN deal exists; fill_info by the deal's order ticket
    in_deal = next(d for d in f._deals if d.position_id == pid)
    fi = b.fill_info(in_deal.order)
    assert fi and fi["position_id"] == pid        # position exists (real-time) → filled
    assert b.fill_info(424242) is None            # no position, no deal → confirmed not-filled
    f.positions_get_none = True
    assert b.fill_info(in_deal.order) == "unknown"   # positions query failed → don't conclude


def test_fill_info_from_history_when_position_already_closed():
    # filled then closed → no live position, but the IN deal is in history → still 'filled'
    # (not a false 'cancelled'); this is the history-lag/closed-fast safety.
    b, f = make_broker()
    res = b.place_market("XAUUSDm", _mkt(anchor=4461.0))
    in_deal = next(d for d in f._deals if d.position_id == res["position_id"])
    f.close_position(res["position_id"], out_price=4465.0, profit=5.0)   # position gone
    fi = b.fill_info(in_deal.order)
    assert fi and fi["position_id"] == res["position_id"]


# ---- exposure ----
def test_open_exposure_counts_and_failclosed():
    b, f = make_broker()
    b.place_limit("XAUUSDm", _lim())
    b.place_market("XAUUSDm", _mkt(anchor=4461.0))
    assert b.open_exposure("XAUUSDm") == 2        # 1 pending + 1 position
    f.orders_get_none = True
    assert b.open_exposure("XAUUSDm") is None     # query failed → None (don't place)
