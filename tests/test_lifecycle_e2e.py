"""END-TO-END lifecycle test of the REAL management orchestration (_manage_open):
place bracket → fill → TP1 close → runner BE → runner close, asserting DB state,
P/L (summary + realized) exactly once. This is the test class that catches bugs in the
INTERACTIONS between decide/broker/trade_db (where the whole-bot review found most bugs).
"""
from __future__ import annotations

import threading

import pytest

import scripts.ug_copier as cp
import src.exec.trade_db as db
import src.exec.notify as notify
from src.exec.broker import Mt5Broker
from src.exec.ug_copier_logic import Order
from tests.fake_mt5 import FakeMt5

SYM = "XAUUSDm"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    db.DB_PATH = tmp_path / "t.db"
    db.init_db()
    monkeypatch.setattr(notify, "send", lambda *a, **k: 1)
    monkeypatch.setattr(cp, "ACCOUNT_LABEL", "DEMO", raising=False)
    yield


def _broker(fake):
    b = Mt5Broker.__new__(Mt5Broker)
    b.mt5 = fake
    b._lock = threading.RLock()
    b._login = fake.acc.login
    b.require_demo = True
    return b


def _now():
    return "2026-06-03T08:00:00+00:00"


def _place_bracket(broker, fake, mid=4460.0):
    """Mirror the copier's placement of a 50pip limit bracket (TP1+TP3) into trade_db."""
    legs = [("tp1", 4465.0), ("tp3", 4475.0)]      # TP1 +5, TP3 +15 (price); SL 4450
    group = "long|4468|4458|4450|50"
    for leg, tp in legs:
        o = Order(side="long", order_type="buy_limit", entry=mid, sl=4450.0, tp=tp,
                  volume=0.01, tp1_pip=50, leg=leg, tp_pip=(50 if leg == "tp1" else 150))
        tk = broker.place_limit(SYM, o)
        db.insert({"signal_ts": "x", "direction": "long", "method_pip": 50.0,
                   "order_type": "buy_limit", "entry": mid, "sl": 4450.0, "tp": tp,
                   "volume": 0.01, "ticket": tk, "status": "pending",
                   "created_at": _now(), "leg": leg, "group_id": group})
    return group


def test_full_lifecycle_win_tp1_then_runner_tp3():
    fake = FakeMt5(bid=4460.0, ask=4460.0)
    b = _broker(fake)
    _place_bracket(b, fake)
    tickets = [int(o.ticket) for o in fake._orders]
    assert len(tickets) == 2

    # poll 1: both still pending (orders resting, no fill) → no change
    cp._manage_open(b, SYM, 4460.0, _now, 240)
    assert {r["status"] for r in db.open_trades()} == {"pending"}

    # both limits fill (price pulled back to entry)
    for tk in tickets:
        fake.fill_pending(tk, 4460.0)
    cp._manage_open(b, SYM, 4460.0, _now, 240)
    assert {r["status"] for r in db.open_trades()} == {"filled"}

    # TP1 leg hits TP1 (+5) → closes; runner SL must move to BE (entry 4460)
    tp1_pid = tickets[0]
    fake.close_position(tp1_pid, out_price=4465.0, profit=5.0)
    cp._manage_open(b, SYM, 4465.0, _now, 240)
    rows = {r["leg"]: r for r in db.recent(5)}
    assert rows["tp1"]["status"] == "closed_tp"
    runner = next(p for p in fake._positions if p.ticket == tickets[1])
    assert runner.sl == 4460.0                      # BE moved to entry
    assert db.siblings(rows["tp1"]["group_id"])      # group intact

    # runner runs to TP3 (+15) → closes
    fake.close_position(tickets[1], out_price=4475.0, profit=15.0)
    cp._manage_open(b, SYM, 4475.0, _now, 240)

    s = db.summary()
    assert s["signals"] == 1 and s["closed"] == 1 and s["wins"] == 1
    assert abs(s["pnl"] - 20.0) < 1e-9              # +5 (TP1) +15 (TP3), counted once
    assert abs(db.realized_pnl_since("2026-06-03T00:00:00+00:00") - 20.0) < 1e-9


def test_full_lifecycle_loss_both_legs_sl_no_be():
    fake = FakeMt5(bid=4460.0, ask=4460.0)
    b = _broker(fake)
    _place_bracket(b, fake)
    tickets = [int(o.ticket) for o in fake._orders]
    for tk in tickets:
        fake.fill_pending(tk, 4460.0)
    cp._manage_open(b, SYM, 4460.0, _now, 240)
    # both legs hit SL (price dropped through 4450) BEFORE TP1 → both closed_sl, NO BE
    fake.close_position(tickets[0], out_price=4450.0, profit=-10.0)
    fake.close_position(tickets[1], out_price=4450.0, profit=-10.0)
    cp._manage_open(b, SYM, 4450.0, _now, 240)
    s = db.summary()
    assert s["closed"] == 1 and s["wins"] == 0 and s["losses"] == 1
    assert abs(s["pnl"] + 20.0) < 1e-9             # -10 -10


def test_just_filled_pending_not_marked_cancelled():
    # the critical bug: a pending that FILLED (position exists) but whose IN deal hasn't
    # hit history must NOT be marked 'cancelled/vanished'.
    fake = FakeMt5(bid=4460.0, ask=4460.0)
    b = _broker(fake)
    _place_bracket(b, fake)
    tk = int(fake._orders[0].ticket)
    fake.fill_pending(tk, 4460.0)
    fake.deals_get_none = True                      # simulate history lag (deals query empty/fail)
    cp._manage_open(b, SYM, 4460.0, _now, 240)
    row = next(r for r in db.recent(5) if r["ticket"] == tk)
    assert row["status"] == "filled"               # positions-first → filled, NOT cancelled
