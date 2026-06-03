"""Orphan-recovery tests (crash between order send and DB insert). Focus on the
easy-to-misread case: a tracked PENDING that FILLED while the copier was offline shows
up only as a POSITION (whose id == the original order ticket) and must NOT be adopted
as a new orphan (it'll resolve via fill_info)."""
from __future__ import annotations

import pytest

import scripts.ug_copier as cp
import src.exec.trade_db as db
import src.exec.notify as notify


class StubBroker:
    def __init__(self, info):
        self._info = info

    def list_magic(self, symbol):
        return self._info


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    db.DB_PATH = tmp_path / "t.db"
    db.init_db()
    monkeypatch.setattr(notify, "send", lambda *a, **k: None)
    monkeypatch.setattr(cp, "ACCOUNT_LABEL", "DEMO", raising=False)
    yield


def _info(pendings=(), positions=()):
    return {"pendings": list(pendings), "positions": list(positions),
            "buy_limit": 2, "pos_buy": 0}


def test_adopt_pending_and_position():
    info = _info(
        pendings=[{"ticket": 111, "type": 2, "entry": 4400, "sl": 4390, "tp": 4405, "volume": 0.01}],
        positions=[{"position_id": 222, "type": 0, "entry": 4400, "sl": 4390, "tp": 4405,
                    "volume": 0.01, "fill_price": 4400}])
    cp._adopt_orphans(StubBroker(info), "XAUUSDm")
    rows = {r["ticket"]: r for r in db.open_trades()}
    assert rows[111]["status"] == "pending" and rows[111]["leg"] == "orphan"
    assert rows[222]["status"] == "filled" and rows[222]["position_id"] == 222
    assert rows[111]["direction"] == "long" and rows[222]["direction"] == "long"


def test_adopt_idempotent():
    info = _info(pendings=[{"ticket": 111, "type": 2, "entry": 4400, "sl": 4390, "tp": 4405, "volume": 0.01}])
    cp._adopt_orphans(StubBroker(info), "XAUUSDm")
    cp._adopt_orphans(StubBroker(info), "XAUUSDm")     # 2nd run must not duplicate
    assert len(db.open_trades()) == 1


def test_filled_while_offline_not_duplicated():
    # tracked PENDING ticket 500 (no position_id yet); it filled while offline → broker now
    # shows a POSITION whose id == 500. Must NOT be adopted as a separate orphan.
    db.insert({"direction": "long", "order_type": "buy_limit", "entry": 4400, "sl": 4390,
               "tp": 4405, "volume": 0.01, "ticket": 500, "status": "pending",
               "leg": "tp1", "group_id": "g1"})
    info = _info(positions=[{"position_id": 500, "type": 0, "entry": 4400, "sl": 4390,
                             "tp": 4405, "volume": 0.01, "fill_price": 4400}])
    cp._adopt_orphans(StubBroker(info), "XAUUSDm")
    assert len(db.open_trades()) == 1                  # still just the original row


def test_list_magic_none_skips_safely():
    cp._adopt_orphans(StubBroker(None), "XAUUSDm")     # query failed → no crash, no rows
    assert db.open_trades() == []


def test_key_canonical_and_method_distinct():
    base = {"direction": "long", "entry_low": 4468, "entry_high": 4458, "sl": 4448}
    k50a = cp._key({**base, "tps_pip": {1: 50}})
    k50b = cp._key({**base, "tps_pip": {1: 50.0}})
    k50c = cp._key({**base, "tps_pip": {"1": "50"}})
    assert k50a == k50b == k50c                       # 50/50.0/'50' dedupe identically
    assert k50a != cp._key({**base, "tps_pip": {1: 150}})   # 50 vs 150 are distinct signals
    # same key doubles as group_id → a signal's two legs (same TP1) group together
    assert cp._key({**base, "tps_pip": {1: 50, 3: 150}}) == k50a


def test_short_position_direction_inference():
    info = _info(positions=[{"position_id": 333, "type": 1, "entry": 4500, "sl": 4510,
                             "tp": 4495, "volume": 0.01, "fill_price": 4500}])
    cp._adopt_orphans(StubBroker(info), "XAUUSDm")
    assert db.open_trades()[0]["direction"] == "short"
