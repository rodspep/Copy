"""Ledger tests — per-SIGNAL summary, daily realized P/L, siblings, schema migration.
Focus on the easy-to-misread interactions (open signal's banked leg vs closed-count;
net vs gross; the day window; legacy single orders)."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

import src.exec.trade_db as db


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    db.DB_PATH = tmp_path / "t.db"
    db.init_db()
    yield


def _ins(**kw):
    base = dict(direction="long", method_pip=50.0, order_type="buy_market",
                entry=4460, sl=4450, tp=4465, volume=0.01)
    base.update(kw)
    return db.insert(base)


# ---- migration ----
def test_migration_adds_columns_preserves_rows(tmp_path):
    old = tmp_path / "old.db"
    c = sqlite3.connect(str(old))
    c.executescript("CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "status TEXT, method_pip REAL, profit REAL, closed_at TEXT);")
    c.execute("INSERT INTO trades (status,method_pip,profit) VALUES ('closed_tp',50,5.0)")
    c.commit(); c.close()
    db.DB_PATH = old
    db.init_db()                      # must ALTER-add leg/group_id without losing the row
    cols = [r[1] for r in sqlite3.connect(str(old)).execute("PRAGMA table_info(trades)")]
    assert "leg" in cols and "group_id" in cols
    assert db.recent(1)[0]["profit"] == 5.0


# ---- siblings ----
def test_siblings():
    _ins(leg="tp1", group_id="g1", status="closed_tp", profit=5)
    _ins(leg="tp3", group_id="g1", status="filled")
    _ins(leg="tp1", group_id="g2", status="pending")
    assert {r["leg"] for r in db.siblings("g1")} == {"tp1", "tp3"}
    assert db.siblings("") == []


# ---- per-signal summary ----
def test_summary_bracket_is_one_signal():
    _ins(leg="tp1", group_id="s1", status="closed_tp", profit=5, closed_at="2026-06-03T08:00:00+00:00")
    _ins(leg="tp3", group_id="s1", status="closed_tp", profit=15, closed_at="2026-06-03T08:30:00+00:00")
    s = db.summary()
    assert s["signals"] == 1 and s["closed"] == 1 and s["wins"] == 1
    assert abs(s["pnl"] - 20) < 1e-9                 # net of the bracket


def test_summary_open_signal_banked_counts_in_pnl_not_in_closed():
    # tp1 won (+5), tp3 still filled (runner open) → signal is OPEN, but +5 is real banked.
    _ins(leg="tp1", group_id="s1", status="closed_tp", profit=5, closed_at="2026-06-03T08:00:00+00:00")
    _ins(leg="tp3", group_id="s1", status="filled")
    s = db.summary()
    assert s["open"] == 1 and s["closed"] == 0       # NOT counted as a closed signal
    assert abs(s["pnl"] - 5) < 1e-9                  # but the banked +5 IS in total P/L
    assert s["by_method"][50.0]["open"] == 1 and abs(s["by_method"][50.0]["pnl"] - 5) < 1e-9


def test_summary_all_cancelled_is_one_cancelled_signal():
    _ins(leg="tp1", group_id="s1", status="cancelled")
    _ins(leg="tp3", group_id="s1", status="cancelled")
    s = db.summary()
    assert s["signals"] == 1 and s["cancelled"] == 1 and s["closed"] == 0


def test_summary_legacy_single_orders_are_own_signals():
    _ins(group_id=None, leg=None, status="closed_tp", profit=5, closed_at="2026-06-03T08:00:00+00:00")
    _ins(group_id=None, leg=None, status="closed_sl", profit=-12, closed_at="2026-06-03T08:00:00+00:00")
    s = db.summary()
    assert s["signals"] == 2 and s["closed"] == 2
    assert s["wins"] == 1 and s["losses"] == 1


def test_summary_net_loss_signal_is_a_loss():
    # bracket: tp1 +5, tp3 -12 (gap through SL before BE) → net -7 → LOSS, not win.
    _ins(leg="tp1", group_id="s1", status="closed_tp", profit=5, closed_at="2026-06-03T08:00:00+00:00")
    _ins(leg="tp3", group_id="s1", status="closed_sl", profit=-12, closed_at="2026-06-03T08:30:00+00:00")
    s = db.summary()
    assert s["closed"] == 1 and s["wins"] == 0 and s["losses"] == 1
    assert abs(s["pnl"] + 7) < 1e-9


# ---- daily realized P/L (circuit-breaker) ----
def test_realized_pnl_since_net_and_window():
    _ins(status="closed_sl", profit=-12, closed_at="2026-06-03T08:00:00+00:00")
    _ins(status="closed_sl", profit=-12, closed_at="2026-06-03T09:00:00+00:00")
    _ins(status="closed_tp", profit=5, closed_at="2026-06-03T10:00:00+00:00")
    _ins(status="closed_sl", profit=-12, closed_at="2026-06-02T10:00:00+00:00")   # yesterday
    _ins(status="filled")                                                          # open, no closed_at
    day = "2026-06-03T00:00:00+00:00"
    assert abs(db.realized_pnl_since(day) - (-19)) < 1e-9     # -12-12+5 (today, net)
    # yesterday's -12 and the open row are excluded
    assert abs(db.realized_pnl_since("2026-06-02T00:00:00+00:00") - (-31)) < 1e-9
