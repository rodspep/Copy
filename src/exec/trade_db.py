"""SQLite trade ledger for the UG copier — full lifecycle + P/L for /stats.

One row per placed order: pending → filled → closed (tp/sl/other) | cancelled.
Stores the Telegram message_id so closes can reply to the original placement.
Pure sqlite, no MT5 — the copier writes it; a /stats reader reads it.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("data/copier_trades.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_ts    TEXT,
    direction    TEXT,            -- long | short
    method_pip   REAL,            -- TP1 pip (50/100/150) = which UG method
    order_type   TEXT,            -- buy_limit | sell_limit
    entry        REAL,
    sl           REAL,
    tp           REAL,
    volume       REAL,
    ticket       INTEGER,         -- pending order ticket
    position_id  INTEGER,         -- position ticket after fill
    status       TEXT,            -- pending|filled|cancelled|closed_tp|closed_sl|closed_other
    fill_price   REAL,
    close_price  REAL,
    profit       REAL,            -- account-currency P/L on close
    tg_msg_id    INTEGER,
    created_at   TEXT,
    filled_at    TEXT,
    closed_at    TEXT,
    note         TEXT,
    leg          TEXT,            -- 'tp1' (scalp) | 'tp3' (runner) — bracket leg
    group_id     TEXT             -- shared key linking the legs of one signal
);
"""

# Columns added after the original schema shipped — ALTER existing DBs in place.
_MIGRATIONS = (("leg", "TEXT"), ("group_id", "TEXT"))


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript(_SCHEMA)
        have = {r["name"] for r in c.execute("PRAGMA table_info(trades)")}
        for col, typ in _MIGRATIONS:
            if col not in have:
                c.execute(f"ALTER TABLE trades ADD COLUMN {col} {typ}")


def insert(rec: dict) -> int:
    cols = ("signal_ts", "direction", "method_pip", "order_type", "entry", "sl", "tp",
            "volume", "ticket", "position_id", "status", "fill_price", "close_price",
            "profit", "tg_msg_id", "created_at", "filled_at", "closed_at", "note",
            "leg", "group_id")
    vals = [rec.get(k) for k in cols]
    with _conn() as c:
        cur = c.execute(
            f"INSERT INTO trades ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)
        return int(cur.lastrowid)


def update(trade_id: int, **fields) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE trades SET {sets} WHERE id=?", [*fields.values(), trade_id])


def open_trades() -> list[sqlite3.Row]:
    """Pending or filled (still live)."""
    with _conn() as c:
        return c.execute("SELECT * FROM trades WHERE status IN ('pending','filled') "
                         "ORDER BY id").fetchall()


def siblings(group_id: str) -> list[sqlite3.Row]:
    """All legs sharing a group_id (the two bracket legs of one signal)."""
    if not group_id:
        return []
    with _conn() as c:
        return c.execute("SELECT * FROM trades WHERE group_id=? ORDER BY id",
                         (group_id,)).fetchall()


def recent(n: int = 10) -> list[sqlite3.Row]:
    with _conn() as c:
        return c.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (n,)).fetchall()


def summary() -> dict:
    """Aggregate stats overall + per method."""
    with _conn() as c:
        rows = c.execute("SELECT * FROM trades").fetchall()
    closed = [r for r in rows if r["status"] and r["status"].startswith("closed")]
    wins = [r for r in closed if (r["profit"] or 0) > 0]
    pnl = sum((r["profit"] or 0) for r in closed)
    by_method: dict = {}
    for m in sorted({r["method_pip"] for r in rows if r["method_pip"] is not None}):
        mc = [r for r in closed if r["method_pip"] == m]
        mw = [r for r in mc if (r["profit"] or 0) > 0]
        by_method[m] = {"closed": len(mc), "wins": len(mw),
                        "pnl": sum((r["profit"] or 0) for r in mc)}
    return {
        "total": len(rows),
        "pending": sum(1 for r in rows if r["status"] == "pending"),
        "filled": sum(1 for r in rows if r["status"] == "filled"),
        "cancelled": sum(1 for r in rows if r["status"] == "cancelled"),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(closed) - len(wins),
        "winrate": (len(wins) / len(closed)) if closed else 0.0,
        "pnl": pnl,
        "by_method": by_method,
    }
