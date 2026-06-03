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
    """Aggregate stats per SIGNAL (a bracket's TP1+TP3 legs count as ONE), overall +
    per method. Legacy single-order rows (no group_id) are each their own signal.

    A signal is: 'open' if any leg is still pending/filled; else 'closed' if it has any
    closed leg; else 'cancelled' (all legs cancelled). Win/loss is judged on the signal's
    NET banked profit (sum of its closed legs). `pnl` is total banked across everything
    (incl. a still-open signal's already-closed leg — it is real money)."""
    with _conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM trades ORDER BY id")]

    groups: dict = {}
    for r in rows:
        key = r["group_id"] or f"single:{r['id']}"
        groups.setdefault(key, []).append(r)

    sigs = []
    for legs in groups.values():
        live = any((l["status"] or "") in ("pending", "filled") for l in legs)
        closed_legs = [l for l in legs if (l["status"] or "").startswith("closed")]
        banked = sum((l["profit"] or 0) for l in closed_legs)
        state = "open" if live else ("closed" if closed_legs else "cancelled")
        sigs.append({"method": legs[0]["method_pip"], "state": state, "banked": banked})

    closed = [s for s in sigs if s["state"] == "closed"]
    wins = [s for s in closed if s["banked"] > 1e-9]
    losses = [s for s in closed if s["banked"] < -1e-9]
    by_method: dict = {}
    for m in sorted({s["method"] for s in sigs if s["method"] is not None}):
        ms = [s for s in sigs if s["method"] == m]
        mc = [s for s in ms if s["state"] == "closed"]
        by_method[m] = {
            "closed": len(mc),
            "wins": sum(1 for s in mc if s["banked"] > 1e-9),
            "open": sum(1 for s in ms if s["state"] == "open"),
            "pnl": sum(s["banked"] for s in ms),
        }
    return {
        "signals": len(sigs),
        "open": sum(1 for s in sigs if s["state"] == "open"),
        "cancelled": sum(1 for s in sigs if s["state"] == "cancelled"),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "winrate": (len(wins) / len(closed)) if closed else 0.0,
        "pnl": sum(s["banked"] for s in sigs),
        "by_method": by_method,
    }
