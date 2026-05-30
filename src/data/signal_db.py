"""SQLite store for live signals — full record + outcome tracking.

Every signal the bot sends is persisted with all the info needed to compare the
live track record against the backtest: entry/SL/TP, R:R, source price, and —
once resolved — whether it hit TP or SL and the realized R. Designed to live on
a Railway volume (set DB_PATH=/data/signals.db) so history survives redeploys.

Pure stdlib (sqlite3) — no extra dependency.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def db_path() -> str:
    return os.environ.get("DB_PATH", "data/signals.db")


def _conn(path: str | None = None) -> sqlite3.Connection:
    p = Path(path or db_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(p))
    c.row_factory = sqlite3.Row
    return c


def init_db(path: str | None = None) -> None:
    c = _conn(path)
    c.execute(
        """CREATE TABLE IF NOT EXISTS signals(
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            sent_at       TEXT,
            symbol        TEXT,
            source        TEXT,
            strategy      TEXT,
            timeframe     TEXT,
            direction     TEXT,
            signal_bar_ts TEXT,
            entry         REAL,
            sl            REAL,
            tp            REAL,
            risk_dist     REAL,
            reward_dist   REAL,
            rr            REAL,
            source_price  REAL,
            price_offset  REAL,
            status        TEXT DEFAULT 'open',
            closed_at     TEXT,
            exit_price    REAL,
            result_r      REAL,
            UNIQUE(timeframe, signal_bar_ts)
        )"""
    )
    c.commit()
    c.close()


def exists(timeframe: str, bar_ts: str, path: str | None = None) -> bool:
    c = _conn(path)
    r = c.execute(
        "SELECT 1 FROM signals WHERE timeframe=? AND signal_bar_ts=?",
        (timeframe, bar_ts),
    ).fetchone()
    c.close()
    return r is not None


def insert_signal(rec: dict, path: str | None = None) -> int | None:
    """Insert a new signal. Returns row id, or None if it was a duplicate."""
    cols = ["sent_at", "symbol", "source", "strategy", "timeframe", "direction",
            "signal_bar_ts", "entry", "sl", "tp", "risk_dist", "reward_dist",
            "rr", "source_price", "price_offset", "status"]
    vals = [rec.get(k) for k in cols]
    c = _conn(path)
    try:
        cur = c.execute(
            f"INSERT INTO signals ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            vals,
        )
        c.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        c.close()


def open_signals(path: str | None = None) -> list[sqlite3.Row]:
    c = _conn(path)
    rows = c.execute("SELECT * FROM signals WHERE status='open' ORDER BY id").fetchall()
    c.close()
    return rows


def close_signal(sig_id: int, closed_at: str, exit_price: float,
                 result_r: float, status: str, path: str | None = None) -> None:
    c = _conn(path)
    c.execute(
        "UPDATE signals SET status=?, closed_at=?, exit_price=?, result_r=? WHERE id=?",
        (status, closed_at, exit_price, result_r, sig_id),
    )
    c.commit()
    c.close()


def recent(n: int = 20, path: str | None = None) -> list[sqlite3.Row]:
    c = _conn(path)
    rows = c.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (n,)).fetchall()
    c.close()
    return rows


def summary(path: str | None = None) -> dict:
    c = _conn(path)
    row = c.execute(
        """SELECT
             COUNT(*)                                            AS total,
             SUM(status='open')                                 AS open_n,
             SUM(status='win')                                  AS wins,
             SUM(status='loss')                                 AS losses,
             COALESCE(SUM(result_r), 0)                         AS sum_r,
             COALESCE(AVG(CASE WHEN status IN ('win','loss')
                               THEN result_r END), 0)           AS exp_r
           FROM signals"""
    ).fetchone()
    c.close()
    d = dict(row)
    closed = (d["wins"] or 0) + (d["losses"] or 0)
    d["winrate"] = (d["wins"] or 0) / closed if closed else 0.0
    return d
