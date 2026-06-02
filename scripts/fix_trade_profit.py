"""One-off: recompute stored trade profit from MT5 deal history.

Early copier rows recorded profit == account balance (the closed_info bug where
history_deals_get(from,to,position=X) ignored the position filter and summed ALL
deals incl. the deposit). This re-derives each closed trade's REAL P/L by filtering
deals on position_id, exactly like the fixed closed_info now does.

Safe to re-run: it only rewrites `profit` for rows that have a position_id, using
the authoritative broker history. Prints before/after for every row.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

DB = "data/copier_trades.db"


def main() -> int:
    import MetaTrader5 as m
    m.initialize()
    frm = datetime(2024, 1, 1, tzinfo=timezone.utc)
    to = datetime.now(timezone.utc) + timedelta(days=1)
    deals = m.history_deals_get(frm, to) or []

    def pnl_for(pos_id: int) -> float:
        ds = [d for d in deals if int(d.position_id) == int(pos_id)]
        return round(sum(d.profit + d.swap + d.commission for d in ds), 2)

    c = sqlite3.connect(DB)
    rows = c.execute(
        "select id, position_id, status, profit from trades "
        "where position_id is not None").fetchall()
    for tid, pos, status, old in rows:
        new = pnl_for(pos)
        if old != new:
            c.execute("update trades set profit=? where id=?", (new, tid))
            print(f"ROW id={tid} pos={pos} status={status} profit {old} -> {new}")
        else:
            print(f"ROW id={tid} pos={pos} status={status} profit {old} (unchanged)")
    c.commit()
    c.close()
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
