"""Emergency flatten: cancel ALL pending orders tagged with the copier's MAGIC.

A safety/ops tool for the live copier — one command to clear all our resting
limit orders (e.g. after a bad test, or to stop everything). Only touches orders
with MAGIC 770150, so it never affects the bot, volscan, or manual trades. Open
POSITIONS are reported but NOT auto-closed (closing a position is a deliberate
act — do it yourself in MT5 if needed).

Run in the MT5 session (via run_cancel_all.bat):
  python -X utf8 -m scripts.ug_cancel_all
"""
from __future__ import annotations

from src.exec.broker import Mt5Broker, MAGIC


def main() -> int:
    b = Mt5Broker()
    acc = b.mt5.account_info()
    if acc is not None:
        demo = acc.trade_mode == b.mt5.ACCOUNT_TRADE_MODE_DEMO
        print(f"account {acc.login} · {acc.server} · {'DEMO' if demo else 'REAL'}")
    orders = b.mt5.orders_get() or []
    mine = [o for o in orders if o.magic == MAGIC]
    print(f"{len(mine)} pending order(s) with magic {MAGIC}")
    done = 0
    for o in mine:
        r = b.mt5.order_send({"action": b.mt5.TRADE_ACTION_REMOVE, "order": int(o.ticket)})
        ok = r is not None and r.retcode == b.mt5.TRADE_RETCODE_DONE
        print(f"  cancel {o.ticket} @ {o.price_open}: "
              f"{'OK' if ok else 'FAIL ' + str(getattr(r, 'retcode', None))}")
        done += int(ok)
    pos = b.mt5.positions_get() or []
    myp = [p for p in pos if p.magic == MAGIC]
    if myp:
        print(f"  NOTE: {len(myp)} OPEN position(s) with magic {MAGIC} — NOT closed "
              f"(close manually in MT5 if you want them flat):")
        for p in myp:
            print(f"    ticket {p.ticket} {('BUY' if p.type == 0 else 'SELL')} "
                  f"{p.volume} @ {p.price_open} sl={p.sl} tp={p.tp}")
    print(f"cancelled {done}/{len(mine)} pending(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
