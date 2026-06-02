"""Self-test the copier's broker on the CURRENT MT5 account (use a DEMO account!).

Places ONE real pending BUY-LIMIT well below market (so it rests, won't fill),
confirms it appears, waits, cancels it, confirms it's gone. Proves place + cancel
work end-to-end before trusting the copier with real signals. ~15s, one command.

Tagged with the copier's MAGIC so it only ever touches its own order.

Run inside the MT5 session (via run_copier_selftest.bat):
  python -X utf8 -m scripts.ug_copier_selftest --symbol XAUUSDm
"""
from __future__ import annotations

import argparse
import time

from src.exec.broker import Mt5Broker, MAGIC
from src.exec.ug_copier_logic import Order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSDm")
    ap.add_argument("--volume", type=float, default=0.01)
    args = ap.parse_args()

    b = Mt5Broker()
    px = b.get_price(args.symbol)
    if not px:
        print(f"FAIL: no price for {args.symbol}")
        return 1
    mid = px["mid"]
    # Resting buy-limit ~10 below market: entry<ask (valid limit), won't fill.
    entry = round(mid - 10.0, 2)
    o = Order(side="long", order_type="buy_limit", entry=entry,
              sl=round(entry - 10.0, 2), tp=round(entry + 15.0, 2),
              volume=args.volume, tp1_pip=150.0)
    print(f"price mid={mid} · test order: buy_limit {args.symbol} {o.volume} "
          f"@ {o.entry} sl={o.sl} tp={o.tp} (magic {MAGIC})")

    ticket = b.place_limit(args.symbol, o)
    if ticket is None:
        print("FAIL: place_limit returned None (order not placed)")
        return 1
    print(f"  placed ticket={ticket}")
    time.sleep(3)

    pend = b.pending_tickets(args.symbol)
    if pend is None or ticket not in pend:
        print(f"FAIL: ticket {ticket} not found in pendings {pend} after place")
        return 1
    print(f"  confirmed pending present: {pend}")
    time.sleep(3)

    if not b.cancel(ticket):
        print(f"FAIL: cancel({ticket}) failed")
        return 1
    time.sleep(3)
    pend2 = b.pending_tickets(args.symbol)
    if pend2 is None or ticket in pend2:
        print(f"FAIL: ticket {ticket} still present after cancel: {pend2}")
        return 1
    print(f"  confirmed cancelled: pendings now {pend2}")
    print("PASS: place + cancel both work on this account.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
