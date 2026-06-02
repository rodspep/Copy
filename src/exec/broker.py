"""Broker adapters for the UG copier.

Mt5Broker  — real orders via the MetaTrader5 package (Windows/VPS, running terminal).
DryRunBroker — reads REAL prices but never places/cancels; logs intended actions.
              This is the default so you can validate decisions on the live feed
              with zero risk before flipping to --live.

Only `Order`s produced by the unit-tested ug_copier_logic.decide() reach here.
"""
from __future__ import annotations

from src.exec.ug_copier_logic import Order

MAGIC = 770150                  # tags this copier's orders
COMMENT = "ug_copier"


class Mt5Broker:
    def __init__(self):
        from src.data import mt5_feed
        mt5_feed.init()                       # attach to the running terminal
        import MetaTrader5 as mt5
        self.mt5 = mt5

    def get_price(self, symbol: str):
        if not self.mt5.symbol_select(symbol, True):
            return None
        t = self.mt5.symbol_info_tick(symbol)
        if t is None:
            return None
        return {"bid": float(t.bid), "ask": float(t.ask), "mid": (t.bid + t.ask) / 2.0}

    def place_limit(self, symbol: str, o: Order) -> int | None:
        mt5 = self.mt5
        otype = mt5.ORDER_TYPE_BUY_LIMIT if o.order_type == "buy_limit" else mt5.ORDER_TYPE_SELL_LIMIT
        req = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": float(o.volume),
            "type": otype,
            "price": float(o.entry),
            "sl": float(o.sl),
            "tp": float(o.tp),
            "magic": MAGIC,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
            "comment": COMMENT,
        }
        r = mt5.order_send(req)
        if r is None or r.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            print(f"  [mt5] order_send FAILED retcode={getattr(r,'retcode',None)} "
                  f"{getattr(r,'comment','')}")
            return None
        return int(r.order)

    def pending_tickets(self, symbol: str) -> set[int]:
        orders = self.mt5.orders_get(symbol=symbol) or ()
        return {int(o.ticket) for o in orders if o.magic == MAGIC}

    def cancel(self, ticket: int) -> bool:
        r = self.mt5.order_send({"action": self.mt5.TRADE_ACTION_REMOVE, "order": int(ticket)})
        ok = r is not None and r.retcode == self.mt5.TRADE_RETCODE_DONE
        if not ok:
            print(f"  [mt5] cancel FAILED ticket={ticket} retcode={getattr(r,'retcode',None)}")
        return ok


class DryRunBroker(Mt5Broker):
    """Real prices, NO order placement. Synthetic tickets so the loop can track."""

    def __init__(self):
        super().__init__()
        self._fake = 9_000_000

    def place_limit(self, symbol: str, o: Order) -> int | None:
        self._fake += 1
        print(f"  [DRY] would place {o.order_type} {symbol} {o.volume} @ {o.entry} "
              f"sl={o.sl} tp={o.tp} (TP1 {o.tp1_pip:g}pip) ticket~{self._fake}")
        return self._fake

    def pending_tickets(self, symbol: str) -> set[int]:
        return set()                          # dry-run keeps no real pendings

    def cancel(self, ticket: int) -> bool:
        print(f"  [DRY] would cancel ticket {ticket}")
        return True
