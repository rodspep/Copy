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
    def __init__(self, require_demo: bool = True):
        from src.data import mt5_feed
        mt5_feed.init()                       # attach to the running terminal
        import MetaTrader5 as mt5
        self.mt5 = mt5
        self.require_demo = require_demo
        acc = mt5.account_info()
        self._login = acc.login if acc else None       # lock the startup account

    def _account_ok(self) -> tuple[bool, str]:
        """Re-verify (every send) that the account hasn't changed and, unless
        explicitly allowed, is still a DEMO account."""
        acc = self.mt5.account_info()
        if acc is None:
            return False, "no account_info"
        if self._login is not None and acc.login != self._login:
            return False, f"account changed {self._login}->{acc.login}"
        if self.require_demo and acc.trade_mode != self.mt5.ACCOUNT_TRADE_MODE_DEMO:
            return False, f"account {acc.login} is NOT demo (require_demo)"
        return True, ""

    def get_price(self, symbol: str):
        if not self.mt5.symbol_select(symbol, True):
            return None
        t = self.mt5.symbol_info_tick(symbol)
        if t is None:
            return None
        return {"bid": float(t.bid), "ask": float(t.ask), "mid": (t.bid + t.ask) / 2.0}

    def _stops_ok(self, symbol: str, o: Order) -> bool:
        """Preflight: SL/TP must respect the broker's min stop distance."""
        info = self.mt5.symbol_info(symbol)
        if info is None:
            return False
        point = info.point or 0.01
        min_dist = (info.trade_stops_level or 0) * point
        if min_dist <= 0:
            return True
        return abs(o.entry - o.sl) >= min_dist and abs(o.tp - o.entry) >= min_dist

    def _find_pending(self, symbol: str, o: Order) -> int | None:
        """Find an already-resting pending that matches this order (for reconcile
        after an ambiguous order_send) — same magic, type, price, volume."""
        orders = self.mt5.orders_get(symbol=symbol)
        if not orders:
            return None
        info = self.mt5.symbol_info(symbol)
        tol = (info.point or 0.01) if info else 0.01     # price tolerance (broker-normalized)
        want = (self.mt5.ORDER_TYPE_BUY_LIMIT if o.order_type == "buy_limit"
                else self.mt5.ORDER_TYPE_SELL_LIMIT)
        for x in orders:
            if (x.magic == MAGIC and x.type == want
                    and abs(x.price_open - o.entry) <= tol
                    and abs(x.volume_current - o.volume) < 1e-9):
                return int(x.ticket)
        return None

    def open_exposure(self, symbol: str) -> int | None:
        """Authoritative concurrent exposure = our pending orders + our open
        positions. None if any query failed (caller should then NOT place)."""
        pend = self.pending_tickets(symbol)
        if pend is None:
            return None
        pos = self.mt5.positions_get(symbol=symbol)
        if pos is None:
            return None
        return len(pend) + sum(1 for p in pos if p.magic == MAGIC)

    def place_limit(self, symbol: str, o: Order) -> int | None:
        mt5 = self.mt5
        ok, why = self._account_ok()             # re-verify account EVERY send
        if not ok:
            print(f"  [mt5] BLOCKED place_limit — {why}")
            return None
        if not self._stops_ok(symbol, o):
            print(f"  [mt5] SL/TP too close to entry (stops_level) — reject {o.entry}")
            return None
        otype = mt5.ORDER_TYPE_BUY_LIMIT if o.order_type == "buy_limit" else mt5.ORDER_TYPE_SELL_LIMIT
        req = {
            "action": mt5.TRADE_ACTION_PENDING, "symbol": symbol,
            "volume": float(o.volume), "type": otype, "price": float(o.entry),
            "sl": float(o.sl), "tp": float(o.tp), "magic": MAGIC,
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_RETURN,
            "comment": COMMENT,
        }
        r = mt5.order_send(req)
        if r is not None and r.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            return int(r.order)
        # Ambiguous/failed: the order MAY still have landed. Reconcile by scanning
        # pendings before declaring failure (avoids an unmanaged orphan).
        print(f"  [mt5] order_send unclear retcode={getattr(r,'retcode',None)} "
              f"{getattr(r,'comment','')} — reconciling...")
        return self._find_pending(symbol, o)

    def pending_tickets(self, symbol: str) -> set[int] | None:
        """Set of our pending tickets, or None if the query FAILED (so the caller
        does NOT mistake an API error for 'all orders gone')."""
        orders = self.mt5.orders_get(symbol=symbol)
        if orders is None:
            return None
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
        self._dry_open: set[int] = set()      # in-memory; never persisted → can't poison live

    def place_limit(self, symbol: str, o: Order) -> int | None:
        self._fake += 1
        self._dry_open.add(self._fake)
        print(f"  [DRY] would place {o.order_type} {symbol} {o.volume} @ {o.entry} "
              f"sl={o.sl} tp={o.tp} (TP1 {o.tp1_pip:g}pip) ticket~{self._fake}")
        return self._fake

    def pending_tickets(self, symbol: str) -> set[int] | None:
        return set(self._dry_open)            # so management/cancel logic runs in dry too

    def open_exposure(self, symbol: str) -> int | None:
        return len(self._dry_open)

    def cancel(self, ticket: int) -> bool:
        self._dry_open.discard(ticket)
        print(f"  [DRY] would cancel ticket {ticket}")
        return True
