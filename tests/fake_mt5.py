"""A controllable fake of the MetaTrader5 module surface the broker uses, so the
stateful broker logic can be tested deterministically WITHOUT a real terminal.

Models the real-world subtleties that have bitten us (and that are easy to misread):
  - a filled market order's POSITION id can differ from the order ticket (resolved via
    the deal), and the actual fill price differs from the requested price;
  - history_deals_get(from,to,position=X) IGNORES the position filter (returns ALL deals
    incl. the account's balance/deposit deal) — closed_info must filter itself;
  - account_info() can transiently return None; the login can change (account switch).
Set attributes / call helpers to drive each scenario.
"""
from __future__ import annotations

from types import SimpleNamespace

# --- constants the broker references ---
TRADE_ACTION_PENDING = 5
TRADE_ACTION_DEAL = 1
TRADE_ACTION_REMOVE = 2
TRADE_ACTION_SLTP = 6
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TYPE_BUY_LIMIT = 2
ORDER_TYPE_SELL_LIMIT = 3
ORDER_TIME_GTC = 0
ORDER_FILLING_RETURN = 2
ORDER_FILLING_IOC = 1
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_PLACED = 10008
ACCOUNT_TRADE_MODE_DEMO = 0
ACCOUNT_TRADE_MODE_REAL = 2
ACCOUNT_MARGIN_MODE_RETAIL_NETTING = 0
ACCOUNT_MARGIN_MODE_EXCHANGE = 1
ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = 2
POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1
DEAL_ENTRY_IN = 0
DEAL_ENTRY_OUT = 1


class FakeMt5:
    def __init__(self, login=433674415, trade_mode=ACCOUNT_TRADE_MODE_DEMO,
                 server="Exness-MT5Trial7", bid=4460.0, ask=4460.3,
                 point=0.001, stops_level=0,
                 margin_mode=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING):
        self.acc = SimpleNamespace(login=login, trade_mode=trade_mode, server=server,
                                   balance=10000.0, margin_mode=margin_mode)
        self.acc_none = False              # set True to simulate transient account_info None
        self.bid, self.ask = bid, ask
        self.point, self.stops_level = point, stops_level
        self._orders, self._positions, self._deals = [], [], []
        self._tk = 1000
        self.next_retcode = TRADE_RETCODE_DONE   # override to simulate failure/unclear
        # for market: make the deal's position_id DIFFERENT from the order ticket + give a
        # distinct fill price, to prove the broker resolves them from the deal (not r.order).
        self.market_position_id = None     # None => position_id = order ticket
        self.market_fill_price = None      # None => fill = requested price
        self.orders_get_none = False
        self.positions_get_none = False
        self.deals_get_none = False

    # mirror constants as attributes (broker uses self.mt5.CONST)
    TRADE_ACTION_PENDING = TRADE_ACTION_PENDING
    TRADE_ACTION_DEAL = TRADE_ACTION_DEAL
    TRADE_ACTION_REMOVE = TRADE_ACTION_REMOVE
    TRADE_ACTION_SLTP = TRADE_ACTION_SLTP
    ORDER_TYPE_BUY = ORDER_TYPE_BUY
    ORDER_TYPE_SELL = ORDER_TYPE_SELL
    ORDER_TYPE_BUY_LIMIT = ORDER_TYPE_BUY_LIMIT
    ORDER_TYPE_SELL_LIMIT = ORDER_TYPE_SELL_LIMIT
    ORDER_TIME_GTC = ORDER_TIME_GTC
    ORDER_FILLING_RETURN = ORDER_FILLING_RETURN
    ORDER_FILLING_IOC = ORDER_FILLING_IOC
    TRADE_RETCODE_DONE = TRADE_RETCODE_DONE
    TRADE_RETCODE_PLACED = TRADE_RETCODE_PLACED
    ACCOUNT_TRADE_MODE_DEMO = ACCOUNT_TRADE_MODE_DEMO
    ACCOUNT_TRADE_MODE_REAL = ACCOUNT_TRADE_MODE_REAL
    ACCOUNT_MARGIN_MODE_RETAIL_NETTING = ACCOUNT_MARGIN_MODE_RETAIL_NETTING
    ACCOUNT_MARGIN_MODE_EXCHANGE = ACCOUNT_MARGIN_MODE_EXCHANGE
    ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = ACCOUNT_MARGIN_MODE_RETAIL_HEDGING
    POSITION_TYPE_BUY = POSITION_TYPE_BUY
    POSITION_TYPE_SELL = POSITION_TYPE_SELL
    DEAL_ENTRY_IN = DEAL_ENTRY_IN
    DEAL_ENTRY_OUT = DEAL_ENTRY_OUT

    # --- queries ---
    def account_info(self):
        return None if self.acc_none else self.acc

    def symbol_select(self, symbol, enable=True):
        return True

    def symbol_info(self, symbol):
        return SimpleNamespace(point=self.point, trade_stops_level=self.stops_level)

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=self.bid, ask=self.ask)

    def orders_get(self, symbol=None, ticket=None):
        if self.orders_get_none:
            return None
        if ticket is not None:                         # real MT5 supports ticket lookup
            return [o for o in self._orders if o.ticket == ticket]
        return list(self._orders)

    def positions_get(self, symbol=None, ticket=None):
        if self.positions_get_none:
            return None
        if ticket is not None:
            return [p for p in self._positions if p.ticket == ticket]
        return list(self._positions)

    def history_deals_get(self, *args, ticket=None, position=None, **kw):
        if self.deals_get_none:
            return None
        # IMPORTANT (real MT5 quirk): when called with (from,to,position=X) the position
        # filter is IGNORED — return ALL deals. Only the ticket= form filters.
        if ticket is not None:
            return [d for d in self._deals if d.ticket == ticket]
        return list(self._deals)

    # --- mutation ---
    def order_send(self, req):
        rc = self.next_retcode
        if rc != TRADE_RETCODE_DONE and rc != TRADE_RETCODE_PLACED:
            return SimpleNamespace(retcode=rc, comment="fake-fail", order=0, deal=0, price=0.0)
        action = req["action"]
        if action == TRADE_ACTION_PENDING:
            self._tk += 1
            self._orders.append(SimpleNamespace(
                ticket=self._tk, magic=req["magic"], type=req["type"],
                price_open=req["price"], sl=req["sl"], tp=req["tp"],
                volume_current=req["volume"]))
            return SimpleNamespace(retcode=rc, order=self._tk, deal=0, price=req["price"])
        if action == TRADE_ACTION_DEAL:                # market open
            self._tk += 1
            order_tk = self._tk
            self._tk += 1
            deal_tk = self._tk
            pid = self.market_position_id or order_tk
            fill = self.market_fill_price if self.market_fill_price is not None else req["price"]
            is_buy = req["type"] == ORDER_TYPE_BUY
            self._positions.append(SimpleNamespace(
                ticket=pid, magic=req["magic"],
                type=POSITION_TYPE_BUY if is_buy else POSITION_TYPE_SELL,
                price_open=fill, sl=req["sl"], tp=req["tp"], volume=req["volume"],
                symbol=req["symbol"]))
            self._deals.append(SimpleNamespace(
                ticket=deal_tk, order=order_tk, position_id=pid, entry=DEAL_ENTRY_IN,
                price=fill, volume=req["volume"], profit=0.0, swap=0.0, commission=0.0,
                time=1))
            return SimpleNamespace(retcode=rc, order=order_tk, deal=deal_tk, price=fill)
        if action == TRADE_ACTION_REMOVE:
            self._orders = [o for o in self._orders if o.ticket != req["order"]]
            return SimpleNamespace(retcode=rc, order=req["order"], deal=0, price=0.0)
        if action == TRADE_ACTION_SLTP:
            for p in self._positions:
                if p.ticket == req["position"]:
                    p.sl = req["sl"]
            return SimpleNamespace(retcode=rc, order=0, deal=0, price=0.0)
        return SimpleNamespace(retcode=99999, comment="unknown-action", order=0, deal=0, price=0.0)

    # --- test helpers ---
    def add_balance_deal(self, amount):
        """A deposit/balance deal (position_id 0) — the trap closed_info must NOT sum."""
        self._tk += 1
        self._deals.append(SimpleNamespace(ticket=self._tk, order=0, position_id=0,
                            entry=DEAL_ENTRY_IN, price=0.0, volume=0.0, profit=amount,
                            swap=0.0, commission=0.0, time=0))

    def fill_pending(self, ticket, fill_price):
        """Simulate a resting pending order FILLING: it leaves active orders and becomes a
        POSITION (ticket == order ticket, hedging) with an IN deal. Mirrors real MT5."""
        o = next((x for x in self._orders if x.ticket == ticket), None)
        if o is None:
            return
        self._orders = [x for x in self._orders if x.ticket != ticket]
        is_buy = o.type == ORDER_TYPE_BUY_LIMIT
        self._positions.append(SimpleNamespace(
            ticket=ticket, magic=o.magic,
            type=POSITION_TYPE_BUY if is_buy else POSITION_TYPE_SELL,
            price_open=fill_price, sl=o.sl, tp=o.tp, volume=o.volume_current, symbol="XAUUSDm"))
        self._tk += 1
        self._deals.append(SimpleNamespace(
            ticket=self._tk, order=ticket, position_id=ticket, entry=DEAL_ENTRY_IN,
            price=fill_price, volume=o.volume_current, profit=0.0, swap=0.0, commission=0.0,
            time=5))

    def close_position(self, position_id, out_price, profit, volume=None):
        """Add an OUT deal that fully closes a position (for closed_info tests)."""
        p = next((x for x in self._positions if x.ticket == position_id), None)
        vol = volume if volume is not None else (p.volume if p else 0.01)
        self._tk += 1
        self._deals.append(SimpleNamespace(ticket=self._tk, order=0, position_id=position_id,
                            entry=DEAL_ENTRY_OUT, price=out_price, volume=vol, profit=profit,
                            swap=0.0, commission=0.0, time=10))
        self._positions = [x for x in self._positions if x.ticket != position_id]
