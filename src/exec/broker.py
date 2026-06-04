"""Broker adapters for the UG copier.

Mt5Broker  — real orders via the MetaTrader5 package (Windows/VPS, running terminal).
DryRunBroker — reads REAL prices but never places/cancels; logs intended actions.
              This is the default so you can validate decisions on the live feed
              with zero risk before flipping to --live.

Only `Order`s produced by the unit-tested ug_copier_logic.decide() reach here.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

from src.exec.ug_copier_logic import Order

MAGIC = 770150                  # tags this copier's orders
COMMENT = "ug_copier"

_TRANSIENT_RC = None
def _retryable_retcode(mt5, rc) -> bool:
    """True if an order_send retcode is a TRANSIENT/price-or-market condition — the copier
    should re-evaluate next poll (no alert), NOT terminally abandon the signal. Hard/unknown
    retcodes (AutoTrading off, no money, trade disabled, ...) stay non-retryable → alert.
    Conservative: only KNOWN-transient codes are retryable; anything else is hard (so a real
    'broker said no' like AutoTrading-off is never silently retried away)."""
    global _TRANSIENT_RC
    if _TRANSIENT_RC is None:
        # ONLY deterministic "broker rejected, nothing landed" codes → safe to re-decide.
        # EXCLUDE transport-ambiguous codes (CONNECTION/TIMEOUT): the order MIGHT have landed
        # while the response was lost, and _find_pending() also returns None on a query failure,
        # so retrying could double-place → keep those HARD (fail closed, alert, orphan-recover).
        names = ("TRADE_RETCODE_REQUOTE", "TRADE_RETCODE_REJECT", "TRADE_RETCODE_PRICE_CHANGED",
                 "TRADE_RETCODE_PRICE_OFF", "TRADE_RETCODE_INVALID_PRICE", "TRADE_RETCODE_INVALID_STOPS",
                 "TRADE_RETCODE_MARKET_CLOSED", "TRADE_RETCODE_TOO_MANY_REQUESTS",
                 "TRADE_RETCODE_NO_PRICES", "TRADE_RETCODE_FROZEN")
        _TRANSIENT_RC = {getattr(mt5, n) for n in names if hasattr(mt5, n)}
    return rc in _TRANSIENT_RC


def _synced(fn):
    """Serialize a broker method on self._lock (MT5 API not thread-safe; command-loop
    thread + main loop call concurrently)."""
    def wrap(self, *a, **k):
        with self._lock:
            return fn(self, *a, **k)
    return wrap


class Mt5Broker:
    # Class-level defaults so instances built bypassing __init__ (e.g. test doubles via
    # __new__) still resolve a magic/comment; __init__ overrides per-instance.
    magic = MAGIC
    comment = COMMENT

    def __init__(self, require_demo: bool = True, magic: int = MAGIC, comment: str = COMMENT):
        from src.data import mt5_feed
        mt5_feed.init()                       # attach to the running terminal
        import MetaTrader5 as mt5
        self.mt5 = mt5
        self.require_demo = require_demo
        # Per-instance order tag so independent bots (UG copier vs standalone SMC) NEVER
        # see or manage each other's orders. Defaults preserve the copier's identity; the
        # SMC bot passes its own magic/comment. ALL magic filters below use self.magic.
        self.magic = magic
        self.comment = comment
        # Serialize ALL MT5 access: the command-loop thread (/flat, /stats, /open) calls
        # the broker concurrently with the main trading loop; the MT5 Python API is not
        # guaranteed thread-safe. RLock so nested calls (open_exposure→pending_tickets)
        # don't deadlock.
        self._lock = threading.RLock()
        acc = self._account_info_retry()
        self._login = acc.login if acc else None       # lock the startup account
        # Reason string for the most recent FAILED place_limit/place_market (None on
        # success). The copier reads it to send ONE Telegram alert when a placement
        # fails (e.g. retcode 10027 AutoTrading off, no margin) instead of failing silently.
        self.last_place_error: str | None = None
        # True when the last failure was a TRANSIENT/price-dependent no-send skip (no tick,
        # price moved off SL/TP side, inside stops_level, exec worse than anchor) — the
        # copier should re-evaluate next poll, NOT alert or terminally abandon the signal.
        # False = HARD failure (account blocked, stops geometry, order_send reject) → alert.
        self.last_place_retryable: bool = False

    def _account_info_retry(self, tries: int = 3):
        """account_info() with a few quick retries — a transient None (terminal busy)
        shouldn't be read as a hard failure that blocks an otherwise-valid placement."""
        for i in range(tries):
            acc = self.mt5.account_info()
            if acc is not None:
                return acc
            if i < tries - 1:
                time.sleep(0.3)
        return None

    def _account_ok(self) -> tuple[bool, str]:
        """Re-verify (every send) that the account hasn't changed and, unless
        explicitly allowed, is still a DEMO account."""
        acc = self._account_info_retry()
        if acc is None:
            return False, "no account_info"
        if self._login is None:
            return False, "startup account unknown — refuse to place (fail-closed)"
        if acc.login != self._login:
            return False, f"account changed {self._login}->{acc.login}"
        if self.require_demo and acc.trade_mode != self.mt5.ACCOUNT_TRADE_MODE_DEMO:
            return False, f"account {acc.login} is NOT demo (require_demo)"
        return True, ""

    @_synced
    def login_changed(self) -> bool:
        """True ONLY if the account login DEFINITELY differs from startup (not a transient
        read failure). The main loop fail-stops on this so no read/write hits the wrong
        account after a mid-run account switch. A transient None is NOT a change (reads
        return None / sends fail-closed instead)."""
        acc = self._account_info_retry()
        return acc is not None and self._login is not None and int(acc.login) != int(self._login)

    @_synced
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
        after an ambiguous order_send) — same magic, type, price, volume AND tp/sl.
        TP/SL matter: the two bracket legs share magic/type/entry/volume and differ
        ONLY by TP — without matching TP we could reconcile the TP3 leg onto the
        already-resting TP1 ticket (duplicate row, broken lifecycle)."""
        orders = self.mt5.orders_get(symbol=symbol)
        if not orders:
            return None
        info = self.mt5.symbol_info(symbol)
        tol = (info.point or 0.01) if info else 0.01     # price tolerance (broker-normalized)
        want = (self.mt5.ORDER_TYPE_BUY_LIMIT if o.order_type == "buy_limit"
                else self.mt5.ORDER_TYPE_SELL_LIMIT)
        for x in orders:
            if (x.magic == self.magic and x.type == want
                    and abs(x.price_open - o.entry) <= tol
                    and abs(x.volume_current - o.volume) < 1e-9
                    and abs(x.tp - o.tp) <= tol
                    and abs(x.sl - o.sl) <= tol):
                return int(x.ticket)
        return None

    @_synced
    def open_exposure(self, symbol: str) -> int | None:
        """Authoritative concurrent exposure = our pending orders + our open
        positions. None if any query failed (caller should then NOT place)."""
        pend = self.pending_tickets(symbol)
        if pend is None:
            return None
        pos = self.mt5.positions_get(symbol=symbol)
        if pos is None:
            return None
        return len(pend) + sum(1 for p in pos if p.magic == self.magic)

    @_synced
    def place_limit(self, symbol: str, o: Order, comment: str | None = None) -> int | None:
        mt5 = self.mt5
        self.last_place_error = None
        self.last_place_retryable = False        # default hard; set per-retcode on a send reject
        ok, why = self._account_ok()             # re-verify account EVERY send
        if not ok:
            print(f"  [mt5] BLOCKED place_limit — {why}")
            self.last_place_error = f"BLOCKED: {why}"
            return None
        if not self._stops_ok(symbol, o):
            print(f"  [mt5] SL/TP too close to entry (stops_level) — reject {o.entry}")
            self.last_place_error = "SL/TP too close to entry (stops_level)"
            return None
        otype = mt5.ORDER_TYPE_BUY_LIMIT if o.order_type == "buy_limit" else mt5.ORDER_TYPE_SELL_LIMIT
        req = {
            "action": mt5.TRADE_ACTION_PENDING, "symbol": symbol,
            "volume": float(o.volume), "type": otype, "price": float(o.entry),
            "sl": float(o.sl), "tp": float(o.tp), "magic": self.magic,
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_RETURN,
            "comment": (comment or self.comment)[:31],     # per-order tag (bracket relink)
        }
        r = mt5.order_send(req)
        if r is not None and r.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            return int(r.order)
        # Ambiguous/failed: the order MAY still have landed. Reconcile by scanning
        # pendings before declaring failure (avoids an unmanaged orphan).
        print(f"  [mt5] order_send unclear retcode={getattr(r,'retcode',None)} "
              f"{getattr(r,'comment','')} — reconciling...")
        found = self._find_pending(symbol, o)
        if found is None:                        # genuine failure → record why + classify
            rc = getattr(r, "retcode", None)
            self.last_place_error = f"retcode={rc} {getattr(r,'comment','') or ''}".strip()
            # price-moved/market-transient limit rejects → retryable (re-decide next poll, e.g.
            # as market); AutoTrading-off / no-money / unknown → hard (alert, don't silently lose).
            self.last_place_retryable = _retryable_retcode(mt5, rc)
        return found

    @_synced
    def place_market(self, symbol: str, o: Order) -> int | None:
        """Open a MARKET position immediately (zone-aware in-zone entry). Returns the
        position_id, or None. On an UNCLEAR send we do NOT retry (a market retry could
        double-open); orphan recovery on next startup adopts any position that did land."""
        mt5 = self.mt5
        self.last_place_error = None
        self.last_place_retryable = False
        ok, why = self._account_ok()
        if not ok:
            print(f"  [mt5] BLOCKED place_market — {why}")
            self.last_place_error = f"BLOCKED: {why}"
            return None
        if not self._stops_ok(symbol, o):
            print(f"  [mt5] SL/TP too close to price (stops_level) — reject market {o.entry}")
            self.last_place_error = "SL/TP too close to price (stops_level)"
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print("  [mt5] no tick for market order — skip")
            self.last_place_error = "no tick for market order"
            self.last_place_retryable = True
            return None
        buy = o.order_type == "buy_market"
        price = tick.ask if buy else tick.bid
        # Re-validate against the ACTUAL execution price (not o.entry): spread/tick move
        # since decide() could put the real fill on the wrong side of SL/TP or inside the
        # broker's min stop distance. Fail closed before sending a real market order.
        info = self.mt5.symbol_info(symbol)
        point = ((info.point if info else 0.01) or 0.01)
        min_dist = (info.trade_stops_level or 0) * point if info else 0
        good_side = (o.sl < price < o.tp) if buy else (o.sl > price > o.tp)
        if not good_side:
            print(f"  [mt5] market price {price} not between SL {o.sl} and TP {o.tp} — skip")
            self.last_place_error = f"market price {price} not between SL/TP"
            self.last_place_retryable = True
            return None
        if min_dist > 0 and (abs(price - o.sl) < min_dist or abs(o.tp - price) < min_dist):
            print(f"  [mt5] market price {price} inside stops_level dist — skip")
            self.last_place_error = f"market price {price} inside stops_level"
            self.last_place_retryable = True
            return None
        # NEVER chase: the actual execution (ask for a buy / bid for a sell) must be
        # AT-OR-BETTER than the signal anchor. If spread pushed it past the anchor, skip
        # (decide will re-evaluate; usually it becomes a limit-at-anchor next poll).
        if o.anchor:
            worse = (price > o.anchor + point) if buy else (price < o.anchor - point)
            if worse:
                print(f"  [mt5] market exec {price} worse than anchor {o.anchor} (spread) — skip")
                self.last_place_error = f"market exec {price} worse than anchor {o.anchor} (spread)"
                self.last_place_retryable = True
                return None
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(o.volume),
            "type": mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL, "price": float(price),
            "sl": float(o.sl), "tp": float(o.tp), "deviation": 30, "magic": self.magic,
            "type_filling": mt5.ORDER_FILLING_IOC, "comment": self.comment,
        }
        r = mt5.order_send(req)
        if r is not None and r.retcode == mt5.TRADE_RETCODE_DONE:
            # Resolve the REAL position_id AND actual fill price from the resulting deal
            # (don't assume position_id==order ticket; don't assume fill==requested price).
            pid = int(r.order)
            fill = float(getattr(r, "price", 0.0) or price)
            deal = int(getattr(r, "deal", 0) or 0)
            for _ in range(3):
                if not deal:
                    break
                try:
                    ds = mt5.history_deals_get(ticket=deal)
                except Exception:
                    ds = None
                if ds:
                    pid = int(ds[0].position_id)
                    fill = float(ds[0].price)
                    break
                time.sleep(0.2)
            return {"position_id": pid, "fill_price": fill}
        print(f"  [mt5] market order unclear/failed retcode={getattr(r,'retcode',None)} "
              f"{getattr(r,'comment','')} — NOT retrying (orphan recovery will adopt if it landed)")
        self.last_place_error = (f"retcode={getattr(r,'retcode',None)} "
                                 f"{getattr(r,'comment','') or ''}".strip())
        return None

    @_synced
    def pending_tickets(self, symbol: str) -> set[int] | None:
        """Set of our pending tickets, or None if the query FAILED (so the caller
        does NOT mistake an API error for 'all orders gone')."""
        if self.login_changed():
            print("  [mt5] pending_tickets: account changed — returning None (fail-closed)")
            return None
        orders = self.mt5.orders_get(symbol=symbol)
        if orders is None:
            return None
        return {int(o.ticket) for o in orders if o.magic == self.magic}

    @_synced
    def cancel(self, ticket: int) -> bool:
        ok, why = self._account_ok()             # never manage the wrong account
        if not ok:
            print(f"  [mt5] BLOCKED cancel ticket={ticket} — {why}")
            return False
        od = self.mt5.orders_get(ticket=int(ticket))   # never cancel a non-magic order
        if od is None:                                  # query FAILED → fail-closed, don't blind-cancel
            print(f"  [mt5] cancel ticket={ticket}: orders_get failed — skip (fail-closed)")
            return False
        if od and od[0].magic != self.magic:
            print(f"  [mt5] REFUSE cancel ticket={ticket} — magic {od[0].magic} != {self.magic}")
            return False
        if not od:
            # Order vanished between the caller's pending snapshot and now. It may have just
            # FILLED (not cancelled). Return False so the lifecycle does NOT mark it cancelled;
            # the next _manage_open pass routes it through fill_info() to resolve correctly.
            print(f"  [mt5] cancel ticket={ticket}: order gone — defer to fill_info (not 'cancelled')")
            return False
        r = self.mt5.order_send({"action": self.mt5.TRADE_ACTION_REMOVE, "order": int(ticket)})
        ok = r is not None and r.retcode == self.mt5.TRADE_RETCODE_DONE
        if not ok:
            print(f"  [mt5] cancel FAILED ticket={ticket} retcode={getattr(r,'retcode',None)}")
        return ok

    # ----- lifecycle tracking (fill → close + P/L) -----
    def _deals(self, **kw):
        frm = datetime(2024, 1, 1, tzinfo=timezone.utc)
        to = datetime.now(timezone.utc) + timedelta(days=1)
        return self.mt5.history_deals_get(frm, to, **kw)

    @_synced
    def cancel_all_pendings(self, symbol: str) -> int | None:
        """Cancel every resting magic pending (the /flat command). None if query failed."""
        pend = self.pending_tickets(symbol)
        if pend is None:
            return None
        return sum(1 for tk in list(pend) if self.cancel(tk))

    @_synced
    def list_magic(self, symbol: str) -> dict | None:
        """Our magic pendings + positions as plain dicts (for orphan recovery on
        startup). None if either query failed (caller must NOT conclude 'none')."""
        if self.login_changed():
            return None                      # wrong account — don't adopt its orders
        orders = self.mt5.orders_get(symbol=symbol)
        positions = self.mt5.positions_get(symbol=symbol)
        if orders is None or positions is None:
            return None
        pend = [{"ticket": int(o.ticket), "type": int(o.type), "entry": float(o.price_open),
                 "sl": float(o.sl), "tp": float(o.tp), "volume": float(o.volume_current),
                 "setup_time": int(getattr(o, "time_setup", 0) or 0),
                 "comment": str(getattr(o, "comment", "") or "")}
                for o in orders if o.magic == self.magic]
        pos = [{"position_id": int(p.ticket), "type": int(p.type), "entry": float(p.price_open),
                "sl": float(p.sl), "tp": float(p.tp), "volume": float(p.volume),
                "fill_price": float(p.price_open),
                "fill_time": int(getattr(p, "time", 0) or 0),
                "comment": str(getattr(p, "comment", "") or "")}
               for p in positions if p.magic == self.magic]
        return {"pendings": pend, "positions": pos,
                "buy_limit": self.mt5.ORDER_TYPE_BUY_LIMIT, "pos_buy": self.mt5.POSITION_TYPE_BUY}

    @_synced
    def modify_sl(self, position_id: int, new_sl: float) -> bool:
        """Move an OPEN position's SL (e.g. to break-even), keeping its TP. True on
        success. False if the position is gone or the broker rejects."""
        ok, why = self._account_ok()             # never manage the wrong account
        if not ok:
            print(f"  [mt5] BLOCKED modify_sl pos={position_id} — {why}")
            return False
        pos = self.mt5.positions_get(ticket=int(position_id))
        if not pos:
            print(f"  [mt5] modify_sl: position {position_id} not found (already closed?)")
            return False
        p = pos[0]
        if p.magic != self.magic:                # never modify another bot's / manual position
            print(f"  [mt5] REFUSE modify_sl pos={position_id} — magic {p.magic} != {self.magic}")
            return False
        r = self.mt5.order_send({
            "action": self.mt5.TRADE_ACTION_SLTP, "symbol": p.symbol,
            "position": int(position_id), "sl": float(new_sl), "tp": float(p.tp),
        })
        ok = r is not None and r.retcode == self.mt5.TRADE_RETCODE_DONE
        if not ok:
            print(f"  [mt5] modify_sl FAILED pos={position_id} retcode={getattr(r,'retcode',None)}")
        return ok

    @_synced
    def close_position(self, position_id: int) -> bool:
        """Market-close an OPEN position (the SMC bot's HORIZON time-stop: close a trade
        unresolved after N bars, mirroring the backtest's close-at-horizon). True if the
        position is closed (or already gone); False on a broker reject / query failure."""
        ok, why = self._account_ok()             # never manage the wrong account
        if not ok:
            print(f"  [mt5] BLOCKED close_position pos={position_id} — {why}")
            return False
        pos = self.mt5.positions_get(ticket=int(position_id))
        if pos is None:
            return False                          # query failed — don't conclude 'closed'
        if not pos:
            return True                           # already gone = closed
        p = pos[0]
        if p.magic != self.magic:                # never close another bot's / manual position
            print(f"  [mt5] REFUSE close_position pos={position_id} — magic {p.magic} != {self.magic}")
            return False
        tick = self.mt5.symbol_info_tick(p.symbol)
        if tick is None:
            return False
        is_buy = p.type == self.mt5.POSITION_TYPE_BUY
        price = tick.bid if is_buy else tick.ask  # close a buy at bid, a sell at ask
        r = self.mt5.order_send({
            "action": self.mt5.TRADE_ACTION_DEAL, "symbol": p.symbol,
            "position": int(position_id), "volume": float(p.volume),
            "type": self.mt5.ORDER_TYPE_SELL if is_buy else self.mt5.ORDER_TYPE_BUY,
            "price": float(price), "deviation": 30, "magic": self.magic,
            "type_filling": self.mt5.ORDER_FILLING_IOC, "comment": self.comment + "_hzn",
        })
        ok = r is not None and r.retcode == self.mt5.TRADE_RETCODE_DONE
        if not ok:
            print(f"  [mt5] close_position FAILED pos={position_id} retcode={getattr(r,'retcode',None)}")
        return ok

    @_synced
    def copy_m15(self, symbol: str, count: int):
        """Last `count` CLOSED M15 bars as a DataFrame (time, open, high, low, close) for
        the SMC engine. Drops the still-forming current bar (index -1) so detection only
        ever runs on closed bars (anti-lookahead). None on any query failure."""
        import pandas as pd
        mt5 = self.mt5
        if not mt5.symbol_select(symbol, True):
            return None
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, int(count) + 1)
        if rates is None or len(rates) < 2:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")     # broker server time
        df = df[["time", "open", "high", "low", "close"]].iloc[:-1].reset_index(drop=True)
        return df

    @_synced
    def fill_info(self, order_ticket: int):
        """Filled → {position_id, fill_price}; confirmed-not-filled → None;
        query FAILED → "unknown" (so the caller doesn't misread a transient failure).

        REAL-TIME FIRST: a filled pending becomes a POSITION with the SAME ticket
        immediately (no history lag). We check positions BEFORE deal history so a
        just-filled order is never misread as 'cancelled/vanished' while history lags.
        History is the fallback (covers a position already closed since it filled)."""
        if self.login_changed():
            return "unknown"                 # wrong account — don't conclude anything
        pos = self.mt5.positions_get(ticket=int(order_ticket))
        if pos is None:
            return "unknown"                 # positions query failed — don't conclude
        if pos:
            p = pos[0]
            return {"position_id": int(p.ticket), "fill_price": float(p.price_open)}
        deals = self._deals()
        if deals is None:
            return "unknown"                 # query failed — don't conclude anything
        for d in deals:
            if int(d.order) == int(order_ticket) and d.entry == self.mt5.DEAL_ENTRY_IN:
                return {"position_id": int(d.position_id), "fill_price": float(d.price)}
        return None

    @_synced
    def closed_info(self, position_id: int) -> dict | None:
        """Fully-closed position → realized profit (incl swap+commission) + last
        close price. None if still open, partially open, or no exit deal yet."""
        if self.login_changed():
            return None                      # wrong account — treat as 'not closed yet'
        # NOTE: history_deals_get(from, to, position=X) IGNORES the position filter
        # when dates are also given — it returns ALL deals in the window (incl. the
        # account's balance/deposit deal). That made profit = account balance, not the
        # trade's P/L. So fetch the window and filter by position_id ourselves.
        deals = self._deals()
        if not deals:
            return None
        deals = [d for d in deals if int(d.position_id) == int(position_id)]
        if not deals:
            return None
        ins = [d for d in deals if d.entry == self.mt5.DEAL_ENTRY_IN]
        outs = [d for d in deals if d.entry == self.mt5.DEAL_ENTRY_OUT]
        if not outs:
            return None
        if sum(d.volume for d in outs) + 1e-9 < sum(d.volume for d in ins):
            return None                       # only partially closed — keep tracking
        profit = sum(d.profit + d.swap + d.commission for d in deals)
        last_out = max(outs, key=lambda d: d.time)
        return {"profit": float(profit), "close_price": float(last_out.price)}


class DryRunBroker(Mt5Broker):
    """Real prices, NO order placement. Synthetic tickets so the loop can track."""

    def __init__(self, magic: int = MAGIC, comment: str = COMMENT):
        super().__init__(magic=magic, comment=comment)
        self._fake = 9_000_000
        self._dry_open: set[int] = set()      # in-memory; never persisted → can't poison live

    def place_limit(self, symbol: str, o: Order, comment: str | None = None) -> int | None:
        self._fake += 1
        self._dry_open.add(self._fake)
        print(f"  [DRY] would place {o.order_type} {symbol} {o.volume} @ {o.entry} "
              f"sl={o.sl} tp={o.tp} ({comment or ''}) ticket~{self._fake}")
        return self._fake

    def place_market(self, symbol: str, o: Order) -> dict | None:
        self._fake += 1
        self._dry_open.add(self._fake)
        print(f"  [DRY] would MARKET {o.order_type} {symbol} {o.volume} @ ~{o.entry} "
              f"sl={o.sl} tp={o.tp} (TP1 {o.tp1_pip:g}pip) pos~{self._fake}")
        return {"position_id": self._fake, "fill_price": o.entry}

    def pending_tickets(self, symbol: str) -> set[int] | None:
        return set(self._dry_open)            # so management/cancel logic runs in dry too

    def open_exposure(self, symbol: str) -> int | None:
        return len(self._dry_open)

    def cancel(self, ticket: int) -> bool:
        self._dry_open.discard(ticket)
        print(f"  [DRY] would cancel ticket {ticket}")
        return True

    def list_magic(self, symbol: str) -> dict | None:
        return {"pendings": [], "positions": [], "buy_limit": 2, "pos_buy": 0}

    def modify_sl(self, position_id: int, new_sl: float) -> bool:
        print(f"  [DRY] would move SL of pos {position_id} -> {new_sl}")
        return True

    def close_position(self, position_id: int) -> bool:
        self._dry_open.discard(position_id)
        print(f"  [DRY] would close pos {position_id} (horizon)")
        return True

    def fill_info(self, order_ticket: int) -> dict | None:
        return None              # dry-run orders never fill

    def closed_info(self, position_id: int) -> dict | None:
        return None
