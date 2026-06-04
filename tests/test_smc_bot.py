"""End-to-end lifecycle tests for the standalone SMC bot harness (scripts/smc_bot.py),
driven by a fake broker over a temp ledger. Covers the full path the live bot walks:
place 2-leg bracket → fill → near-leg wins → runner SL→BE → runner closes; plus the
expiry cancel and the HORIZON time-stop. No MT5, no network.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.exec import trade_db, notify
from src.exec.smc_logic import build_setup
import scripts.smc_bot as bot


class Clock:
    def __init__(self, t):
        self.t = pd.Timestamp(t, tz="UTC")

    def __call__(self):
        return self.t.isoformat()

    def advance(self, minutes):
        self.t += pd.Timedelta(minutes=minutes)

    def epoch(self, minutes_ago=0):
        return int((self.t - pd.Timedelta(minutes=minutes_ago)).timestamp())


class FakeBroker:
    """Minimal broker double: records placements and lets the test drive fills/closes."""
    def __init__(self):
        self.magic = bot.SMC_MAGIC; self.comment = bot.SMC_COMMENT
        self.last_place_error = None; self.last_place_retryable = False
        self._tk = 1000
        self.pendings: set[int] = set()
        self.fills: dict[int, tuple[int, float]] = {}   # ticket -> (position_id, fill_price)
        self.closes: dict[int, dict] = {}               # position_id -> {close_price, profit}
        self.modified: list[tuple[int, float]] = []
        self.hzn_closed: list[int] = []
        self.place_fail = False                         # force a hard place failure
        self.runner_fail = False                        # force only the runner leg to fail
        self.magic_pendings: list = []                  # for list_magic / orphan adoption
        self.magic_positions: list = []
        self.list_magic_on = False

    def place_limit(self, symbol, o, comment=None):
        self.last_comment = comment
        if self.place_fail:
            self.last_place_error = "forced fail"; self.last_place_retryable = False
            return None
        if self.runner_fail and o.leg == "tp3":
            self.last_place_error = "runner fail"; self.last_place_retryable = False
            return None
        self._tk += 1; self.pendings.add(self._tk); return self._tk

    def pending_tickets(self, symbol):
        return set(self.pendings)

    def cancel(self, tk):
        self.pendings.discard(tk); return True

    def fill(self, tk, pid, px):
        """Simulate a real fill: a filled limit leaves orders_get (becomes a position),
        so it vanishes from pending_tickets and fill_info then reports it filled."""
        self.pendings.discard(tk)
        self.fills[tk] = (pid, px)

    def fill_info(self, tk):
        if tk in self.fills:
            pid, px = self.fills[tk]
            return {"position_id": pid, "fill_price": px}
        return None                                     # confirmed not filled

    def closed_info(self, pid):
        return self.closes.get(pid)

    def modify_sl(self, pid, sl):
        self.modified.append((pid, sl)); return True

    def close_position(self, pid):
        self.hzn_closed.append(pid); return True

    def list_magic(self, symbol):
        if not self.list_magic_on:
            return None                                 # default: skip orphan scan
        return {"pendings": self.magic_pendings, "positions": self.magic_positions,
                "buy_limit": 2, "pos_buy": 0}


@pytest.fixture
def env(tmp_path, monkeypatch):
    trade_db.DB_PATH = tmp_path / "smc_test.db"
    trade_db.init_db()
    monkeypatch.setattr(notify, "send", lambda *a, **k: None)
    clk = Clock("2026-06-01T00:00:00")
    monkeypatch.setattr(bot, "now_iso", clk)
    return clk


def _setup(entry=4460.0, sl=4452.0, R=8.0, sign=1, t="2026-06-01T00:00:00"):
    return build_setup(sign, entry, sl, R, pd.Timestamp(t))


def _rows():
    return {r["leg"]: r for r in trade_db.open_trades()}


def test_full_lifecycle_place_fill_be_close(env):
    clk = env
    b = FakeBroker()
    assert bot._place_setup(b, "XAUUSDm", _setup(), 0.01) is True
    rows = _rows()
    assert set(rows) == {"tp1", "tp3"}                  # both legs inserted
    near_tk, run_tk = rows["tp1"]["ticket"], rows["tp3"]["ticket"]
    assert rows["tp1"]["tp"] == 4460.0 + 4 * 8.0        # +4R
    assert rows["tp3"]["tp"] == 4460.0 + 10 * 8.0       # +10R runner

    # both legs fill at the OB retest
    b.fill(near_tk, 5001, 4460.0); b.fill(run_tk, 5002, 4460.0)
    bot._manage_open(b, "XAUUSDm", clk, 360, 1440)
    rows = _rows()
    assert rows["tp1"]["status"] == "filled" and rows["tp3"]["status"] == "filled"

    # near leg hits +4R (win) → next pass closes it AND moves runner SL → BE (entry)
    b.closes[5001] = {"close_price": 4492.0, "profit": 8.0}
    bot._manage_open(b, "XAUUSDm", clk, 360, 1440)
    assert (5002, 4460.0) in b.modified                 # runner SL dragged to entry
    run = {r["leg"]: r for r in trade_db.siblings(rows["tp3"]["group_id"])}["tp3"]
    assert run["sl"] == 4460.0 and run["status"] == "filled"

    # runner later closes (rides to +10R)
    b.closes[5002] = {"close_price": 4540.0, "profit": 20.0}
    bot._manage_open(b, "XAUUSDm", clk, 360, 1440)
    s = trade_db.summary()
    assert s["signals"] == 1 and s["closed"] == 1 and s["wins"] == 1
    assert s["pnl"] == pytest.approx(28.0)              # 8 + 20 banked


def test_expiry_cancels_unfilled_pending(env):
    clk = env
    b = FakeBroker()
    bot._place_setup(b, "XAUUSDm", _setup(), 0.01)
    assert len(b.pendings) == 2
    clk.advance(361)                                    # past the 360-min retest window
    bot._manage_open(b, "XAUUSDm", clk, 360, 1440)
    assert len(b.pendings) == 0                          # both cancelled
    assert all(r["status"] == "cancelled" for r in trade_db.recent(5)
               if r["status"] not in ("filled",))


def test_horizon_time_stop_closes_filled(env):
    clk = env
    b = FakeBroker()
    bot._place_setup(b, "XAUUSDm", _setup(), 0.01)
    rows = _rows()
    b.fill(rows["tp1"]["ticket"], 6001, 4460.0)
    b.fill(rows["tp3"]["ticket"], 6002, 4460.0)
    bot._manage_open(b, "XAUUSDm", clk, 360, 1440)       # both filled
    clk.advance(1441)                                    # past the 24h horizon
    bot._manage_open(b, "XAUUSDm", clk, 360, 1440)
    assert 6001 in b.hzn_closed and 6002 in b.hzn_closed  # both market-closed by time-stop


def test_hard_place_failure_places_nothing(env):
    clk = env
    b = FakeBroker(); b.place_fail = True
    assert bot._place_setup(b, "XAUUSDm", _setup(), 0.01) is False
    assert trade_db.open_trades() == []                  # nothing tracked


def test_runner_fail_keeps_near_leg(env):
    clk = env
    b = FakeBroker(); b.runner_fail = True
    assert bot._place_setup(b, "XAUUSDm", _setup(), 0.01) is True
    rows = _rows()
    assert set(rows) == {"tp1"}                           # near tracked, runner absent
    assert len(b.pendings) == 1


def test_short_setup_geometry(env):
    clk = env
    b = FakeBroker()
    bot._place_setup(b, "XAUUSDm", _setup(entry=4540.0, sl=4548.0, R=8.0, sign=-1), 0.01)
    rows = _rows()
    assert rows["tp1"]["order_type"] == "sell_limit"
    assert rows["tp1"]["tp"] == 4540.0 - 4 * 8.0          # below entry for a short
    assert rows["tp3"]["tp"] == 4540.0 - 10 * 8.0


def test_orphan_runner_relinks_via_comment_and_BEs(env):
    """Crash lost the runner's DB row, AND the near already closed_tp before the orphan
    sweep. The runner's order COMMENT carries its exact group → it relinks and (siblings
    include the closed near) SL→BE still fires (fixes #2/#3/#4)."""
    clk = env
    b = FakeBroker(); b.list_magic_on = True
    gid = "smcabc123L"                                        # short gid (as embedded in comments)
    trade_db.insert({"direction": "long", "order_type": "buy_limit", "entry": 4460.0,
                     "sl": 4452.0, "tp": 4492.0, "volume": 0.01, "ticket": 7001,
                     "position_id": 7001, "fill_price": 4460.0, "status": "closed_tp",
                     "profit": 8.0, "close_price": 4492.0, "created_at": clk(),
                     "filled_at": clk(), "closed_at": clk(), "leg": "tp1", "group_id": gid})
    # the runner exists at the broker as an untracked magic position, comment = "<gid>-tp3"
    b.magic_positions = [{"position_id": 7002, "type": 0, "entry": 4460.0, "sl": 4452.0,
                          "tp": 4540.0, "volume": 0.01, "fill_price": 4460.0,
                          "fill_time": clk.epoch(), "comment": f"{gid}-tp3"}]
    bot._adopt_orphans(b, "XAUUSDm")
    sibs = {r["leg"]: r for r in trade_db.siblings(gid)}
    assert "tp3" in sibs and sibs["tp3"]["position_id"] == 7002    # relinked to the exact group
    # near already won (closed_tp) → BE loop moves the relinked runner's SL to entry
    bot._manage_open(b, "XAUUSDm", clk, 360, 1440)
    assert (7002, 4460.0) in b.modified


def test_orphan_unknown_comment_is_standalone(env):
    """A magic order with no recognisable comment is adopted as a bounded standalone orphan."""
    clk = env
    b = FakeBroker(); b.list_magic_on = True
    b.magic_positions = [{"position_id": 7100, "type": 0, "entry": 4460.0, "sl": 4452.0,
                          "tp": 4540.0, "volume": 0.01, "fill_price": 4460.0,
                          "fill_time": clk.epoch(), "comment": "manual trade"}]
    bot._adopt_orphans(b, "XAUUSDm")
    row = [r for r in trade_db.open_trades() if r["position_id"] == 7100][0]
    assert row["leg"] == "orphan" and row["group_id"] == "orphan:7100"


def test_orphan_stale_pending_uses_real_age_and_expires(env):
    """An adopted pending keeps its real setup time, so a stale one expires immediately
    instead of getting a fresh window (fix #3)."""
    clk = env
    b = FakeBroker(); b.list_magic_on = True
    b.pendings.add(8001)                                  # broker still shows it resting
    b.magic_pendings = [{"ticket": 8001, "type": 2, "entry": 4460.0, "sl": 4452.0,
                         "tp": 4492.0, "volume": 0.01, "setup_time": clk.epoch(minutes_ago=500)}]
    bot._adopt_orphans(b, "XAUUSDm")
    bot._manage_open(b, "XAUUSDm", clk, 360, 1440)        # age ~500min > 360 → cancel
    assert 8001 not in b.pendings


def test_horizon_close_does_not_trigger_be(env):
    """A near leg closed by the horizon time-stop is closed_other (NOT closed_tp), so it
    must NOT move the runner to BE (fix #4)."""
    clk = env
    b = FakeBroker()
    bot._place_setup(b, "XAUUSDm", _setup(), 0.01)
    rows = _rows()
    b.fill(rows["tp1"]["ticket"], 9001, 4460.0)
    b.fill(rows["tp3"]["ticket"], 9002, 4460.0)
    bot._manage_open(b, "XAUUSDm", clk, 360, 1440)
    # mark the near leg as horizon-closed, then it reports a small-profit close
    near = {r["leg"]: r for r in trade_db.open_trades()}["tp1"]
    trade_db.update(near["id"], note="horizon 24h time-stop")
    b.closes[9001] = {"close_price": 4470.0, "profit": 3.0}   # partial profit (near tp=4492)
    bot._manage_open(b, "XAUUSDm", clk, 360, 1440)
    closed_near = {r["leg"]: r for r in trade_db.siblings(rows["tp3"]["group_id"])}["tp1"]
    assert closed_near["status"] == "closed_other"        # not closed_tp
    assert b.modified == []                                # runner NOT moved to BE
