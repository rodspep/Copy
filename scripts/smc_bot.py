"""Standalone SMC trading bot — FULLY INDEPENDENT of the UG copier.

It generates its OWN signals from price (smc_logic.decide on closed M15 bars), with its
OWN broker magic (770820 ≠ copier's 770150 → the two bots never see/touch each other's
orders), OWN ledger (data/smc_trades[_real_<login>].db), OWN state/lock/pause, and OWN
Telegram (configs/smc_telegram.json). Shares only the battle-tested execution layer
(broker.py) and ledger module (trade_db) — isolated by magic + a separate DB file.

Strategy (walk-forward-selected, parity-locked to the backtest by tests/test_smc_logic.py):
  CHOCH+sweep+OB on M15, H1-EMA50 trend filter, OB-retest LIMIT entry, 2-leg 50/50 exit:
  0.01 @ +4R (books then drags the runner's SL → BE) + 0.01 @ +10R runner; pending expires
  after RETEST_BARS; a filled trade unresolved after HORIZON is market-closed (time-stop).
  Up to --max-setups (default 4) concurrent setups. REALIZABLE backtest @ 0.02/signal over
  18mo (scripts/smc_live_sim.py, no-gate cap 4): +$5019, WR 28%, maxDD -$915 — needs a
  SEPARATE HEDGING account funded ~$4500-5000 (maxDD ≈ 18-20%). NOT for a small account.
  (The idealized smc_legged +$5562/-$644 used look-ahead fill and is NOT realizable.)

Dry-run by default (reads real prices, places nothing). --live places real orders;
--allow-real is additionally required to trade a non-demo account.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import threading
import time
from pathlib import Path

import pandas as pd

from src.exec import notify, trade_db
from src.exec.ug_copier_logic import Order
from src.exec.smc_logic import (decide, RETEST_BARS, HORIZON, LOT_PER_LEG,
                                 TP_NEAR_R, TP_RUN_R)

SMC_MAGIC = 770820                 # ≠ copier 770150 — total order isolation
SMC_COMMENT = "smc_bot"
WINDOW_BARS = 1000                 # M15 bars per poll (HTF EMA50 + swings well warmed)
M15_MIN = 15

# per-namespace paths (set in main once the ledger tag is known)
LOCK = Path("data/smc_bot.lock")
PAUSE = Path("data/smc_bot.pause")
STATE = Path("data/smc_bot_state.json")


def now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


# ----------------------------------------------------------------- singleton / pause
def _acquire_singleton(path: Path) -> None:
    """Refuse to start a 2nd instance on the same ledger (double-placing real orders)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import msvcrt  # noqa
        global _LOCK_FH
        _LOCK_FH = open(path, "w")
        msvcrt.locking(_LOCK_FH.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit(f"another smc_bot holds {path} — refusing to start a 2nd instance")
    except ImportError:
        pass


def _paused() -> bool:
    return PAUSE.exists()


def _load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_bar": None}


def _save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE)


# ----------------------------------------------------------------- formatting
def _vn(ts_iso: str) -> str:
    try:
        return (pd.Timestamp(ts_iso).tz_convert("Asia/Ho_Chi_Minh")
                .strftime("%H:%M:%S %d/%m"))
    except Exception:
        return ts_iso or "?"


def _setup_key(direction: str, entry: float, sl: float) -> str:
    return f"{direction}|{round(entry, 2)}|{round(sl, 2)}"


def _b36(n: int) -> str:
    n = int(n)
    if n == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"; s = ""
    while n:
        n, r = divmod(n, 36); s = digits[r] + s
    return s


def _gid_for(setup) -> str:
    """Short, stable group id (fits a 31-char MT5 order comment) from the setup bar epoch
    + direction. The two legs share it; it's stored as group_id AND embedded in each leg's
    order comment so an orphaned leg relinks to its exact bracket (not by fuzzy geometry)."""
    epoch = int(pd.Timestamp(setup.bar_time).timestamp())
    return f"smc{_b36(epoch)}{'L' if setup.direction == 'long' else 'S'}"


def _parse_comment(cmt: str):
    """Recover (group_id, leg) from an order comment 'smc<base36><L|S>-<leg>'. None if not ours."""
    if not cmt or not cmt.startswith("smc") or "-" not in cmt:
        return None
    gid, _, leg = cmt.rpartition("-")
    return (gid, leg) if (leg in ("tp1", "tp3") and len(gid) > 3) else None


# ----------------------------------------------------------------- placement
def _legs_for(setup, volume: float) -> list[Order]:
    """Two limit Orders (near=tp1 books→BE, runner=tp3) from a smc_logic.Setup."""
    otype = "buy_limit" if setup.direction == "long" else "sell_limit"
    near, run = setup.legs
    mk = lambda leg, lbl: Order(side=setup.direction, order_type=otype, entry=setup.entry,
                                sl=setup.sl, tp=leg.tp_price, volume=volume,
                                tp1_pip=0.0, leg=lbl, tp_pip=0.0, anchor=0.0)
    return [mk(near, "tp1"), mk(run, "tp3")]


def _place_setup(broker, symbol, setup, volume) -> bool:
    """Place the bracket: near leg first; only add the runner if the near landed (a
    runner without a near would never get its SL→BE). Returns True if anything landed."""
    gid = _gid_for(setup)
    near_o, run_o = _legs_for(setup, volume)
    rr = f"R={setup.R:.2f} entry {setup.entry:.2f} SL {setup.sl:.2f}"
    near_tk = broker.place_limit(symbol, near_o, comment=f"{gid}-tp1")
    if not near_tk:
        if not broker.last_place_retryable:           # HARD fail → alert; transient → silent retry
            notify.send(f"⛔ <b>SMC: đặt lệnh THẤT BẠI</b> ({setup.direction})\n{rr}\n"
                        f"Lý do: {broker.last_place_error}")
        print(f"  [{now_iso()}] PLACE near failed ({broker.last_place_error})")
        return False
    base = dict(signal_ts=pd.Timestamp(setup.bar_time).isoformat(), direction=setup.direction,
                method_pip=None, order_type=near_o.order_type, entry=setup.entry, sl=setup.sl,
                volume=volume, status="pending", created_at=now_iso(), group_id=gid,
                note=f"SMC {rr}")
    mid = notify.send(f"🎯 <b>SMC {setup.direction.upper()}</b> @ {setup.entry:.2f}\n"
                      f"SL {setup.sl:.2f} · TP +{TP_NEAR_R:g}R/{near_o.tp:.2f} → "
                      f"runner +{TP_RUN_R:g}R/{run_o.tp:.2f}\n{rr}")
    trade_db.insert({**base, "tp": near_o.tp, "ticket": near_tk, "leg": "tp1", "tg_msg_id": mid})
    run_tk = broker.place_limit(symbol, run_o, comment=f"{gid}-tp3")
    if run_tk:
        trade_db.insert({**base, "tp": run_o.tp, "ticket": run_tk, "leg": "tp3", "tg_msg_id": mid})
    else:
        notify.send("⚠️ <b>SMC: chỉ vào được leg near</b> (runner lỗi) — vẫn có SL riêng",
                    reply_to=mid)
    print(f"  [{now_iso()}] PLACED SMC {setup.direction} {rr} near={near_tk} run={run_tk}")
    return True


# ----------------------------------------------------------------- lifecycle mgmt
def _manage_open(broker, symbol, now_iso, expiry_min, horizon_min):
    """One management pass over DB-tracked trades: pending→fill/cancel/expiry,
    filled→close(+P/L), runner SL→BE after the near leg wins, and the HORIZON time-stop.
    Returns 'skip' if the pending query failed. Extracted so it is testable end-to-end."""
    live_tickets = broker.pending_tickets(symbol)
    if live_tickets is None:
        print(f"  [{now_iso()}] pending query failed; skip management this cycle")
        return "skip"

    for r in trade_db.open_trades():
        tid, tk, leg = r["id"], r["ticket"], (r["leg"] or "tp1")
        if r["status"] == "pending":
            if tk in live_tickets:
                age = (pd.Timestamp(now_iso()) - pd.Timestamp(r["created_at"])).total_seconds() / 60
                if age > expiry_min:
                    if broker.cancel(tk):
                        trade_db.update(tid, status="cancelled", closed_at=now_iso(),
                                        note=f"hết hạn {age:.0f}min (chưa retest)")
                        notify.send(f"🚫 <b>SMC hủy {leg}</b> — hết hạn {age:.0f}min",
                                    reply_to=r["tg_msg_id"])
                        print(f"  [{now_iso()}] CANCEL {leg} {tk} (expiry {age:.0f}min)")
            else:
                fi = broker.fill_info(tk)
                if fi == "unknown":
                    continue
                if fi:
                    trade_db.update(tid, status="filled", position_id=fi["position_id"],
                                    fill_price=fi["fill_price"], entry=fi["fill_price"],
                                    filled_at=now_iso())
                    notify.send(f"📌 <b>SMC vào lệnh</b> ({leg}) @ {fi['fill_price']:.2f}",
                                reply_to=r["tg_msg_id"])
                    print(f"  [{now_iso()}] FILLED {leg} {tk} @ {fi['fill_price']:.2f}")
                else:
                    trade_db.update(tid, status="cancelled", closed_at=now_iso(),
                                    note="pending vanished")
        elif r["status"] == "filled":
            ci = broker.closed_info(r["position_id"])
            if ci:
                cp, pnl = ci["close_price"], ci["profit"]
                horizon_note = (r["note"] or "").startswith("horizon")
                # Classify with a real tolerance — NOT nearest-of-tp/sl. A horizon/other close
                # at partial profit must NOT be mislabelled closed_tp, because the runner's
                # SL→BE fires ONLY on a genuine near-leg closed_tp.
                R = abs((r["entry"] or 0) - (r["sl"] or 0)) or 1.0
                tol = max(0.30, 0.05 * R)
                if horizon_note:
                    reason = "closed_other"                 # time-stop: never a TP
                elif abs(cp - r["tp"]) <= tol:
                    reason = "closed_tp"
                elif abs(cp - (r["entry"] or 0)) <= tol:
                    reason = "closed_be"
                else:
                    reason = "closed_sl"
                trade_db.update(tid, status=reason, close_price=cp, profit=pnl, closed_at=now_iso())
                icon = "✅ WIN" if pnl > 1e-9 else ("❌ LOSS" if pnl < -1e-9 else "⚪ BE")
                hit = {"closed_tp": "TP", "closed_sl": "SL", "closed_be": "hòa vốn (BE)",
                       "closed_other": "hết giờ (24h)" if horizon_note else "đóng khác"}.get(reason, reason)
                notify.send(f"{icon} <b>SMC {leg}</b> — đóng tại {hit} @ {cp:.2f}\n"
                            f"💰 {pnl:+.2f} USD", reply_to=r["tg_msg_id"])
                print(f"  [{now_iso()}] CLOSED {leg} {hit} @ {cp:.2f} pnl {pnl:+.2f}")
            else:
                # HORIZON time-stop: a trade unresolved after N min is market-closed
                # (mirrors the backtest's close-at-horizon). Tag the note so the eventual
                # close alert reads 'hết giờ'.
                ft = r["filled_at"] or r["created_at"]
                age = (pd.Timestamp(now_iso()) - pd.Timestamp(ft)).total_seconds() / 60
                if age > horizon_min and not (r["note"] or "").startswith("horizon"):
                    if broker.close_position(r["position_id"]):
                        trade_db.update(tid, note="horizon 24h time-stop")
                        print(f"  [{now_iso()}] HORIZON close {leg} pos {r['position_id']} ({age:.0f}min)")

    # Break-even: an OPEN runner whose near sibling already WON → SL → entry. Re-query so a
    # runner that filled earlier this same pass is seen 'filled' now (BE applies immediately).
    for r in trade_db.open_trades():
        if r["leg"] == "tp3" and r["status"] == "filled" and r["position_id"] and r["group_id"]:
            sibs = trade_db.siblings(r["group_id"])
            near_won = any(s["leg"] == "tp1" and s["status"] == "closed_tp" for s in sibs)
            if near_won and r["sl"] is not None and abs(r["sl"] - r["entry"]) > 1e-6:
                if broker.modify_sl(r["position_id"], r["entry"]):
                    trade_db.update(r["id"], sl=r["entry"], note="SL→BE sau near")
                    notify.send("🛡️ <b>SMC runner: SL dời hòa vốn</b> (near đã thắng)",
                                reply_to=r["tg_msg_id"])
                    print(f"  [{now_iso()}] BE move pos {r['position_id']} -> {r['entry']}")
    return None


def _adopt_orphans(broker, symbol: str) -> None:
    """Adopt our magic orders/positions missing from the ledger (e.g. a crash between
    order_send and the DB insert). Each leg's order COMMENT carries its exact group_id+leg
    ('smc<b36><L|S>-tp1/tp3'), so an orphaned runner relinks to its precise bracket — even
    if the near already closed or other setups run concurrently (no fuzzy geometry guessing).
    `siblings()` includes closed rows, so SL→BE still fires if the near already won. Real MT5
    setup/fill times are used so expiry/horizon measure TRUE age (a stale orphan is then
    cancelled/closed by the next _manage_open pass). Unrecognised comment → standalone orphan,
    still bounded by its own SL/TP."""
    info = broker.list_magic(symbol)
    if not info:
        print("  [adopt] list_magic unavailable — skipping orphan scan")
        return
    open_now = trade_db.open_trades()
    known_ord = {int(r["ticket"]) for r in open_now if r["ticket"]}
    known_pos = {int(r["position_id"]) for r in open_now if r["position_id"]}

    def _iso(epoch):
        try:
            return pd.Timestamp(int(epoch), unit="s", tz="UTC").isoformat() if epoch else now_iso()
        except Exception:
            return now_iso()

    adopted = 0
    for o in info["pendings"]:
        if o["ticket"] in known_ord:
            continue
        gid, leg = _parse_comment(o.get("comment", "")) or (f"orphan:{o['ticket']}", "orphan")
        long = o["type"] == info["buy_limit"]
        trade_db.insert({"direction": "long" if long else "short",
                         "order_type": "buy_limit" if long else "sell_limit",
                         "entry": o["entry"], "sl": o["sl"], "tp": o["tp"], "volume": o["volume"],
                         "ticket": o["ticket"], "status": "pending",
                         "created_at": _iso(o.get("setup_time")), "leg": leg, "group_id": gid,
                         "note": "adopted orphan pending"}); adopted += 1
    for p in info["positions"]:
        if p["position_id"] in known_pos or p["position_id"] in known_ord:
            continue
        gid, leg = _parse_comment(p.get("comment", "")) or (f"orphan:{p['position_id']}", "orphan")
        long = p["type"] == info["pos_buy"]
        ts = _iso(p.get("fill_time"))
        trade_db.insert({"direction": "long" if long else "short",
                         "order_type": "buy_limit" if long else "sell_limit",
                         "entry": p["entry"], "sl": p["sl"], "tp": p["tp"], "volume": p["volume"],
                         "ticket": p["position_id"], "position_id": p["position_id"],
                         "fill_price": p["fill_price"], "status": "filled", "created_at": ts,
                         "filled_at": ts, "leg": leg, "group_id": gid,
                         "note": "adopted orphan position"}); adopted += 1
    if adopted:
        print(f"  [adopt] recovered {adopted} untracked magic order(s)/position(s)")
        notify.send(f"♻️ <b>SMC nhận lại {adopted} lệnh mồ côi</b> vào quản lý.")


# ----------------------------------------------------------------- telegram commands
def _command_loop(broker, symbol: str) -> None:
    """Minimal getUpdates loop: /stats /open /pause /resume /flat. Best-effort; a Telegram
    hiccup never affects trading. Reuses notify creds (the SMC bot's own token)."""
    creds = notify.creds()
    if creds is None:
        return
    token, _ = creds
    api = f"https://api.telegram.org/bot{token}"
    import requests
    try:
        requests.post(f"{api}/setMyCommands", json={"commands": [
            {"command": "stats", "description": "Thống kê P/L"},
            {"command": "open", "description": "Lệnh đang mở"},
            {"command": "pause", "description": "Dừng vào lệnh mới"},
            {"command": "resume", "description": "Chạy lại"},
            {"command": "flat", "description": "Hủy chờ + đóng tất cả"},
        ]}, timeout=(3.05, 5))
    except Exception:
        pass
    offset = None
    while True:
        try:
            r = requests.get(f"{api}/getUpdates",
                             params={"timeout": 50, "offset": offset}, timeout=(3.05, 60))
            for u in r.json().get("result", []):
                offset = u["update_id"] + 1
                msg = (u.get("message") or {}).get("text", "") or ""
                cmd = msg.strip().lower().split("@")[0]
                if cmd == "/stats":
                    s = trade_db.summary()
                    notify.send(f"📊 <b>SMC</b>\nSignals {s['signals']} · mở {s['open']} · "
                                f"đóng {s['closed']}\nWR {s['winrate']*100:.0f}% "
                                f"({s['wins']}/{s['closed']})\n💰 P/L {s['pnl']:+.2f} USD")
                elif cmd == "/open":
                    rows = [r for r in trade_db.open_trades()]
                    if not rows:
                        notify.send("Không có lệnh mở.")
                    else:
                        txt = "\n".join(f"• {r['direction']} {r['leg']} @ {r['entry']:.2f} "
                                        f"[{r['status']}]" for r in rows[:20])
                        notify.send(f"<b>SMC mở ({len(rows)})</b>\n{txt}")
                elif cmd == "/pause":
                    PAUSE.write_text("paused", encoding="utf-8")
                    notify.send("⏸️ SMC tạm dừng vào lệnh mới (lệnh đang mở vẫn được quản lý).")
                elif cmd == "/resume":
                    PAUSE.unlink(missing_ok=True)
                    notify.send("▶️ SMC chạy lại.")
                elif cmd == "/flat":
                    PAUSE.write_text("flat", encoding="utf-8")   # fail-closed: stop NEW entries first
                    n = broker.cancel_all_pendings(symbol)
                    closed = 0
                    info = broker.list_magic(symbol)
                    if info:
                        for p in info["positions"]:
                            if broker.close_position(p["position_id"]):
                                closed += 1
                    notify.send(f"🧹 SMC flat: hủy {n} chờ, đóng {closed} vị thế.\n"
                                f"⏸️ ĐÃ TẠM DỪNG — gõ /resume để vào lệnh lại.")
        except Exception:
            time.sleep(3)


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="place REAL orders (default: dry-run)")
    ap.add_argument("--allow-real", action="store_true",
                    help="permit trading a NON-demo account (real money)")
    ap.add_argument("--symbol", default="XAUUSDm")
    ap.add_argument("--volume", type=float, default=LOT_PER_LEG, help="lot PER LEG (2 legs/signal)")
    ap.add_argument("--poll", type=int, default=20, help="seconds between polls")
    ap.add_argument("--max-setups", type=int, default=4, help="cap on concurrent live setups")
    ap.add_argument("--expiry-min", type=int, default=RETEST_BARS * M15_MIN,
                    help="cancel an unfilled pending after N min (OB-retest window)")
    ap.add_argument("--horizon-min", type=int, default=HORIZON * M15_MIN,
                    help="market-close a filled trade unresolved after N min (time-stop)")
    args = ap.parse_args()

    from src.exec.broker import Mt5Broker, DryRunBroker
    broker = (Mt5Broker(require_demo=not args.allow_real, magic=SMC_MAGIC, comment=SMC_COMMENT)
              if args.live else DryRunBroker(magic=SMC_MAGIC, comment=SMC_COMMENT))

    acc = broker._account_info_retry()
    server = (getattr(acc, "server", "") or "")
    mode_demo = (acc.trade_mode == broker.mt5.ACCOUNT_TRADE_MODE_DEMO) if acc else True
    login = acc.login if acc else None
    real = (not mode_demo) or ("real" in server.lower())
    if real and args.live and not args.allow_real:
        print(f"  [SAFETY] account {login} server='{server}' looks REAL but --allow-real not set. "
              f"Refusing to place. (dry-run prints only)"); return 2

    # The 2-leg bracket (two same-direction positions, independent SL/TP, BE on one leg)
    # REQUIRES a HEDGING account. On a NETTING account same-symbol positions merge into one
    # net position → per-leg SL/TP + lifecycle accounting break. Fail closed before placing.
    margin_mode = getattr(acc, "margin_mode", None)
    hedging = getattr(broker.mt5, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING", 2)
    if margin_mode is not None and margin_mode != hedging:
        msg = (f"account {login} margin_mode={margin_mode} is NOT hedging — the 2-leg model "
               f"needs a hedging account")
        if args.live:
            print(f"  [SAFETY] {msg}. Refusing to trade.")
            notify.send(f"⛔ <b>SMC DỪNG</b>: {msg}.")
            return 4
        print(f"  [SAFETY][dry-run] WARNING: {msg} — would refuse on --live.")

    # AUTO ledger/namespace: demo shares smc_bot.*; each real account isolates its own files.
    tag = f"_real_{login}" if real else ""
    global LOCK, PAUSE, STATE
    LOCK = Path(f"data/smc_bot{tag}.lock")
    PAUSE = Path(f"data/smc_bot{tag}.pause")
    STATE = Path(f"data/smc_bot_state{tag}.json")
    trade_db.DB_PATH = Path(f"data/smc_trades{tag}.db")
    notify.set_config(f"configs/smc_telegram{tag if real else ''}.json")
    # fall back to the base smc telegram if no per-account file
    if not Path(f"configs/smc_telegram{tag if real else ''}.json").exists():
        notify.set_config("configs/smc_telegram.json")

    _acquire_singleton(LOCK)
    trade_db.init_db()

    label = "REAL 💵" if real else "DEMO"
    mode = "LIVE" if args.live else "DRY-RUN"
    print(f"[smc_bot] {mode} · {label} · acct {login} · server '{server}' · {args.symbol} · "
          f"vol {args.volume}/leg (x2) · poll {args.poll}s · ledger {trade_db.DB_PATH.name}")
    notify.send(f"🤖 <b>SMC bot khởi động</b> — {mode} · {label}\n"
                f"acct {login} · {args.symbol} · {args.volume:g} lot/leg · "
                f"expiry {args.expiry_min}min · horizon {args.horizon_min}min")

    _adopt_orphans(broker, args.symbol)
    if args.live:
        threading.Thread(target=_command_loop, args=(broker, args.symbol), daemon=True).start()

    def _on_exit():
        notify.send("🛑 <b>SMC bot DỪNG</b> (thoát/crash) — lệnh đang mở vẫn có SL/TP ở broker.")
    atexit.register(_on_exit)

    st = _load_state()
    poll_n = 0
    print(f"  [smc_bot] running. pause: create {PAUSE}")
    while True:
        try:
            if broker.login_changed():
                notify.send("⛔ <b>SMC: tài khoản đổi giữa chừng — DỪNG</b> (fail-closed).")
                print("  [smc_bot] login changed — fail-stop"); return 3

            # Periodic orphan sweep (not only at startup): recover any magic order that
            # landed but wasn't tracked (crash between order_send and the DB insert) within
            # ~a minute, so it gets managed/reported and can't double-place.
            poll_n += 1
            if poll_n % 30 == 0:
                _adopt_orphans(broker, args.symbol)

            if _manage_open(broker, args.symbol, now_iso, args.expiry_min, args.horizon_min) != "skip":
                bars = broker.copy_m15(args.symbol, WINDOW_BARS)
                if bars is not None and len(bars) > 50:
                    last_bar = str(bars["time"].iloc[-1])
                    if last_bar != st.get("last_bar"):          # only on a NEWLY closed bar
                        if st.get("retry_bar") != last_bar:     # fresh bar → reset retry budget
                            st["retry"] = 0; st["retry_bar"] = last_bar
                        open_now = trade_db.open_trades()
                        # concurrency = distinct still-live setups (pending + filled). The
                        # realizable backtest (scripts/smc_live_sim.py, no-gate cap 4) shows
                        # capping CONCURRENT setups — NOT one-at-a-time — captures the edge.
                        groups = len({r["group_id"] for r in open_now if r["group_id"]})
                        advance = True
                        if _paused() or groups >= args.max_setups:
                            pass                                # at cap / paused → place nothing
                        else:
                            try:
                                setup = decide(bars)
                            except Exception as e:
                                setup = None
                                print(f"  [{now_iso()}] decide error: {e}")
                            if setup is not None:
                                key = _setup_key(setup.direction, setup.entry, setup.sl)
                                dup = any(_setup_key(r["direction"], r["entry"], r["sl"]) == key
                                          for r in open_now)
                                if dup:
                                    print(f"  [{now_iso()}] setup {key} already live — skip")
                                else:
                                    try:
                                        placed = _place_setup(broker, args.symbol, setup, args.volume)
                                    except Exception as e:
                                        # an order may have landed before the error — recover it
                                        # and do NOT retry this bar (avoid a double-place).
                                        print(f"  [{now_iso()}] place error: {e}; adopting orphans")
                                        _adopt_orphans(broker, args.symbol)
                                        placed = True
                                    if not placed and broker.last_place_retryable:
                                        # transient fail → retry next poll, but cap retries/bar
                                        # so a persistent transient (e.g. MARKET_CLOSED) can't
                                        # hammer order_send until a new bar closes.
                                        st["retry"] = st.get("retry", 0) + 1
                                        if st["retry"] < 3:
                                            advance = False
                                        else:
                                            print(f"  [{now_iso()}] giving up bar after "
                                                  f"{st['retry']} transient retries")
                        if advance:
                            st["last_bar"] = last_bar
                            st["retry"] = 0
                        _save_state(st)
        except KeyboardInterrupt:
            print("  [smc_bot] interrupted"); return 0
        except Exception as e:
            print(f"  [smc_bot] loop error: {e}")
            notify.send(f"⚠️ <b>SMC loop error</b>: {str(e)[:200]}")
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
