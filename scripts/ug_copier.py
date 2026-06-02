"""UG signal copier — filter, place a pending LIMIT, manage. MT5 / VPS.

Reads parsed UG signals from a file feed (data/ug/live_signals.jsonl), and for
each NEW signal: decide() [filter TP1∈{50,100,150}; entry per method (50→mid,
100→near, 150→deep); TP=TP1 from that entry; skip if price past TP1 / wrong side]
→ place a pending LIMIT (or DRY-log). Tracks placed pendings and CANCELS any that
hasn't filled once price reaches TP1 (don't chase) or after an expiry.

DESIGN — TP1-only, all-out: we set ONE TP (=TP1) and the broker SL on the pending.
Once it fills, the position runs to that SL/TP under MT5 (we do NOT do UG's
TP1..TP4 partials / break-even / trailing). That is intentional for now.

SAFETY: DRY-RUN by default (real prices, no orders). --live places real orders and
ABORTS unless the account is DEMO (override: --allow-real). The broker RE-checks
the account on every send. Singleton lock prevents two copiers. Exposure cap
(--max-open). Volume 0.01. Only orders tagged with our MAGIC are touched. Run
inside the MT5 interactive session (like the signal bot).

Feed line = one parsed UG signal dict (see scripts/parse_ug_export.py), e.g.:
  {"ts":"...","direction":"long","entry_low":4468,"entry_high":4458,"sl":4448,
   "tps_pip":{"1":150,"2":200,"3":300,"4":400}}

Run (VPS, dry):  python -X utf8 -m scripts.ug_copier
Run (VPS, live): python -X utf8 -m scripts.ug_copier --live --symbol XAUUSDm
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd

from src.exec.ug_copier_logic import decide
from src.exec import notify, trade_db

PIP = 0.1
VN = timezone(timedelta(hours=7))


_LOCK_HANDLE = None


def _acquire_singleton(path: Path) -> None:
    """OS-held exclusive lock (Windows msvcrt). A second copier physically cannot
    acquire it → it aborts. Stronger than a heartbeat file (can't be defeated by a
    restart script deleting the lock). The handle is held for the process lifetime."""
    global _LOCK_HANDLE
    import msvcrt
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOCK_HANDLE = path.open("a+b")
    try:
        _LOCK_HANDLE.seek(0)        # ALWAYS lock byte 0 (not the append position)
        msvcrt.locking(_LOCK_HANDLE.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit(f"ABORT: another copier already owns {path.name}")
    try:                           # write our PID for observability (owning handle)
        _LOCK_HANDLE.seek(0)
        _LOCK_HANDLE.write(str(os.getpid()).encode("ascii"))
        _LOCK_HANDLE.flush()
    except OSError:
        pass


def _vn(ts_iso: str) -> str:
    try:
        t = pd.Timestamp(ts_iso)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        return t.tz_convert(VN).strftime("%H:%M:%S %d/%m GMT+7")
    except Exception:
        return str(ts_iso)


def _method_label(pip) -> str:
    return {50.0: "PP2 scalp", 100.0: "method-100", 150.0: "PRI 150"}.get(float(pip or 0), f"TP1 {pip}")


def _place_msg(o, sig, lag: float) -> str:
    arrow = "🟢 BUY" if o.side == "long" else "🔴 SELL"
    return (f"📥 <b>COPY · {arrow} XAUUSDm</b>\n"
            f"➡️ Entry <b>{o.entry:.2f}</b> · 🛑 SL {o.sl:.2f} · 🎯 TP1 {o.tp:.2f}\n"
            f"📦 Vol {o.volume} · {_method_label(o.tp1_pip)} (TP1 {o.tp1_pip:g}pip)\n"
            f"🕐 từ UG {_vn(sig.get('ts',''))} · lag {lag:.0f}s")

FEED = Path("data/ug/live_signals.jsonl")
# State path is set per run-mode (live vs dry) so a dry-run can never poison the
# live dedup/exposure record. Assigned in main().
STATE = Path("data/ug/copier_state.json")


def _key(sig: dict) -> str:
    return "|".join(str(sig.get(k)) for k in ("ts", "direction", "entry_low", "entry_high", "sl"))


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"done": {}}        # key -> {ticket, order, placed_at} | {skipped: reason}


def _save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")        # atomic write: tmp + replace
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=0), encoding="utf-8")
    os.replace(tmp, STATE)




def _read_feed() -> list[dict]:
    if not FEED.exists():
        return []
    out = []
    for line in FEED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="place REAL orders (default: dry-run)")
    ap.add_argument("--symbol", default="XAUUSDm")
    ap.add_argument("--volume", type=float, default=0.01)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--expiry-min", type=int, default=240, help="cancel unfilled pending after N min")
    ap.add_argument("--max-open", type=int, default=5, help="hard cap on concurrent placed orders")
    ap.add_argument("--allow-real", action="store_true",
                    help="permit --live on a REAL account (default: live allowed on DEMO only)")
    args = ap.parse_args()

    global STATE
    mode_tag = "live" if args.live else "dry"
    STATE = Path(f"data/ug/copier_state_{mode_tag}.json")

    # Singleton lock — refuse to start if another copier of this mode is alive
    # (a fresh heartbeat lock). Prevents two processes double-placing the same
    # signal / bypassing the exposure cap.
    LOCK = STATE.with_name(f"copier_{mode_tag}.lock")
    _acquire_singleton(LOCK)        # OS lock — a 2nd copier can't start, period

    from src.exec.broker import Mt5Broker, DryRunBroker
    broker = Mt5Broker(require_demo=not args.allow_real) if args.live else DryRunBroker()
    mode = "LIVE" if args.live else "DRY-RUN"
    print(f"UG copier {mode} · {args.symbol} · vol {args.volume} · poll {args.poll}s "
          f"· expiry {args.expiry_min}min")
    if args.live:
        # Safety: --live is allowed only on a DEMO account unless explicitly forced.
        acc = broker.mt5.account_info()
        if acc is None:
            raise SystemExit("ABORT: cannot read account_info")
        is_demo = acc.trade_mode == broker.mt5.ACCOUNT_TRADE_MODE_DEMO
        print(f"  account {acc.login} · {acc.server} · "
              f"{'DEMO' if is_demo else 'REAL/CONTEST'} · balance {acc.balance}")
        if not is_demo and not args.allow_real:
            raise SystemExit("ABORT: --live on a non-DEMO account. Pass --allow-real to trade real money.")
        print("  !! LIVE order placement ENABLED !!")

    st = _load_state()
    trade_db.init_db()
    now_iso = lambda: pd.Timestamp.now(tz="UTC").isoformat()
    notify.send(f"📥 <b>UG Copier khởi động</b> — {mode} · {args.symbol} · vol "
                f"{args.volume} · lọc TP1∈{{50,100,150}} · đang theo dõi UG.")

    while True:
        try:
            px = broker.get_price(args.symbol)
            if not px:
                print(f"  [{now_iso()}] no price for {args.symbol}; retry")
                time.sleep(args.poll); continue
            mid = px["mid"]

            # 1) New signals → decide + place.
            for sig in _read_feed():
                k = _key(sig)
                if k in st["done"]:
                    continue
                # End-to-end latency: signal's Telegram time → now (covers
                # telegram→listener→file→this poll). Logged so we can see if the
                # copier is too slow vs how fast UG's price moves.
                try:
                    lag = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(sig["ts"])).total_seconds()
                except Exception:
                    lag = -1.0
                d = decide(sig, mid, volume=args.volume)
                if d.action == "skip":
                    print(f"  [{now_iso()}] (lag {lag:.1f}s) SKIP {sig.get('direction')} — {d.reason}")
                    st["done"][k] = {"skipped": d.reason, "at": now_iso()}
                elif (exp := broker.open_exposure(args.symbol)) is None:
                    print(f"  [{now_iso()}] HOLD {sig.get('direction')} — exposure unknown "
                          f"(broker query failed); not placing this cycle")
                    continue
                elif exp >= args.max_open:
                    print(f"  [{now_iso()}] HOLD {sig.get('direction')} — max-open "
                          f"{args.max_open} reached (exposure {exp}); reconsider next poll")
                    continue        # don't mark done → reconsider when exposure frees
                else:
                    o = d.order
                    # Pre-mark BEFORE order_send so a crash between send and save can
                    # never double-place on restart. If we then crash, the order (if
                    # it was sent) still carries its own SL/TP — safe, just unmanaged.
                    st["done"][k] = {"status": "placing", "at": now_iso(), "order": o.__dict__}
                    _save_state(st)
                    ticket = broker.place_limit(args.symbol, o)
                    if ticket is None:
                        print(f"  [{now_iso()}] place FAILED {o.order_type} @ {o.entry}")
                        st["done"][k] = {"status": "place_failed", "at": now_iso(),
                                         "order": o.__dict__}   # not retried (safety)
                        _save_state(st)
                        continue
                    print(f"  [{now_iso()}] (lag {lag:.1f}s signal→placed) PLACED "
                          f"{o.order_type} {args.symbol} {o.volume} "
                          f"@ {o.entry} sl={o.sl} tp={o.tp} ticket={ticket}")
                    msg_id = notify.send(_place_msg(o, sig, lag))   # → Telegram, keep id
                    tid = trade_db.insert({
                        "signal_ts": sig.get("ts"), "direction": o.side,
                        "method_pip": o.tp1_pip, "order_type": o.order_type,
                        "entry": o.entry, "sl": o.sl, "tp": o.tp, "volume": o.volume,
                        "ticket": ticket, "status": "pending", "tg_msg_id": msg_id,
                        "created_at": now_iso()})
                    st["done"][k] = {"trade_id": tid, "ticket": ticket}
                    _save_state(st)

            # 2) Lifecycle management of DB-tracked trades (pending→filled→closed),
            #    with a Telegram reply + P/L on each transition.
            live_tickets = broker.pending_tickets(args.symbol)
            if live_tickets is None:        # query FAILED — don't mistake for 'all gone'
                print(f"  [{now_iso()}] pending query failed; skip management this cycle")
                time.sleep(args.poll); continue

            for r in trade_db.open_trades():
                tid, tk = r["id"], r["ticket"]
                long = r["direction"] == "long"
                if r["status"] == "pending":
                    if tk in live_tickets:
                        # cancel if price reached TP1 (don't chase) or order expired
                        reached = (mid >= r["tp"]) if long else (mid <= r["tp"])
                        age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(r["created_at"])).total_seconds() / 60
                        if reached or age > args.expiry_min:
                            why = "giá chạm TP1, chưa khớp" if reached else f"hết hạn {age:.0f}min"
                            if broker.cancel(tk):
                                trade_db.update(tid, status="cancelled", closed_at=now_iso(), note=why)
                                notify.send(f"🚫 <b>Hủy lệnh chờ</b> — {why}", reply_to=r["tg_msg_id"])
                                print(f"  [{now_iso()}] CANCEL ticket {tk} — {why}")
                    else:
                        fi = broker.fill_info(tk)
                        if fi == "unknown":
                            continue          # transient history-query failure — retry next poll
                        if fi:
                            trade_db.update(tid, status="filled", position_id=fi["position_id"],
                                            fill_price=fi["fill_price"], filled_at=now_iso())
                            notify.send(f"🎯 <b>Đã khớp</b> @ {fi['fill_price']:.2f}",
                                        reply_to=r["tg_msg_id"])
                            print(f"  [{now_iso()}] FILLED ticket {tk} @ {fi['fill_price']:.2f}")
                        else:
                            trade_db.update(tid, status="cancelled", closed_at=now_iso(),
                                            note="pending vanished")
                            notify.send("🚫 <b>Lệnh chờ đã biến mất</b> (hủy ngoài?)",
                                        reply_to=r["tg_msg_id"])
                elif r["status"] == "filled":
                    ci = broker.closed_info(r["position_id"])
                    if ci:
                        cp, pnl = ci["close_price"], ci["profit"]
                        # classify by which level the close sits nearest
                        reason = ("closed_tp" if abs(cp - r["tp"]) <= abs(cp - r["sl"])
                                  else "closed_sl")
                        trade_db.update(tid, status=reason, close_price=cp, profit=pnl,
                                        closed_at=now_iso())
                        icon = "✅ WIN" if pnl > 0 else ("❌ LOSS" if pnl < 0 else "⚪ BE")
                        hit = "TP1" if reason == "closed_tp" else "SL"
                        notify.send(f"{icon} — chạm {hit} @ {cp:.2f}\n"
                                    f"💰 Lời/Lỗ: <b>{pnl:+.2f} USD</b>", reply_to=r["tg_msg_id"])
                        print(f"  [{now_iso()}] CLOSED {hit} @ {cp:.2f} pnl {pnl:+.2f}")
        except Exception as e:
            print(f"  [{now_iso()}] loop error {e}")
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
