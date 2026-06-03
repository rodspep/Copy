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
import threading
import time
from datetime import timedelta, timezone
from pathlib import Path

import requests

import pandas as pd

from src.exec.ug_copier_logic import decide, DEEP_LIMIT
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


def _paused() -> bool:
    return PAUSE_FLAG.exists()


def _stats_text() -> str:
    s = trade_db.summary()
    lines = [f"📊 <b>UG Copier — Track record</b> (tính theo signal, 1 bracket = 1 lệnh)",
             f"Signal: {s['signals']} · đang mở {s['open']} · hủy {s['cancelled']}",
             f"Đã đóng: {s['closed']} (✅ {s['wins']} / ❌ {s['losses']})",
             f"🎯 Win-rate: <b>{s['winrate']:.0%}</b> · 💰 P/L đã chốt: <b>{s['pnl']:+.2f} USD</b>"]
    if s["by_method"]:
        lines.append("— theo method —")
        for m, d in s["by_method"].items():
            extra = f" · {d['open']} đang mở" if d.get("open") else ""
            lines.append(f"  TP1 {m:g}pip: {d['closed']} đóng · win {d['wins']} · "
                         f"{d['pnl']:+.2f} USD{extra}")
    return "\n".join(lines)


def _open_text() -> str:
    rows = trade_db.open_trades()
    if not rows:
        return "📭 Không có lệnh đang mở/chờ."
    out = [f"📂 <b>{len(rows)} lệnh mở/chờ</b>:"]
    for r in rows:
        st = "⏳chờ" if r["status"] == "pending" else "📈mở"
        out.append(f"{st} #{r['id']} {r['direction']} @{r['entry']:.2f} "
                   f"SL {r['sl']:.2f} TP {r['tp']:.2f} (TP1 {r['method_pip']:g})")
    return "\n".join(out)


def _last_text(n: int = 8) -> str:
    rows = trade_db.recent(n)
    if not rows:
        return "📭 Chưa có lệnh nào."
    icons = {"pending": "⏳", "filled": "📈", "cancelled": "🚫",
             "closed_tp": "✅", "closed_sl": "❌"}
    out = [f"📜 <b>{len(rows)} lệnh gần nhất</b>:"]
    for r in rows:
        pl = f" {r['profit']:+.2f}$" if r["profit"] is not None else ""
        out.append(f"{icons.get(r['status'],'•')} #{r['id']} {r['direction']} "
                   f"@{r['entry']:.2f} [{r['status']}{pl}]")
    return "\n".join(out)


def _command_loop(broker, symbol: str) -> None:
    """Long-poll the COPIER's own bot (separate token) → /stats /open /last /flat
    /pause /resume. No conflict with the signal bot (different token)."""
    c = notify.creds()
    if not c:
        print("  [cmd] no copier bot creds — command loop disabled")
        return
    token, chat = c
    api = lambda m: f"https://api.telegram.org/bot{token}/{m}"
    try:
        r = requests.get(api("getUpdates"), params={"timeout": 0}, timeout=20).json()
        offset = r["result"][-1]["update_id"] + 1 if r.get("result") else 0
    except Exception:
        offset = 0
    print("  [cmd] copier command loop online")
    while True:
        try:
            r = requests.get(api("getUpdates"), params={"offset": offset, "timeout": 25},
                             timeout=35).json()
            cmds = {}
            for u in r.get("result", []):
                offset = u["update_id"] + 1
                msg = u.get("message") or u.get("channel_post") or {}
                if str((msg.get("chat") or {}).get("id", "")) != chat:
                    continue
                text = (msg.get("text") or "").strip().lower()
                if text.startswith("/"):
                    cmds[text[1:].split()[0].split("@")[0]] = True   # coalesce per batch
            for w in cmds:
                if w in ("stats",):
                    notify.send(_stats_text())
                elif w in ("open", "positions"):
                    notify.send(_open_text())
                elif w == "last":
                    notify.send(_last_text())
                elif w in ("flat", "cancelall", "flatten"):
                    n = broker.cancel_all_pendings(symbol)
                    notify.send("🧹 Đã hủy " + (f"{n} lệnh chờ." if n is not None
                                else "(query lỗi)") + " (vị thế đang mở KHÔNG đóng — tự làm trong MT5).")
                elif w in ("pause", "stop"):
                    PAUSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
                    PAUSE_FLAG.write_text("paused")
                    notify.send("⏸ <b>Tạm dừng</b> đặt lệnh mới (lệnh đang mở vẫn quản lý). /resume để chạy lại.")
                elif w in ("resume", "start"):
                    PAUSE_FLAG.unlink(missing_ok=True)
                    notify.send("▶️ <b>Tiếp tục</b> đặt lệnh.")
                elif w in ("help", "commands"):
                    notify.send("Lệnh: /stats /open /last /flat /pause /resume")
        except Exception as e:
            print(f"  [cmd] loop error {e}")
            time.sleep(5)


def _vn(ts_iso: str) -> str:
    try:
        t = pd.Timestamp(ts_iso)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        return t.tz_convert(VN).strftime("%H:%M:%S %d/%m GMT+7")
    except Exception:
        return str(ts_iso)


def _method_label(pip) -> str:
    return {50.0: "PP2 scalp", 100.0: "method-100", 150.0: "PRI 150"}.get(float(pip or 0), f"TP1 {pip}")


def _leg_label(leg) -> str:
    """Human label for a bracket leg in notifications ('khớp tp1' read like 'hit TP1';
    this disambiguates: it's the leg that AIMS for that TP)."""
    return {"tp1": "chân 1 · chốt TP1", "tp3": "chân 2 · runner TP3"}.get(leg or "tp1", str(leg))


def _place_msg(orders, sig, lag: float) -> str:
    o = orders[0]
    arrow = "🟢 BUY" if o.side == "long" else "🔴 SELL"
    nlegs = len(orders)
    if nlegs > 1:
        tps = " · ".join(f"{'TP1' if x.leg == 'tp1' else 'TP3'} {x.tp:.2f}" for x in orders)
        plan = "2 chân TP1+TP3, SL→BE sau TP1"
    else:
        tps = f"TP1 {o.tp:.2f}"
        plan = "TP1 full"
    return (f"📥 <b>COPY · {arrow} XAUUSDm</b>\n"
            f"➡️ Entry <b>{o.entry:.2f}</b> · 🛑 SL {o.sl:.2f}\n"
            f"🎯 {tps}\n"
            f"📦 Vol {o.volume}×{nlegs} · {_method_label(o.tp1_pip)} · {plan}\n"
            f"🕐 từ UG {_vn(sig.get('ts',''))} · lag {lag:.0f}s")

FEED = Path("data/ug/live_signals.jsonl")
PAUSE_FLAG = Path("data/ug/copier_paused.flag")
# State path is set per run-mode (live vs dry) so a dry-run can never poison the
# live dedup/exposure record. Assigned in main().
STATE = Path("data/ug/copier_state.json")


def _key(sig: dict) -> str:
    # Dedup by CONTENT (not ts): UG reposts the same signal ~1 min apart with a new
    # timestamp — those must collapse to ONE order, not one per repost.
    return "|".join(str(sig.get(k)) for k in ("direction", "entry_low", "entry_high", "sl"))


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
    ap.add_argument("--max-signal-age-min", type=float, default=15.0,
                    help="skip signals older than N minutes (UG edge is fresh; guards "
                         "against stale reposts / feed backlog being placed late)")
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
    # copier's own command bot (separate token) → /stats /open /last /flat /pause
    threading.Thread(target=_command_loop, args=(broker, args.symbol), daemon=True).start()
    notify.send(f"📥 <b>UG Copier khởi động</b> — {mode} · {args.symbol} · vol "
                f"{args.volume} · lọc TP1∈{{50,100,150}} · đang theo dõi UG.\n"
                f"Lệnh: /stats /open /last /flat /pause /resume")

    while True:
        try:
            px = broker.get_price(args.symbol)
            if not px:
                print(f"  [{now_iso()}] no price for {args.symbol}; retry")
                time.sleep(args.poll); continue
            mid = px["mid"]

            # 1) New signals → decide + place (skipped while paused; management still runs).
            for sig in ([] if _paused() else _read_feed()):
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
                # STALE GUARD: UG's edge is a fresh pull-back; a signal we only see
                # long after its post (old repost, feed backlog, key-format migration)
                # must NOT be placed. A bad/unparseable ts (lag<0) is also untrusted.
                # Mark done so it is never reconsidered.
                if lag < 0 or lag > args.max_signal_age_min * 60:
                    print(f"  [{now_iso()}] STALE skip {sig.get('direction')} "
                          f"(lag {lag:.0f}s > {args.max_signal_age_min:g}min) — not placing")
                    st["done"][k] = {"stale": lag, "at": now_iso()}
                    _save_state(st)        # durable: never reconsider across restart
                    continue
                d = decide(sig, mid, volume=args.volume, real_mode=args.allow_real)
                if d.action == "skip":
                    print(f"  [{now_iso()}] (lag {lag:.1f}s) SKIP {sig.get('direction')} — {d.reason}")
                    st["done"][k] = {"skipped": d.reason, "at": now_iso()}
                    _save_state(st)        # durable: skip-notify fires once, not per restart
                    # Notify skips ONLY for the proven 50pip edge (low volume of skips
                    # under deep-limit) so you see the bot saw it + why; other methods stay quiet.
                    _tps = sig.get("tps_pip") if isinstance(sig.get("tps_pip"), dict) else {}
                    try:
                        _tp1 = float(_tps.get(1) or _tps.get("1") or 0)
                    except (TypeError, ValueError):
                        _tp1 = 0.0
                    if _tp1 == 50.0:
                        notify.send(f"👀 <b>Bỏ qua signal 50pip</b> ({sig.get('direction')}) — {d.reason}")
                elif (exp := broker.open_exposure(args.symbol)) is None:
                    print(f"  [{now_iso()}] HOLD {sig.get('direction')} — exposure unknown "
                          f"(broker query failed); not placing this cycle")
                    continue
                elif exp + len(d.orders) > args.max_open:
                    print(f"  [{now_iso()}] HOLD {sig.get('direction')} — max-open "
                          f"{args.max_open} would be exceeded (exposure {exp} + "
                          f"{len(d.orders)} legs); reconsider next poll")
                    continue        # don't mark done → reconsider when exposure frees
                else:
                    # Pre-mark BEFORE placing so a crash mid-place can't double-place
                    # on restart (k stays in done). Any leg that DID land carries its
                    # own SL/TP, so a crash leaves it safe (just possibly unmanaged).
                    st["done"][k] = {"status": "placing", "at": now_iso()}
                    _save_state(st)
                    placed = []          # [(Order, ticket)] for legs that actually landed
                    for o in d.orders:
                        ticket = broker.place_limit(args.symbol, o)
                        if ticket is None:
                            print(f"  [{now_iso()}] place FAILED {o.order_type} {o.leg} @ {o.entry}")
                            continue       # not retried (safety); other leg may still place
                        print(f"  [{now_iso()}] (lag {lag:.1f}s) PLACED {o.order_type} {o.leg} "
                              f"{args.symbol} {o.volume} @ {o.entry} sl={o.sl} tp={o.tp} ticket={ticket}")
                        placed.append((o, ticket))
                    if not placed:
                        st["done"][k] = {"status": "place_failed", "at": now_iso()}
                        _save_state(st)
                        continue
                    msg_id = notify.send(_place_msg([o for o, _ in placed], sig, lag))
                    recs = []
                    for o, ticket in placed:
                        tid = trade_db.insert({
                            "signal_ts": sig.get("ts"), "direction": o.side,
                            "method_pip": o.tp1_pip, "order_type": o.order_type,
                            "entry": o.entry, "sl": o.sl, "tp": o.tp, "volume": o.volume,
                            "ticket": ticket, "status": "pending", "tg_msg_id": msg_id,
                            "created_at": now_iso(), "leg": o.leg, "group_id": k})
                        recs.append({"trade_id": tid, "ticket": ticket, "leg": o.leg})
                    st["done"][k] = {"placed": recs}
                    _save_state(st)

            # 2) Lifecycle management of DB-tracked trades (pending→filled→closed),
            #    with a Telegram reply + P/L on each transition.
            live_tickets = broker.pending_tickets(args.symbol)
            if live_tickets is None:        # query FAILED — don't mistake for 'all gone'
                print(f"  [{now_iso()}] pending query failed; skip management this cycle")
                time.sleep(args.poll); continue

            opens = trade_db.open_trades()
            # The don't-chase trigger for a whole bracket is the SCALP target (TP1),
            # not each leg's own TP — the tp3 runner's TP is far away. Map group→TP1.
            group_tp1 = {r["group_id"]: r["tp"] for r in opens
                         if r["leg"] == "tp1" and r["group_id"]}

            for r in opens:
                tid, tk = r["id"], r["ticket"]
                long = r["direction"] == "long"
                leg = r["leg"] or "tp1"
                if r["status"] == "pending":
                    if tk in live_tickets:
                        # DEEP_LIMIT: a resting pull-back limit is NOT cancelled just because
                        # price touched TP1 — we wait for the pull-back to fill (cancel only on
                        # expiry). Chase mode (DEEP_LIMIT=False) cancels on the group's TP1.
                        trig = group_tp1.get(r["group_id"], r["tp"])
                        reached = (not DEEP_LIMIT) and ((mid >= trig) if long else (mid <= trig))
                        age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(r["created_at"])).total_seconds() / 60
                        if reached or age > args.expiry_min:
                            why = "giá chạm TP1, chưa khớp" if reached else f"hết hạn {age:.0f}min"
                            if broker.cancel(tk):
                                trade_db.update(tid, status="cancelled", closed_at=now_iso(), note=why)
                                notify.send(f"🚫 <b>Hủy {_leg_label(leg)}</b> — {why}", reply_to=r["tg_msg_id"])
                                print(f"  [{now_iso()}] CANCEL {leg} ticket {tk} — {why}")
                    else:
                        fi = broker.fill_info(tk)
                        if fi == "unknown":
                            continue          # transient history-query failure — retry next poll
                        if fi:
                            trade_db.update(tid, status="filled", position_id=fi["position_id"],
                                            fill_price=fi["fill_price"], filled_at=now_iso())
                            notify.send(f"📌 <b>Đã vào lệnh</b> ({_leg_label(leg)}) "
                                        f"@ {fi['fill_price']:.2f}", reply_to=r["tg_msg_id"])
                            print(f"  [{now_iso()}] FILLED {leg} ticket {tk} @ {fi['fill_price']:.2f}")
                        else:
                            trade_db.update(tid, status="cancelled", closed_at=now_iso(),
                                            note="pending vanished")
                            notify.send(f"🚫 <b>Lệnh chờ ({_leg_label(leg)}) đã biến mất</b> "
                                        f"(hủy ngoài?)", reply_to=r["tg_msg_id"])
                elif r["status"] == "filled":
                    ci = broker.closed_info(r["position_id"])
                    if ci:
                        cp, pnl = ci["close_price"], ci["profit"]
                        # classify by nearest level; r["sl"] reflects any BE move already saved
                        reason = ("closed_tp" if abs(cp - r["tp"]) <= abs(cp - r["sl"])
                                  else "closed_sl")
                        trade_db.update(tid, status=reason, close_price=cp, profit=pnl,
                                        closed_at=now_iso())
                        icon = "✅ WIN" if pnl > 0 else ("❌ LOSS" if pnl < 0 else "⚪ BE")
                        if reason == "closed_tp":
                            hit = "TP1" if leg == "tp1" else "TP3"
                        elif abs(cp - r["entry"]) <= 0.30:        # ~spread → break-even exit
                            hit = "hòa vốn (BE)"
                        else:
                            hit = "SL"
                        notify.send(f"{icon} <b>{_leg_label(leg)}</b> — đóng tại {hit} @ {cp:.2f}\n"
                                    f"💰 Lời/Lỗ: <b>{pnl:+.2f} USD</b>", reply_to=r["tg_msg_id"])
                        print(f"  [{now_iso()}] CLOSED {leg} {hit} @ {cp:.2f} pnl {pnl:+.2f}")

            # 3) Break-even reconciliation: any OPEN runner (tp3) whose scalp (tp1)
            #    sibling already WON → move the runner's SL to entry. Standing check
            #    (re-fetches siblings fresh) so it retries until the broker accepts.
            for r in opens:
                if (r["leg"] == "tp3" and r["status"] == "filled"
                        and r["position_id"] and r["group_id"]):
                    sibs = trade_db.siblings(r["group_id"])
                    tp1_won = any(s["leg"] == "tp1" and s["status"] == "closed_tp" for s in sibs)
                    if tp1_won and r["sl"] is not None and abs(r["sl"] - r["entry"]) > 1e-6:
                        if broker.modify_sl(r["position_id"], r["entry"]):
                            trade_db.update(r["id"], sl=r["entry"], note="SL→BE sau TP1")
                            notify.send("🛡️ <b>Runner TP3: SL dời về hòa vốn</b> (TP1 đã thắng)",
                                        reply_to=r["tg_msg_id"])
                            print(f"  [{now_iso()}] BE move pos {r['position_id']} -> {r['entry']}")
        except Exception as e:
            print(f"  [{now_iso()}] loop error {e}")
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
