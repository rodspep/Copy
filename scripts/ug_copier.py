"""UG signal copier — filter, place a zone-aware bracket, manage. MT5 / VPS.

Reads parsed UG signals from a file feed (data/ug/live_signals.jsonl). For each NEW
signal: decide() [filter TP1∈{50,100,150} method types; ZONE-AWARE entry: MARKET when
price is at/better than the anchor (mid), else a LIMIT at the anchor waiting for the
pull-back; skip if price is past the zone the wrong way]. Auto-detects demo vs real and
keeps a separate ledger/state/telegram per account.

EXIT — UNIFIED bracket for ALL methods: two equal legs at the same entry+SL —
  leg 'tp1' : FIXED 50 pip   (the proven high-WR scalp; ~88% across all method entries)
  leg 'tp3' : FIXED 150 pip  (runner; its SL → break-even once the tp1 leg wins)
The signal's published far TPs (100/150/300...) are NOT used as targets; tp1_pip only
identifies the method (50/100/150) for /stats + dedup. Stale-guard, daily-loss
circuit-breaker, heartbeat + external watchdog, orphan recovery on restart.

SAFETY: DRY-RUN by default. --live places real orders, ABORTS on a non-DEMO account
unless --allow-real. Account is re-checked every send (cancel/modify too) and the loop
fail-stops on a mid-run login change; ledger-feeding reads fail closed on mismatch.
Singleton lock; exposure cap (--max-open); vol 0.01/leg. MAGIC-tagged orders only.
Run inside the MT5 interactive session.

Feed line = one parsed UG signal dict (see scripts/parse_ug_export.py), e.g.:
  {"ts":"...","direction":"long","entry_low":4468,"entry_high":4458,"sl":4448,
   "tps_pip":{"1":150,"2":200,"3":300,"4":400}}

Run (VPS, dry):  python -X utf8 -m scripts.ug_copier
Run (VPS, live): python -X utf8 -m scripts.ug_copier --live --symbol XAUUSDm
"""
from __future__ import annotations

import argparse
import atexit
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


ACCOUNT_LABEL = "?"        # "DEMO" | "MAIN" — set in main() from the detected account


def _paused() -> bool:
    return PAUSE_FLAG.exists()


def _stats_text() -> str:
    s = trade_db.summary()
    lines = [f"📊 <b>UG Copier [{ACCOUNT_LABEL}] — Track record</b> (theo signal, 1 bracket = 1 lệnh)",
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
    out = [f"📂 <b>[{ACCOUNT_LABEL}] {len(rows)} lệnh mở/chờ</b>:"]
    for r in rows:
        st = "⏳chờ" if r["status"] == "pending" else "📈mở"
        m = r["method_pip"]
        tag = f"TP1 {m:g}" if m is not None else (r["leg"] or "orphan")
        out.append(f"{st} #{r['id']} {r['direction']} @{r['entry']:.2f} "
                   f"SL {r['sl']:.2f} TP {r['tp']:.2f} ({tag})")
    return "\n".join(out)


def _last_text(n: int = 8) -> str:
    rows = trade_db.recent(n)
    if not rows:
        return "📭 Chưa có lệnh nào."
    icons = {"pending": "⏳", "filled": "📈", "cancelled": "🚫",
             "closed_tp": "✅", "closed_sl": "❌"}
    out = [f"📜 <b>[{ACCOUNT_LABEL}] {len(rows)} lệnh gần nhất</b>:"]
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
    # TP1 pip → UG signal family (confirmed from a week of signal headers):
    #   50 = Phương Pháp 2 + Ai Scalp Signal · 100 = PRI GOLD · 150 = Ai Signals
    return {50.0: "PP2/Scalp", 100.0: "PRI GOLD", 150.0: "Ai Signals"}.get(float(pip or 0), f"TP1 {pip}")


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
    if o.order_type.endswith("market"):
        style = "⚡ <b>VÀO NGAY</b> (market — đã khớp, không có tin khớp riêng)"
    else:
        style = "⏳ <b>LỆNH CHỜ</b> (limit — đợi giá hồi về; sẽ báo khi khớp)"
    return (f"📥 <b>COPY · {arrow} XAUUSDm</b>\n"
            f"{style}\n"
            f"➡️ Entry <b>{o.entry:.2f}</b> · 🛑 SL {o.sl:.2f}\n"
            f"🎯 {tps}\n"
            f"📦 Vol {o.volume}×{nlegs} · {_method_label(o.tp1_pip)} · {plan}\n"
            f"🕐 từ UG {_vn(sig.get('ts',''))} · lag {lag:.0f}s")

FEED = Path("data/ug/live_signals.jsonl")
PAUSE_FLAG = Path("data/ug/copier_paused.flag")
HEARTBEAT = Path("data/ug/copier_heartbeat.json")   # per-account; refreshed each poll
# State path is set per run-mode (live vs dry) so a dry-run can never poison the
# live dedup/exposure record. Assigned in main().
STATE = Path("data/ug/copier_state.json")


def _key(sig: dict) -> str:
    # Dedup by CONTENT (not ts): UG reposts the same signal ~1 min apart with a new
    # timestamp — those must collapse to ONE order, not one per repost. INCLUDE TP1 so a
    # different method (50 vs 100/150) with the same zone/SL isn't wrongly deduped, and so
    # this doubles safely as the bracket group_id (both legs share the same TP1).
    tps = sig.get("tps_pip") if isinstance(sig.get("tps_pip"), dict) else {}
    try:
        tp1 = f"{float(tps.get(1) or tps.get('1')):g}"      # canonical (50/50.0/'50' → '50')
    except (TypeError, ValueError):
        tp1 = "?"
    return "|".join(str(sig.get(k)) for k in ("direction", "entry_low", "entry_high", "sl")) + f"|{tp1}"


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"done": {}}        # key -> {ticket, order, placed_at} | {skipped: reason}


def _save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")        # atomic write: tmp + replace
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=0), encoding="utf-8")
    os.replace(tmp, STATE)




def _adopt_orphans(broker, symbol: str) -> None:
    """On startup, adopt any of OUR magic orders/positions at the broker that aren't in
    the trade ledger (e.g. a crash between order_send and the DB insert left them
    untracked). They carry their own SL/TP so they're bounded, but adopting them lets
    the lifecycle manager cancel/close + report them. Each becomes its own group (no
    BE-linking — that history is lost — but it's protected by its own SL)."""
    info = broker.list_magic(symbol)
    if not info:
        print("  [adopt] list_magic unavailable — skipping orphan scan")
        return
    known_ord, known_pos = set(), set()
    for r in trade_db.open_trades():
        if r["ticket"]:
            known_ord.add(int(r["ticket"]))
        if r["position_id"]:
            known_pos.add(int(r["position_id"]))
    now = pd.Timestamp.now(tz="UTC").isoformat()
    adopted = 0
    for o in info["pendings"]:
        if o["ticket"] in known_ord:
            continue
        long = o["type"] == info["buy_limit"]
        trade_db.insert({"direction": "long" if long else "short",
                         "order_type": "buy_limit" if long else "sell_limit",
                         "entry": o["entry"], "sl": o["sl"], "tp": o["tp"], "volume": o["volume"],
                         "ticket": o["ticket"], "status": "pending", "created_at": now,
                         "leg": "orphan", "group_id": f"orphan:{o['ticket']}",
                         "note": "adopted orphan pending"})
        adopted += 1
    for p in info["positions"]:
        # Skip if tracked as a position OR if it came from a tracked PENDING that filled
        # while we were offline: in MT5 a filled limit's position_id == its order ticket,
        # so a position_id matching a known pending ticket is already ours (it'll resolve
        # via fill_info). Avoids a duplicate orphan row for one live position.
        if p["position_id"] in known_pos or p["position_id"] in known_ord:
            continue
        long = p["type"] == info["pos_buy"]
        trade_db.insert({"direction": "long" if long else "short",
                         "order_type": "buy_limit" if long else "sell_limit",
                         "entry": p["entry"], "sl": p["sl"], "tp": p["tp"], "volume": p["volume"],
                         "ticket": p["position_id"], "position_id": p["position_id"],
                         "fill_price": p["fill_price"], "status": "filled", "created_at": now,
                         "filled_at": now, "leg": "orphan", "group_id": f"orphan:{p['position_id']}",
                         "note": "adopted orphan position"})
        adopted += 1
    if adopted:
        print(f"  [adopt] recovered {adopted} untracked magic order(s)/position(s)")
        notify.send(f"♻️ <b>Nhận lại {adopted} lệnh mồ côi</b> (chưa được theo dõi) vào quản lý.")


def _manage_open(broker, symbol, mid, now_iso, expiry_min):
    """Lifecycle management of DB-tracked trades over one poll: pending→fill/cancel,
    filled→close (+P/L), then break-even of the runner. Pure orchestration over
    trade_db + broker + notify. Returns 'skip' if the pending query failed (caller should
    skip the rest of this cycle), else None. Extracted from the loop so it is testable
    end-to-end (place→fill→BE→close)."""
    live_tickets = broker.pending_tickets(symbol)
    if live_tickets is None:        # query FAILED — don't mistake for 'all gone'
        print(f"  [{now_iso()}] pending query failed; skip management this cycle")
        return "skip"

    opens = trade_db.open_trades()
    # The don't-chase trigger for a whole bracket is the SCALP target (TP1), not each
    # leg's own TP — the tp3 runner's TP is far away. Map group→TP1.
    group_tp1 = {r["group_id"]: r["tp"] for r in opens
                 if r["leg"] == "tp1" and r["group_id"]}

    for r in opens:
        tid, tk = r["id"], r["ticket"]
        long = r["direction"] == "long"
        leg = r["leg"] or "tp1"
        if r["status"] == "pending":
            if tk in live_tickets:
                trig = group_tp1.get(r["group_id"], r["tp"])
                reached = (not DEEP_LIMIT) and ((mid >= trig) if long else (mid <= trig))
                age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(r["created_at"])).total_seconds() / 60
                if reached or age > expiry_min:
                    why = "giá chạm TP1, chưa khớp" if reached else f"hết hạn {age:.0f}min"
                    if broker.cancel(tk):
                        trade_db.update(tid, status="cancelled", closed_at=now_iso(), note=why)
                        notify.send(f"🚫 <b>Hủy {_leg_label(leg)}</b> — {why}", reply_to=r["tg_msg_id"])
                        print(f"  [{now_iso()}] CANCEL {leg} ticket {tk} — {why}")
            else:
                fi = broker.fill_info(tk)
                if fi == "unknown":
                    continue          # transient query failure — retry next poll
                if fi:
                    # entry := ACTUAL fill (a limit can gap-fill better than its price) so
                    # break-even later moves the runner SL to the true entry.
                    trade_db.update(tid, status="filled", position_id=fi["position_id"],
                                    fill_price=fi["fill_price"], entry=fi["fill_price"],
                                    filled_at=now_iso())
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
                reason = ("closed_tp" if abs(cp - r["tp"]) <= abs(cp - r["sl"]) else "closed_sl")
                trade_db.update(tid, status=reason, close_price=cp, profit=pnl, closed_at=now_iso())
                icon = "✅ WIN" if pnl > 0 else ("❌ LOSS" if pnl < 0 else "⚪ BE")
                if reason == "closed_tp":
                    hit = "TP1" if leg == "tp1" else "TP3"
                elif abs(cp - r["entry"]) <= 0.30:
                    hit = "hòa vốn (BE)"
                else:
                    hit = "SL"
                notify.send(f"{icon} <b>{_leg_label(leg)}</b> — đóng tại {hit} @ {cp:.2f}\n"
                            f"💰 Lời/Lỗ: <b>{pnl:+.2f} USD</b>", reply_to=r["tg_msg_id"])
                print(f"  [{now_iso()}] CLOSED {leg} {hit} @ {cp:.2f} pnl {pnl:+.2f}")

    # Break-even: any OPEN runner (tp3) whose scalp (tp1) sibling already WON → move the
    # runner's SL to entry. Standing check (re-fetches siblings) → retries until accepted.
    for r in opens:
        if (r["leg"] == "tp3" and r["status"] == "filled" and r["position_id"] and r["group_id"]):
            sibs = trade_db.siblings(r["group_id"])
            tp1_won = any(s["leg"] == "tp1" and s["status"] == "closed_tp" for s in sibs)
            if tp1_won and r["sl"] is not None and abs(r["sl"] - r["entry"]) > 1e-6:
                if broker.modify_sl(r["position_id"], r["entry"]):
                    trade_db.update(r["id"], sl=r["entry"], note="SL→BE sau TP1")
                    notify.send("🛡️ <b>Runner TP3: SL dời về hòa vốn</b> (TP1 đã thắng)",
                                reply_to=r["tg_msg_id"])
                    print(f"  [{now_iso()}] BE move pos {r['position_id']} -> {r['entry']}")
    return None


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
    ap.add_argument("--max-daily-loss", type=float, default=0.0,
                    help="circuit-breaker: stop NEW entries once today's NET realized P/L "
                         "<= -N USD (0=off). Open trades keep being managed; auto-resets "
                         "at the next VN day.")
    ap.add_argument("--max-signal-age-min", type=float, default=15.0,
                    help="skip signals older than N minutes (UG edge is fresh; guards "
                         "against stale reposts / feed backlog being placed late)")
    ap.add_argument("--allow-real", action="store_true",
                    help="permit --live on a REAL account (default: live allowed on DEMO only)")
    ap.add_argument("--tag", default="",
                    help="ledger namespace (e.g. 'real'): separates DB/state/lock so a real "
                         "account's P/L never mixes with demo. Empty = default (demo).")
    args = ap.parse_args()

    global STATE, ACCOUNT_LABEL, PAUSE_FLAG, HEARTBEAT
    from src.exec.broker import Mt5Broker, DryRunBroker
    broker = Mt5Broker(require_demo=not args.allow_real) if args.live else DryRunBroker()
    mode = "LIVE" if args.live else "DRY-RUN"

    # AUTO-DETECT the real account → derive ledger, real-mode and label from the ACTUAL
    # account (not from launcher flags). So real-money trades NEVER mix into the demo
    # ledger/stats, and 100/150 are auto-skipped on a real account.
    acc = broker._account_info_retry()
    if args.live and acc is None:
        raise SystemExit("ABORT: cannot read account_info")
    # Two independent signals: MT5's trade_mode flag AND the server name (Exness demo
    # servers contain 'Trial', real servers contain 'Real'). CONSERVATIVE: treat as REAL
    # if EITHER says real — never mis-classify a real account as demo.
    server = (acc.server or "") if acc else ""
    mode_demo = (acc.trade_mode == broker.mt5.ACCOUNT_TRADE_MODE_DEMO) if acc else True
    name_real = "real" in server.lower()
    is_demo = mode_demo and not name_real
    login = int(acc.login) if acc else 0
    if acc:
        print(f"  [detect] trade_mode={'demo' if mode_demo else 'real'} · server='{server}' "
              f"(name_real={name_real}) → {'DEMO' if is_demo else 'MAIN/REAL'}")
    if args.live and not is_demo and not args.allow_real:
        raise SystemExit("ABORT: --live on a non-DEMO account. Pass --allow-real to trade real money.")
    ACCOUNT_LABEL = "DEMO" if is_demo else "MAIN"
    real_mode = not is_demo                  # kept for API; unified exit means it gates nothing now

    # Auto ledger: DEMO keeps copier_trades.db; each REAL account gets its own ledger
    # (copier_trades_real_<login>.db) so its WR/P&L is tracked from scratch, separate.
    acct_tag = "" if is_demo else f"_real_{login}"
    if args.tag:                             # optional manual override
        acct_tag = f"_{args.tag}"
    trade_db.DB_PATH = Path(f"data/copier_trades{acct_tag}.db")
    mode_tag = ("live" if args.live else "dry") + acct_tag
    STATE = Path(f"data/ug/copier_state_{mode_tag}.json")
    PAUSE_FLAG = Path(f"data/ug/copier_paused_{mode_tag}.flag")   # per-account pause
    HEARTBEAT = Path(f"data/ug/copier_heartbeat_{mode_tag}.json")  # per-account liveness
    # Singleton lock (per account+mode) — a 2nd copier of the same account can't start.
    LOCK = STATE.with_name(f"copier_{mode_tag}.lock")
    _acquire_singleton(LOCK)

    # Telegram: a REAL account uses its OWN bot/group if configured, so real-money
    # alerts never mix with demo (and a future demo+real co-run needs separate tokens —
    # one getUpdates consumer per token). Falls back to the shared config if absent.
    if not is_demo:
        real_cfg = Path("configs/copier_telegram_real.json")
        if real_cfg.exists():
            notify.set_config(real_cfg)
            print(f"  [notify] REAL alerts → {real_cfg.name} (separate bot/group)")
        else:
            print("  [notify] no copier_telegram_real.json — REAL shares the demo bot/group; "
                  "create it (separate bot+group) to fully isolate real alerts")

    print(f"UG copier {mode} · {ACCOUNT_LABEL} · {args.symbol} · vol {args.volume} · "
          f"poll {args.poll}s · expiry {args.expiry_min}min · ledger {trade_db.DB_PATH.name}")
    if acc:
        print(f"  account {acc.login} · {acc.server} · {ACCOUNT_LABEL} · balance {acc.balance}")
    if args.live and not is_demo:
        print("  !! REAL-MONEY order placement ENABLED · all UG methods, unified 50/150 exit !!")
    elif args.live:
        print("  !! LIVE order placement ENABLED (demo — all methods) !!")

    st = _load_state()
    trade_db.init_db()
    now_iso = lambda: pd.Timestamp.now(tz="UTC").isoformat()
    _adopt_orphans(broker, args.symbol)        # recover any untracked magic orders/positions
    # copier's own command bot (separate token) → /stats /open /last /flat /pause
    threading.Thread(target=_command_loop, args=(broker, args.symbol), daemon=True).start()
    _filter = "50/100/150 (unified 50pip+runner150 exit)"
    notify.send(f"📥 <b>UG Copier khởi động — [{ACCOUNT_LABEL}]</b> · {mode} · {args.symbol} · "
                f"vol {args.volume} · lọc TP1∈{{{_filter}}} · ledger {trade_db.DB_PATH.name}\n"
                f"Lệnh: /stats /open /last /flat /pause /resume")

    def _on_exit():                      # graceful crash / exception / Ctrl-C alert
        try:
            notify.send(f"⚠️ <b>[{ACCOUNT_LABEL}] UG Copier ĐÃ DỪNG</b> — tiến trình thoát. "
                        f"Kiểm tra VPS/log. (hard-kill thì watchdog báo)")
        except Exception:
            pass
    atexit.register(_on_exit)

    # daily-loss breaker tripped-date — PERSISTED in state so a restart same-day stays
    # halted (a later closed win must not un-halt it).
    _tripped_day = None
    try:
        if st.get("tripped_day"):
            _tripped_day = pd.Timestamp(st["tripped_day"])
    except Exception:
        _tripped_day = None
    while True:
        try:
            try:                          # atomic write so a hard-kill mid-write can't tear it
                _hb_tmp = HEARTBEAT.with_suffix(".json.tmp")
                _hb_tmp.write_text(json.dumps({"ts": now_iso(), "label": ACCOUNT_LABEL,
                                   "pid": os.getpid(), "real": ACCOUNT_LABEL == "MAIN"}),
                                   encoding="utf-8")
                os.replace(_hb_tmp, HEARTBEAT)
            except Exception:
                pass

            # FAIL-STOP if the terminal's account login changed mid-run (someone switched
            # the account): never read/manage/place on the wrong account.
            if broker.login_changed():
                notify.send(f"🛑 <b>[{ACCOUNT_LABEL}] Account login đã ĐỔI</b> — DỪNG copier "
                            f"để an toàn (đừng để chạy nhầm tài khoản).")
                print(f"  [{now_iso()}] ACCOUNT LOGIN CHANGED — halting copier")
                raise SystemExit("account login changed mid-run")

            px = broker.get_price(args.symbol)
            if not px:
                print(f"  [{now_iso()}] no price for {args.symbol}; retry")
                time.sleep(args.poll); continue
            mid = px["mid"]

            # Circuit-breaker: once today's NET realized P/L hits the daily-loss limit,
            # halt NEW entries for the REST OF THE DAY (STICKY — a later win does NOT
            # un-halt; open trades keep being managed). Auto-resets at the VN day roll.
            loss_tripped = False
            if args.max_daily_loss > 0:
                today = pd.Timestamp.now(tz=VN).normalize()
                if _tripped_day is not None and _tripped_day == today:
                    loss_tripped = True
                else:
                    dl = trade_db.realized_pnl_since(today.tz_convert("UTC").isoformat())
                    if dl <= -args.max_daily_loss:
                        loss_tripped = True
                        _tripped_day = today
                        st["tripped_day"] = today.isoformat()    # persist: sticky across restart
                        _save_state(st)
                        notify.send(f"🛑 <b>[{ACCOUNT_LABEL}] Dừng vào lệnh hết ngày hôm nay</b> — "
                                    f"lỗ ngày {dl:+.2f} USD ≥ giới hạn {args.max_daily_loss:g}. "
                                    f"Vẫn quản lý lệnh đang mở; tự mở lại sang ngày mới.")
                        print(f"  [{now_iso()}] DAILY-LOSS tripped (sticky): {dl:+.2f}")

            # 1) New signals → decide + place. Always read the feed so loss-halted signals
            #    can be MARKED done (won't be placed late at the day-reset boundary). Pause
            #    leaves them un-done (resume reconsiders); management still runs below.
            for sig in _read_feed():
                k = _key(sig)
                if k in st["done"]:
                    continue
                if _paused():
                    continue                      # paused → leave for resume (not marked done)
                if loss_tripped:
                    st["done"][k] = {"loss_halt": True, "at": now_iso()}
                    _save_state(st)
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
                d = decide(sig, mid, volume=args.volume, real_mode=real_mode)
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
                    recs = []            # each landed leg, INSERTED into trade_db immediately
                    known_tk = {int(r["ticket"]) for r in trade_db.open_trades() if r["ticket"]}
                    for o in d.orders:
                        is_market = o.order_type.endswith("market")
                        if is_market:
                            res = broker.place_market(args.symbol, o)
                            if res is None:
                                print(f"  [{now_iso()}] place FAILED {o.order_type} {o.leg} @ {o.entry}")
                                continue
                            ticket, fill = res["position_id"], res["fill_price"]
                        else:
                            ticket = broker.place_limit(args.symbol, o)
                            if ticket is None:
                                print(f"  [{now_iso()}] place FAILED {o.order_type} {o.leg} @ {o.entry}")
                                continue
                            fill = o.entry                # a limit fills AT its price
                        if int(ticket) in known_tk:
                            # reconcile returned an ALREADY-TRACKED ticket (unclear send
                            # matched an old/sibling order) — don't create a duplicate row.
                            print(f"  [{now_iso()}] {o.leg} reconciled to already-tracked "
                                  f"ticket {ticket} — skip dup insert")
                            continue
                        known_tk.add(int(ticket))
                        print(f"  [{now_iso()}] (lag {lag:.1f}s) PLACED {o.order_type} {o.leg} "
                              f"{args.symbol} {o.volume} @ {fill} sl={o.sl} tp={o.tp} ticket={ticket}")
                        # INSERT NOW (before notify / next leg) so a crash can't leave the
                        # bracket unmanaged: the row carries group_id → BE linkage survives.
                        # entry = ACTUAL fill (market: real deal price; limit: its price) so
                        # break-even moves the runner SL to the true entry. MARKET legs are
                        # already filled positions → 'filled' + position_id; LIMIT → 'pending'.
                        rec = {"signal_ts": sig.get("ts"), "direction": o.side,
                               "method_pip": o.tp1_pip, "order_type": o.order_type,
                               "entry": fill, "sl": o.sl, "tp": o.tp, "volume": o.volume,
                               "ticket": ticket, "created_at": now_iso(),
                               "leg": o.leg, "group_id": k}
                        if is_market:
                            rec.update(status="filled", position_id=ticket,
                                       fill_price=fill, filled_at=now_iso())
                        else:
                            rec.update(status="pending")
                        tid = trade_db.insert(rec)
                        recs.append({"trade_id": tid, "ticket": ticket, "leg": o.leg, "order": o})
                    if not recs:
                        st["done"][k] = {"status": "place_failed", "at": now_iso()}
                        _save_state(st)
                        continue
                    st["done"][k] = {"placed": [{"trade_id": r["trade_id"], "ticket": r["ticket"],
                                                 "leg": r["leg"]} for r in recs]}
                    _save_state(st)
                    # one notify for the legs that landed; backfill tg_msg_id so closes can reply.
                    msg_id = notify.send(_place_msg([r["order"] for r in recs], sig, lag))
                    if msg_id:
                        for r in recs:
                            trade_db.update(r["trade_id"], tg_msg_id=msg_id)

            # Re-check the account right before management (which READS the broker and
            # WRITES the ledger): if the login flipped since the poll-top check, skip this
            # cycle so wrong-account reads can't drive ledger updates (next poll halts).
            if broker.login_changed():
                print(f"  [{now_iso()}] account changed mid-poll — skip management this cycle")
                time.sleep(args.poll); continue

            # 2+3) Manage open trades (pending→fill/cancel, filled→close+P/L, runner BE).
            if _manage_open(broker, args.symbol, mid, now_iso, args.expiry_min) == "skip":
                time.sleep(args.poll); continue
        except Exception as e:
            print(f"  [{now_iso()}] loop error {e}")
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
