"""Live XAU signal bot → Telegram, with SQLite history + outcome tracking.

Fetches FRESH gold price (Binance PAXGUSDT — clean, free, no API key, ~0.91
return-corr with COMEX GC=F), generates `ob_fvg_trend` signals on H1 + M30
(long-only — the walk-forward-validated deploy config), pushes a Telegram
message on each new entry, and records every signal in a SQLite DB. On each
pass it also resolves still-open signals (did price hit TP or SL first?) so a
live track record accrues to compare against the backtest.

DATA-SOURCE ONLY: Binance is the price feed; you trade on YOUR broker (Exness).
`PRICE_OFFSET` maps PAXG levels onto your broker's XAU price.

Config — environment variables (Railway) OR configs/telegram.json (local):
  TELEGRAM_BOT_TOKEN   bot token from @BotFather            (required)
  TELEGRAM_CHAT_ID     target chat/channel id               (required)
  TIMEFRAMES           comma list, default "H1,M30"
  RISK_PCT             suggested risk %, default 1.0
  PRICE_OFFSET         Exness_XAU - PAXG, default 0.0
  POLL_SECONDS         loop interval; if set, run loops forever (Railway worker)
  DB_PATH              sqlite path, default data/signals.db (set /data/... on a volume)

Local commands:
  python -X utf8 -m scripts.live_signal_bot --get-chat-id
  python -X utf8 -m scripts.live_signal_bot --test
  python -X utf8 -m scripts.live_signal_bot --status
  python -X utf8 -m scripts.live_signal_bot            # one pass
  python -X utf8 -m scripts.live_signal_bot --loop 900 # poll every 15 min
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import timedelta, timezone
from pathlib import Path

# Display timezone: Vietnam (UTC+7, no DST). Data stays in UTC; only formatting
# for Telegram messages uses this. Fixed offset → no tzdata dependency on Windows.
VN_TZ = timezone(timedelta(hours=7))

import pandas as pd
import requests

# Shared health snapshot updated by the market loop, read by the /check thread
# (so /check never calls MT5 from a second thread).
_HEALTH = {"loop_at": None, "data_at": None, "last_bar": None, "tf": None,
           "price": None, "errors": None}

from src.data import paxg_loader, signal_db
from src.strategies.xau.ob_fvg_trend import XauObFvgTrend

CFG_PATH = Path("configs/telegram.json")
TG_API = "https://api.telegram.org/bot{token}/{method}"

# Pause flag: when this file exists, the bot skips NEW signal alerts but still
# resolves already-open signals (so trades you've taken still get closed). The
# flag is on disk so a restart doesn't accidentally un-pause you.
PAUSE_FLAG = Path("logs/paused.flag")


def is_paused() -> bool:
    return PAUSE_FLAG.exists()

DEPLOY_PARAMS = {
    "swing_left": 3, "swing_right": 3, "ema_fast": 50, "ema_slow": 100,
    "atr_period": 14, "tol_atr": 0.3, "sl_buf_atr": 1.0, "tp_rr": 3.0,
    "allow_short": False,
}
WARMUP_DAYS = 60
SYMBOL = "XAUUSD"
STRATEGY = "ob_fvg_trend"
# Data source: 'mt5' (real Exness prices on the VPS) or 'paxg' (proxy, cloud/dev).
DATA_SOURCE = os.environ.get("DATA_SOURCE", "paxg").strip().lower()
MT5_SYMBOL = os.environ.get("MT5_SYMBOL", "XAUUSDm").strip()
if DATA_SOURCE not in ("mt5", "paxg"):       # fail loud, never silently fall back
    raise SystemExit(f"Invalid DATA_SOURCE={DATA_SOURCE!r} — must be 'mt5' or 'paxg'.")
SOURCE = "MT5" if DATA_SOURCE == "mt5" else "PAXG"


# ---------- config ----------
def load_cfg() -> dict:
    cfg: dict = {}
    if CFG_PATH.exists():
        try:
            cfg = json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))
        except Exception:
            cfg = {}
    # Environment variables override the file (Railway / production).
    env = os.environ
    token = str(env.get("TELEGRAM_BOT_TOKEN", cfg.get("bot_token") or "")).strip()
    chat = str(env.get("TELEGRAM_CHAT_ID", cfg.get("chat_id") or "")).strip()
    if not token or not chat or "PUT_YOUR" in token:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID "
                         "(env vars or configs/telegram.json).")
    tfs = env.get("TIMEFRAMES")
    raw_tfs = (tfs.split(",") if tfs else cfg.get("timeframes", ["H1", "M30"]))
    timeframes = [t.strip().upper() for t in raw_tfs if t and t.strip()]
    return {
        "bot_token": str(token),
        "chat_id": str(chat),
        "timeframes": timeframes,
        "risk_pct": float(env.get("RISK_PCT", cfg.get("risk_pct", 1.0))),
        "price_offset": float(env.get("PRICE_OFFSET", cfg.get("price_offset", 0.0))),
        "poll_seconds": int(env.get("POLL_SECONDS", 0)),
    }


# ---------- telegram ----------
def tg_send(token: str, chat_id: str, text: str) -> bool:
    url = TG_API.format(token=token, method="sendMessage")
    t0 = time.monotonic()
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text,
                                     "parse_mode": "HTML",
                                     "disable_web_page_preview": True}, timeout=30)
    except Exception as e:
        print(f"  [telegram exception] {e} (after {time.monotonic()-t0:.1f}s)")
        return False
    dt = time.monotonic() - t0
    if dt > 2:
        print(f"  [tg_send] SLOW {dt:.1f}s status={r.status_code}")
    if not r.ok:
        print(f"  [telegram error] {r.status_code} {r.text[:200]}")
    return r.ok


BOT_COMMANDS = [
    ("check",  "Trạng thái bot + dữ liệu"),
    ("stats",  "Track record (win/loss/R)"),
    ("last",   "5 signal gần nhất"),
    ("pause",  "Tạm dừng bắn signal mới"),
    ("resume", "Tiếp tục bắn signal"),
]


def register_commands(token: str) -> bool:
    """Push the command list to Telegram so the / menu shows suggestions."""
    url = TG_API.format(token=token, method="setMyCommands")
    cmds = [{"command": c, "description": d} for c, d in BOT_COMMANDS]
    try:
        r = requests.post(url, json={"commands": cmds}, timeout=15)
        if not r.ok:
            print(f"  [setMyCommands] {r.status_code} {r.text[:200]}")
        return r.ok
    except Exception as e:
        print(f"  [setMyCommands exception] {e}")
        return False


def get_chat_id(token: str) -> None:
    url = TG_API.format(token=token, method="getUpdates")
    r = requests.get(url, timeout=30).json()
    seen = set()
    for upd in r.get("result", []):
        chat = (upd.get("message") or upd.get("channel_post") or {}).get("chat", {})
        if chat.get("id") and chat["id"] not in seen:
            seen.add(chat["id"])
            print(f"  chat_id = {chat['id']}  ({chat.get('type')}, "
                  f"{chat.get('first_name') or chat.get('title','')})")
    if not seen:
        print("  No chats. Message your bot (or add it to the channel) first.")


# ---------- on-demand /check command ----------
def _tg_init_offset(token: str) -> int:
    """Return last update_id+1 so we ignore messages sent before the bot started."""
    try:
        r = requests.get(TG_API.format(token=token, method="getUpdates"),
                         params={"timeout": 0}, timeout=20).json()
        ups = r.get("result", [])
        return ups[-1]["update_id"] + 1 if ups else 0
    except Exception:
        return 0


def _age(ts) -> str:
    if ts is None:
        return "chưa có"
    return f"{(pd.Timestamp.now(tz='UTC') - ts).total_seconds():.0f}s trước"


def _fmt_ts(ts) -> str:
    """Any ISO string / pandas Timestamp → 'YYYY-MM-DD HH:MM GMT+7' (giờ VN).

    All data is stored/processed in UTC; this converts for DISPLAY only. A fixed
    +7h offset is used (Vietnam has no DST) so we don't depend on the tzdata
    package being installed on the Windows VPS.
    """
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.tz_convert(VN_TZ).strftime("%Y-%m-%d %H:%M GMT+7")


def _check_reply() -> str:
    """Health reply from the cached snapshot (no MT5 call from this thread)."""
    h = _HEALTH
    if h["data_at"] is None:
        return "🤖 <b>XAU bot vừa khởi động</b>\nChưa có lần đọc dữ liệu — thử lại sau ~1 phút."
    ok = (h["errors"] == 0)
    d = signal_db.summary()
    last = _fmt_ts(h["last_bar"])
    if is_paused():
        header = "⏸ <b>XAU bot — TẠM DỪNG</b>\nkhông bắn signal mới · nguồn " + SOURCE
    else:
        header = "🤖 <b>XAU bot — ĐANG CHẠY</b>\nnguồn " + SOURCE
    data_line = (f"✅ Dữ liệu OK · đọc {_age(h['data_at'])}" if ok
                 else f"⚠️ Dữ liệu LỖI · thử {_age(h['data_at'])}")
    return (f"{header}\n\n"
            f"{data_line}\n"
            f"🔄 Vòng quét: {_age(h['loop_at'])}\n"
            f"🕐 Nến {h['tf']}: {last}\n"
            f"💰 Giá: {h['price']:.2f}\n\n"
            f"📊 {d['total']} signal · đang mở {d['open_n']}\n"
            f"🏆 win {d['wins']} / loss {d['losses']} · kỳ vọng {d['exp_r']:+.2f}R")


def _stats_reply() -> str:
    d = signal_db.summary()
    closed = (d["wins"] or 0) + (d["losses"] or 0)
    return (f"📊 <b>Track record</b>\n"
            f"{STRATEGY}\n\n"
            f"Tổng signal:  <b>{d['total']}</b>  ·  đang mở {d['open_n']}\n"
            f"Đã đóng:  {closed}  (✅ {d['wins']} / ❌ {d['losses']})\n\n"
            f"🎯 Win-rate:  <b>{d['winrate']:.0%}</b>\n"
            f"💰 Tổng R:  <b>{d['sum_r']:+.1f}</b>\n"
            f"📈 Kỳ vọng:  <b>{d['exp_r']:+.2f}R</b> / lệnh")


def _last_reply(n: int = 5) -> str:
    rows = signal_db.recent(n)
    if not rows:
        return "📭 Chưa có signal nào."
    icons = {"win": "✅", "loss": "❌", "open": "🟡"}
    lines = [f"📜 <b>{len(rows)} signal gần nhất</b>\n"]
    for r in rows:
        icon = icons.get(r["status"], "•")
        when = _fmt_ts(r["signal_bar_ts"])
        side = r["direction"].upper()
        res = "đang mở" if r["result_r"] is None else f"{r['result_r']:+.1f}R"
        lines.append(f"{icon} <b>#{r['id']}</b> {r['timeframe']} {side} @{r['entry']:.2f}\n"
                     f"     {when} · {res}")
    return "\n".join(lines)


def _command_loop(cfg: dict) -> None:
    """Background long-polling loop → replies to commands within ~1s."""
    token, chat = cfg["bot_token"], str(cfg["chat_id"])
    offset = _tg_init_offset(token)
    while True:
        try:
            _t0 = time.monotonic()
            r = requests.get(TG_API.format(token=token, method="getUpdates"),
                             params={"offset": offset, "timeout": 25},
                             timeout=35).json()
            _dt = time.monotonic() - _t0
            # Long-poll returns instantly on a new message, else ~25s on timeout.
            # A return between 2s and 24s with NO updates == a stall/hiccup worth
            # seeing (it's a window where commands could have queued).
            if 2 < _dt < 24 and not r.get("result"):
                print(f"  [cmd] getUpdates returned empty after {_dt:.1f}s (stall?)")
            # 409 in steady state == a SECOND consumer is polling this same bot
            # token (e.g. a leftover Railway worker). Telegram splits updates
            # between consumers → commands feel laggy/dropped. Log it loudly.
            if not r.get("ok", True) and "Conflict" in str(r.get("description", "")):
                print(f"  [cmd] ⚠️ getUpdates CONFLICT — another consumer is polling "
                      f"this bot token: {r.get('description')}")
            for u in r.get("result", []):
                offset = u["update_id"] + 1
                msg = u.get("message") or u.get("channel_post") or {}
                text = (msg.get("text") or "").strip().lower()
                cid = str((msg.get("chat") or {}).get("id", ""))
                if cid != chat:
                    continue
                if text:
                    recv = pd.Timestamp.now(tz="UTC")
                    edits = msg.get("edit_date")
                    sent_epoch = edits or msg.get("date")
                    lag = (recv.timestamp() - sent_epoch) if sent_epoch else -1
                    print(f"  [cmd] recv {text.split()[0]} · lag {lag:.1f}s "
                          f"(sent→received)")
                if text.startswith("/check") or text.startswith("/status"):
                    tg_send(token, chat, _check_reply())
                elif text.startswith("/stats"):
                    tg_send(token, chat, _stats_reply())
                elif text.startswith("/last"):
                    tg_send(token, chat, _last_reply())
                elif text.startswith("/pause") or text.startswith("/stop"):
                    if is_paused():
                        tg_send(token, chat, "⏸ Bot đã ở trạng thái tạm dừng từ trước.")
                    else:
                        PAUSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
                        PAUSE_FLAG.write_text(pd.Timestamp.now(tz="UTC").isoformat(),
                                              encoding="utf-8")
                        tg_send(token, chat, "⏸ <b>Đã TẠM DỪNG</b> bắn signal mới.\n"
                                "Lệnh đang mở vẫn được resolve khi chạm TP/SL.\n"
                                "Gõ /resume để chạy lại.")
                elif text.startswith("/resume") or text.startswith("/start"):
                    if not is_paused():
                        tg_send(token, chat, "▶️ Bot đang chạy bình thường — không cần resume.")
                    else:
                        try:
                            PAUSE_FLAG.unlink()
                        except FileNotFoundError:
                            pass
                        tg_send(token, chat, "▶️ <b>Đã TIẾP TỤC</b> bắn signal.\n"
                                "Bot sẽ quét lại ở vòng kế.")
        except Exception as e:
            print(f"  cmd loop error {e}")
            time.sleep(5)


# ---------- data ----------
def fresh_ohlcv(tf: str) -> pd.DataFrame:
    if DATA_SOURCE == "mt5":
        from src.data import mt5_feed                  # lazy, Windows-only
        return mt5_feed.bars(MT5_SYMBOL, tf, n=3000)
    start = str((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=WARMUP_DAYS)).date())
    if tf == "H1":
        return paxg_loader.download(interval="1h", start=start, overwrite=True)
    if tf == "M30":
        m15 = paxg_loader.download(interval="15m", start=start, overwrite=True)
        return (m15.set_index("timestamp")
                .resample("30min", label="left", closed="left")
                .agg({"open": "first", "high": "max", "low": "min",
                      "close": "last", "volume": "sum"})
                .dropna().reset_index())
    raise ValueError(f"unsupported tf {tf}")


def fresh_m5() -> pd.DataFrame:
    """5-minute bars for outcome resolution, from the active source."""
    if DATA_SOURCE == "mt5":
        from src.data import mt5_feed
        return mt5_feed.bars(MT5_SYMBOL, "M5", n=17280)   # ~60 trading days
    start = str((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=WARMUP_DAYS)).date())
    return paxg_loader.download(interval="5m", start=start, overwrite=True)


# ---------- signal detection ----------
def check_tf(tf: str, cfg: dict) -> dict | None:
    df = fresh_ohlcv(tf)
    if len(df) < 200:
        print(f"  {tf}: not enough bars ({len(df)})")
        return None
    df = df.iloc[:-1].reset_index(drop=True)         # drop forming bar
    # health snapshot (read by /check without touching MT5 from another thread)
    _HEALTH.update({"data_at": pd.Timestamp.now(tz="UTC"),
                    "last_bar": df["timestamp"].iloc[-1], "tf": tf,
                    "price": float(df["close"].iloc[-1])})
    sig = XauObFvgTrend().generate_signals(df, {}, params=DEPLOY_PARAMS).signals
    i = len(df) - 1
    action = sig.at[i, "action"]
    bar_ts = df.at[i, "timestamp"].isoformat()
    if action not in ("enter_long", "enter_short"):
        print(f"  {tf}: no new signal (last closed {bar_ts})")
        return None
    if signal_db.exists(tf, bar_ts, SOURCE, SYMBOL, STRATEGY):
        print(f"  {tf}: already recorded {bar_ts}")
        return None
    off = 0.0 if DATA_SOURCE == "mt5" else cfg["price_offset"]
    raw = float(df.at[i, "close"])
    entry, sl, tp = raw + off, float(sig.at[i, "sl"]) + off, float(sig.at[i, "tp"]) + off
    return {"tf": tf, "ts": bar_ts, "action": action, "raw": raw, "offset": off,
            "entry": entry, "sl": sl, "tp": tp,
            "risk_dist": abs(entry - sl), "reward_dist": abs(tp - entry)}


def format_msg(s: dict, risk_pct: float) -> str:
    long = s["action"] == "enter_long"
    side = "🟢 <b>LONG</b> (mua)" if long else "🔴 <b>SHORT</b> (bán)"
    rr = s["reward_dist"] / s["risk_dist"] if s["risk_dist"] else 0
    ts = _fmt_ts(s["ts"])
    if SOURCE == "MT5":
        src = "Exness MT5 (giá thật)"
    else:
        src = "XAU (PAXG +offset)" if s["offset"] else "PAXG (proxy vàng)"
    # Monospace block → Entry/TP/SL columns line up; TP shown as +reward, SL as -risk.
    table = (
        f"Entry  {s['entry']:>9.2f}\n"
        f"TP     {s['tp']:>9.2f}  {s['reward_dist']:>+8.2f}\n"
        f"SL     {s['sl']:>9.2f}  {-s['risk_dist']:>+8.2f}\n"
        f"R:R    1 : {rr:.1f}"
    )
    return (
        f"🟡 <b>XAU — TÍN HIỆU MỚI</b>\n\n"
        f"{side}   ·   khung <b>{s['tf']}</b>\n"
        f"🕐 {ts}\n"
        f"💰 {src}\n\n"
        f"<pre>{table}</pre>\n"
        f"⚖️ Risk đề xuất: <b>{risk_pct:.1f}%</b> tài khoản\n\n"
        f"📌 Đặt 2 lệnh chờ — vào khi nến KẾ mở, SL/TP cố định.\n"
        f"⚠️ Đang tracking — CHƯA vào tiền thật."
    )


def close_msg(row, result_r: float, status: str) -> str:
    win = status == "win"
    icon = "✅ <b>WIN</b>" if win else "❌ <b>LOSS</b>"
    exit_label = "TP" if win else "SL"
    exit_px = row["tp"] if win else row["sl"]
    opened = _fmt_ts(row["signal_bar_ts"])
    return (f"{icon}   {row['timeframe']} {row['direction'].upper()}   "
            f"<b>{result_r:+.2f}R</b>\n\n"
            f"Entry {row['entry']:.2f}  →  {exit_label} {exit_px:.2f}\n"
            f"🕐 Mở: {opened}")


# Bar duration per timeframe (minutes) — used to find the entry candle's open.
TF_MINUTES = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}


# ---------- outcome resolution ----------
def resolve_open(cfg: dict) -> int:
    """Resolve open signals against 5m price; close those that hit TP or SL.

    Parity: a signal printed at bar i's CLOSE is entered at bar i+1's OPEN. So
    outcomes are evaluated only from the entry bar onward (signal_bar_ts + one
    timeframe), never from inside the signal candle itself.
    """
    rows = signal_db.open_signals()
    if not rows:
        return 0
    m5 = fresh_m5()
    m5["timestamp"] = pd.to_datetime(m5["timestamp"], utc=True)   # ensure tz-aware
    off = 0.0 if DATA_SOURCE == "mt5" else cfg["price_offset"]
    m5 = m5.assign(high=m5["high"] + off, low=m5["low"] + off)
    closed = 0
    for r in rows:
        if r["timeframe"] not in TF_MINUTES:
            print(f"  resolve: unknown timeframe {r['timeframe']!r} on #{r['id']} — skip")
            continue
        tf_min = TF_MINUTES[r["timeframe"]]
        ets = pd.Timestamp(r["signal_bar_ts"])
        ets = ets.tz_localize("UTC") if ets.tzinfo is None else ets.tz_convert("UTC")
        entry_ts = ets + pd.Timedelta(minutes=tf_min)
        seg = m5[m5["timestamp"] >= entry_ts]      # from entry bar open onward
        if seg.empty:
            if len(m5) and entry_ts < m5["timestamp"].min():
                print(f"  resolve: #{r['id']} entry {entry_ts} predates M5 window "
                      f"{m5['timestamp'].min()} — cannot resolve, widen fresh_m5 n")
            continue
        long = r["direction"] == "long"
        sl, tp = r["sl"], r["tp"]
        hit = None
        for _, b in seg.iterrows():
            lo, hi = b["low"], b["high"]
            if long:
                if lo <= sl:   hit = ("loss", sl, b["timestamp"]); break  # conservative: SL first
                if hi >= tp:   hit = ("win", tp, b["timestamp"]); break
            else:
                if hi >= sl:   hit = ("loss", sl, b["timestamp"]); break
                if lo <= tp:   hit = ("win", tp, b["timestamp"]); break
        if hit:
            status, px, ts = hit
            result_r = r["rr"] if status == "win" else -1.0
            # Send the close alert FIRST; only mark resolved if it was delivered,
            # so a Telegram failure is retried next pass (no lost notification).
            if not tg_send(cfg["bot_token"], cfg["chat_id"], close_msg(r, result_r, status)):
                print(f"  resolve: tg_send failed for #{r['id']} — leaving open to retry")
                continue
            signal_db.close_signal(r["id"], ts.isoformat(), float(px), float(result_r), status)
            print(f"  resolved #{r['id']} {r['timeframe']} -> {status} {result_r:+.1f}R")
            closed += 1
    return closed


# ---------- main pass ----------
def run_once(cfg: dict) -> int:
    signal_db.init_db()
    sent = 0
    errors = 0
    last_err = ""
    for tf in cfg["timeframes"]:
        try:
            s = check_tf(tf, cfg)
        except Exception as e:
            print(f"  {tf}: ERROR {e}")
            errors += 1
            last_err = f"{tf}: {e}"
            continue
        if not s:
            continue
        # Re-check pause right before the network call so a /pause arriving
        # mid-pass blocks the alert (snapshotting earlier would race).
        if is_paused():
            print(f"  {tf}: [paused] signal detected at {s['ts']} but NOT sent/saved")
            continue
        if tg_send(cfg["bot_token"], cfg["chat_id"], format_msg(s, cfg["risk_pct"])):
            rec = {
                "sent_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "symbol": SYMBOL, "source": SOURCE, "strategy": STRATEGY,
                "timeframe": tf, "direction": "long" if s["action"] == "enter_long" else "short",
                "signal_bar_ts": s["ts"], "entry": s["entry"], "sl": s["sl"], "tp": s["tp"],
                "risk_dist": s["risk_dist"], "reward_dist": s["reward_dist"],
                "rr": s["reward_dist"] / s["risk_dist"] if s["risk_dist"] else 0,
                "source_price": s["raw"], "price_offset": s["offset"], "status": "open",
            }
            rid = signal_db.insert_signal(rec)
            if rid is None:
                print(f"  {tf}: !! ALERT SENT BUT NOT SAVED (dup/constraint) {s['ts']} "
                      f"— outcome won't be tracked")
            else:
                print(f"  {tf}: SENT + saved #{rid} {s['ts']} ({s['action']})")
            sent += 1
    try:
        resolve_open(cfg)
    except Exception as e:
        print(f"  resolve ERROR {e}")
        errors += 1
        last_err = f"resolve: {e}"
    return sent, errors, last_err


def print_status() -> None:
    signal_db.init_db()
    d = signal_db.summary()
    print(f"DB: {d['total']} signals | open {d['open_n']} | "
          f"win {d['wins']} loss {d['losses']} | WR {d['winrate']:.0%} | "
          f"sumR {d['sum_r']:+.1f} | expR {d['exp_r']:+.2f}")
    for r in signal_db.recent(10):
        rsuffix = "" if r["result_r"] is None else f" {r['result_r']:+.1f}R"
        print(f"  #{r['id']} {r['timeframe']:3} {r['direction']:5} "
              f"{r['signal_bar_ts'][:16]} entry{r['entry']:.1f} "
              f"{r['status']}{rsuffix}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--get-chat-id", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--loop", type=int, default=0)
    args = ap.parse_args()

    if args.status:
        print_status(); return 0
    cfg = load_cfg()
    if args.get_chat_id:
        get_chat_id(cfg["bot_token"]); return 0
    if args.test:
        ok = tg_send(cfg["bot_token"], cfg["chat_id"],
                     "✅ <b>Bot kết nối thành công</b>\nSẽ bắn signal XAU (H1+M30) tại đây.")
        print("test sent:", ok); return 0 if ok else 1

    interval = args.loop or cfg["poll_seconds"]
    if interval > 0:
        print(f"polling every {interval}s — Ctrl+C to stop")
        register_commands(cfg["bot_token"])
        # Separate thread handles /check via long-polling → replies in ~1s,
        # independent of the 60s market loop.
        threading.Thread(target=_command_loop, args=(cfg,), daemon=True).start()
        err_streak = 0
        in_error = False
        ALERT_AFTER = 3                       # alert after N consecutive failed passes
        while True:
            print(f"[{pd.Timestamp.now(tz='UTC').isoformat()}] checking...")
            try:
                _, errors, last_err = run_once(cfg)
            except Exception as e:
                print(f"  pass ERROR {e}")
                errors, last_err = 99, f"pass: {e}"
            _HEALTH["loop_at"] = pd.Timestamp.now(tz="UTC")
            _HEALTH["errors"] = errors
            # Alert ONLY on error (and once on recovery).
            if errors > 0:
                err_streak += 1
                if err_streak >= ALERT_AFTER and not in_error:
                    tg_send(cfg["bot_token"], cfg["chat_id"],
                            f"⚠️ <b>XAU bot LỖI</b>\n{err_streak} vòng liên tiếp lỗi.\n"
                            f"<code>{last_err[:300]}</code>\nBot vẫn tự retry — kiểm tra VPS/MT5.")
                    in_error = True
            else:
                if in_error:
                    tg_send(cfg["bot_token"], cfg["chat_id"],
                            "✅ <b>XAU bot phục hồi</b>\nĐã đọc lại dữ liệu bình thường.")
                err_streak = 0
                in_error = False
            time.sleep(interval)
    else:
        s, e, _ = run_once(cfg)
        print(f"done — {s} signal(s), {e} error(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
