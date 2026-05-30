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
import time
from pathlib import Path

import pandas as pd
import requests

from src.data import paxg_loader, signal_db
from src.strategies.xau.ob_fvg_trend import XauObFvgTrend

CFG_PATH = Path("configs/telegram.json")
TG_API = "https://api.telegram.org/bot{token}/{method}"

DEPLOY_PARAMS = {
    "swing_left": 3, "swing_right": 3, "ema_fast": 50, "ema_slow": 100,
    "atr_period": 14, "tol_atr": 0.3, "sl_buf_atr": 1.0, "tp_rr": 3.0,
    "allow_short": False,
}
WARMUP_DAYS = 60
SYMBOL = "XAUUSD"
SOURCE = "PAXG"
STRATEGY = "ob_fvg_trend"


# ---------- config ----------
def load_cfg() -> dict:
    cfg: dict = {}
    if CFG_PATH.exists():
        try:
            cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    # Environment variables override the file (Railway / production).
    env = os.environ
    token = env.get("TELEGRAM_BOT_TOKEN", cfg.get("bot_token"))
    chat = env.get("TELEGRAM_CHAT_ID", cfg.get("chat_id"))
    if not token or not chat or "PUT_YOUR" in str(token):
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID "
                         "(env vars or configs/telegram.json).")
    tfs = env.get("TIMEFRAMES")
    timeframes = ([t.strip() for t in tfs.split(",")] if tfs
                  else cfg.get("timeframes", ["H1", "M30"]))
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
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text,
                                     "parse_mode": "HTML",
                                     "disable_web_page_preview": True}, timeout=30)
    except Exception as e:
        print(f"  [telegram exception] {e}")
        return False
    if not r.ok:
        print(f"  [telegram error] {r.status_code} {r.text[:200]}")
    return r.ok


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


# ---------- data ----------
def fresh_ohlcv(tf: str) -> pd.DataFrame:
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


# ---------- signal detection ----------
def check_tf(tf: str, cfg: dict) -> dict | None:
    df = fresh_ohlcv(tf)
    if len(df) < 200:
        print(f"  {tf}: not enough bars ({len(df)})")
        return None
    df = df.iloc[:-1].reset_index(drop=True)         # drop forming bar
    sig = XauObFvgTrend().generate_signals(df, {}, params=DEPLOY_PARAMS).signals
    i = len(df) - 1
    action = sig.at[i, "action"]
    bar_ts = df.at[i, "timestamp"].isoformat()
    if action not in ("enter_long", "enter_short"):
        print(f"  {tf}: no new signal (last closed {bar_ts})")
        return None
    if signal_db.exists(tf, bar_ts):
        print(f"  {tf}: already recorded {bar_ts}")
        return None
    off = cfg["price_offset"]
    raw = float(df.at[i, "close"])
    entry, sl, tp = raw + off, float(sig.at[i, "sl"]) + off, float(sig.at[i, "tp"]) + off
    return {"tf": tf, "ts": bar_ts, "action": action, "raw": raw, "offset": off,
            "entry": entry, "sl": sl, "tp": tp,
            "risk_dist": abs(entry - sl), "reward_dist": abs(tp - entry)}


def format_msg(s: dict, risk_pct: float) -> str:
    long = s["action"] == "enter_long"
    arrow = "🟢 LONG (mua)" if long else "🔴 SHORT (bán)"
    rr = s["reward_dist"] / s["risk_dist"] if s["risk_dist"] else 0
    ts = s["ts"].replace("+00:00", " UTC")
    src = "XAU (đã +offset)" if s["offset"] else "PAXG (proxy vàng)"
    return (
        f"🟡 <b>XAU SIGNAL</b> — ob_fvg_trend\n"
        f"📊 Khung <b>{s['tf']}</b> | {arrow}\n"
        f"🕐 Nến đóng: {ts}\n"
        f"💰 Giá nguồn: {src}\n\n"
        f"➡️ <b>Entry</b> (vào khi nến KẾ mở): <b>{s['entry']:.2f}</b>\n"
        f"🛑 <b>SL</b>: {s['sl']:.2f}  ({s['risk_dist']:.2f})\n"
        f"🎯 <b>TP</b>: {s['tp']:.2f}  ({s['reward_dist']:.2f})  |  R:R {rr:.1f}\n"
        f"⚖️ Risk đề xuất: {risk_pct:.1f}% tài khoản\n\n"
        f"📌 SL/TP CỐ ĐỊNH — đặt 2 lệnh chờ rồi để chạy.\n"
        f"⚠️ Tracking — CHƯA vào tiền thật."
    )


def close_msg(row, result_r: float, status: str) -> str:
    icon = "✅ WIN" if status == "win" else "❌ LOSS"
    return (f"{icon} — {row['timeframe']} {row['direction'].upper()}\n"
            f"Mở: {row['signal_bar_ts'].replace('+00:00',' UTC')}\n"
            f"Entry {row['entry']:.2f} → "
            f"{'TP' if status=='win' else 'SL'} {row['tp'] if status=='win' else row['sl']:.2f}\n"
            f"Kết quả: <b>{result_r:+.2f}R</b>")


# ---------- outcome resolution ----------
def resolve_open(cfg: dict) -> int:
    """Check open signals against 5m price; close those that hit TP or SL."""
    rows = signal_db.open_signals()
    if not rows:
        return 0
    start = min(pd.Timestamp(r["signal_bar_ts"]) for r in rows) - pd.Timedelta(days=1)
    m5 = paxg_loader.download(interval="5m", start=str(start.date()), overwrite=True)
    off = cfg["price_offset"]
    m5 = m5.assign(high=m5["high"] + off, low=m5["low"] + off)
    closed = 0
    for r in rows:
        # Evaluate bars strictly AFTER the signal bar (entry = next-bar fill).
        seg = m5[m5["timestamp"] > pd.Timestamp(r["signal_bar_ts"])]
        if seg.empty:
            continue
        long = r["direction"] == "long"
        sl, tp = r["sl"], r["tp"]
        hit = None
        for _, b in seg.iterrows():
            lo, hi = b["low"], b["high"]
            if long:
                if lo <= sl:   hit = ("loss", sl); break    # conservative: SL first
                if hi >= tp:   hit = ("win", tp); break
            else:
                if hi >= sl:   hit = ("loss", sl); break
                if lo <= tp:   hit = ("win", tp); break
            ts_hit = b["timestamp"]
        if hit:
            status, px = hit
            result_r = r["rr"] if status == "win" else -1.0
            ts_close = b["timestamp"].isoformat()
            signal_db.close_signal(r["id"], ts_close, float(px), float(result_r), status)
            tg_send(cfg["bot_token"], cfg["chat_id"], close_msg(r, result_r, status))
            print(f"  resolved #{r['id']} {r['timeframe']} -> {status} {result_r:+.1f}R")
            closed += 1
    return closed


# ---------- main pass ----------
def run_once(cfg: dict) -> int:
    signal_db.init_db()
    sent = 0
    for tf in cfg["timeframes"]:
        try:
            s = check_tf(tf, cfg)
        except Exception as e:
            print(f"  {tf}: ERROR {e}")
            continue
        if not s:
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
            signal_db.insert_signal(rec)
            print(f"  {tf}: SENT + saved {s['ts']} ({s['action']})")
            sent += 1
    try:
        resolve_open(cfg)
    except Exception as e:
        print(f"  resolve ERROR {e}")
    return sent


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
        while True:
            print(f"[{pd.Timestamp.now(tz='UTC').isoformat()}] checking...")
            try:
                run_once(cfg)
            except Exception as e:
                print(f"  pass ERROR {e}")
            time.sleep(interval)
    else:
        print(f"done — {run_once(cfg)} signal(s) sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
