"""Telegram user-client (Telethon) to capture signals from the UG bot.

This logs in as YOUR Telegram account (MTProto) — the only way to read another
bot's messages; a Bot-API bot cannot. The session file grants full account
access, so it is gitignored and must be treated as a secret.

Pipeline role: this is the CAPTURE stage. It dumps UG's messages (history +
live) to data/ug/messages.jsonl as the raw source of truth. The PARSER and the
VALIDATORS (which reverse-engineer UG's logic against TradingView data) are
built afterwards, informed by the real message format we see here.

Config — configs/ug.json (gitignored):
  { "api_id": 123456, "api_hash": "...", "phone": "+84...",
    "session": "configs/ug.session", "ug_chat": "@SomeUgBot" }
api_id / api_hash come from https://my.telegram.org → API development tools.

Usage:
  # one-time interactive login (YOU run this; enter the code Telegram sends):
  python -X utf8 -m scripts.ug_reader --auth
  # find the UG chat (prints id / title / @username of every dialog):
  python -X utf8 -m scripts.ug_reader --list
  python -X utf8 -m scripts.ug_reader --list --query ug
  # dump history of the configured ug_chat to data/ug/messages.jsonl:
  python -X utf8 -m scripts.ug_reader --history --limit 500
  # live-capture new messages (append):
  python -X utf8 -m scripts.ug_reader --listen
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta, timezone
from pathlib import Path

CFG_PATH = Path("configs/ug.json")
OUT_PATH = Path("data/ug/messages.jsonl")
VN_TZ = timezone(timedelta(hours=7))


def _load_cfg() -> dict:
    if not CFG_PATH.exists():
        raise SystemExit(
            f"Missing {CFG_PATH}. Create it:\n"
            '  {"api_id": 123456, "api_hash": "...", "phone": "+84...",\n'
            '   "session": "configs/ug.session", "ug_chat": "@TheUgBot"}\n'
            "api_id/api_hash from https://my.telegram.org")
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))
    for k in ("api_id", "api_hash"):
        if not cfg.get(k):
            raise SystemExit(f"{CFG_PATH}: missing {k}")
    cfg.setdefault("session", "configs/ug.session")
    return cfg


def _client(cfg: dict):
    try:
        from telethon.sync import TelegramClient
    except ImportError:
        raise SystemExit("Telethon not installed. Run:\n"
                         "  .venv\\Scripts\\python.exe -m pip install telethon")
    return TelegramClient(cfg["session"], int(cfg["api_id"]), str(cfg["api_hash"]))


def _fmt_vn(dt) -> str:
    return dt.astimezone(VN_TZ).strftime("%Y-%m-%d %H:%M:%S GMT+7")


def cmd_auth(cfg: dict) -> int:
    with _client(cfg) as client:
        # phone from config; code is always interactive; 2FA password from config
        # if present (avoids mistypes in the hidden prompt), else prompts.
        client.start(phone=cfg.get("phone") or None,
                     password=cfg.get("password") or None)
        me = client.get_me()
        uname = f"@{me.username}" if me.username else ""
        print(f"✅ Logged in as {me.first_name} {uname} (id {me.id}). "
              f"Session saved to {cfg['session']}")
    return 0


def cmd_list(cfg: dict, query: str | None) -> int:
    q = (query or "").lower()
    with _client(cfg) as client:
        client.start(phone=cfg.get("phone") or None)
        print(f"{'id':>14}  {'type':<8} {'@username':<22} title")
        for d in client.iter_dialogs():
            ent = d.entity
            uname = getattr(ent, "username", None) or ""
            kind = "bot" if getattr(ent, "bot", False) else (
                "channel" if d.is_channel else "group" if d.is_group else "user")
            name = d.name or ""
            if q and q not in name.lower() and q not in uname.lower():
                continue
            print(f"{d.id:>14}  {kind:<8} {('@'+uname) if uname else '':<22} {name}")
    return 0


def _resolve(client, ug_chat):
    """Resolve ug_chat (id / @username / exact title) to a Telethon entity.

    For a bare numeric id, get_entity(id) needs the access_hash cached — a freshly
    authed session doesn't have it. So we iterate dialogs first (which POPULATES
    the entity cache) and match by id/title there. You have a DM with the UG bot,
    so it's in your dialogs.
    """
    if ug_chat is None:
        raise SystemExit("Set 'ug_chat' in configs/ug.json (run --list to find it).")
    s = str(ug_chat)
    if s.startswith("@"):
        return client.get_entity(s)
    target_id = int(s) if s.lstrip("-").isdigit() else None
    for d in client.iter_dialogs():          # caches entities + matches
        if target_id is not None and getattr(d.entity, "id", None) == target_id:
            return d.entity
        if target_id is None and (d.name or "") == s:
            return d.entity
    # fallback (works now that dialogs are cached)
    return client.get_entity(target_id if target_id is not None else s)


def _record(m) -> dict:
    return {
        "id": m.id,
        "date_utc": m.date.astimezone(timezone.utc).isoformat(),
        "date_vn": _fmt_vn(m.date),
        "text": m.message or "",
        "has_media": m.media is not None,
        "sender_id": m.sender_id,
    }


def cmd_history(cfg: dict, limit: int) -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _client(cfg) as client:
        client.start(phone=cfg.get("phone") or None)
        ent = _resolve(client, cfg.get("ug_chat"))
        rows = [_record(m) for m in client.iter_messages(ent, limit=limit)]
    rows.reverse()                            # oldest → newest
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} messages → {OUT_PATH}")
    for r in rows[-8:]:                        # preview the latest few
        preview = r["text"].replace("\n", " ⏎ ")[:120]
        print(f"  [{r['date_vn']}] {preview}")
    return 0


FEED_PATH = Path("data/ug/live_signals.jsonl")   # structured signals for the copier


def cmd_listen(cfg: dict) -> int:
    from telethon import events
    from scripts.parse_ug_export import parse_signal     # reuse the proven parser
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = _client(cfg)
    client.start(phone=cfg.get("phone") or None)
    ent = _resolve(client, cfg.get("ug_chat"))
    chat_id = getattr(ent, "id", None)
    print(f"Listening to {cfg.get('ug_chat')} (id {chat_id}).")
    print(f"  raw → {OUT_PATH}   ·   parsed signals → {FEED_PATH}")
    print("  Ctrl+C to stop.")

    @client.on(events.NewMessage(chats=ent))
    async def _on(event):                      # noqa: ANN001
        r = _record(event.message)
        with OUT_PATH.open("a", encoding="utf-8") as f:      # raw audit trail
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  [{r['date_vn']}] {r['text'].replace(chr(10),' ⏎ ')[:100]}")
        try:                                                  # parse → copier feed
            sig = parse_signal(r["date_utc"], r["text"])
        except Exception as e:
            print(f"    [parse error] {e}"); return
        if sig:
            with FEED_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(sig, ensure_ascii=False) + "\n")
            tp1 = (sig.get("tps_pip") or {}).get(1)
            print(f"    → SIGNAL {sig['direction']} entry {sig['entry_low']}-{sig['entry_high']} "
                  f"sl {sig['sl']} TP1 {tp1}pip → fed to copier")

    client.run_until_disconnected()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth", action="store_true", help="interactive login (run once)")
    ap.add_argument("--list", action="store_true", help="list dialogs to find UG")
    ap.add_argument("--query", default=None, help="filter --list by name/username substring")
    ap.add_argument("--history", action="store_true", help="dump ug_chat history to jsonl")
    ap.add_argument("--limit", type=int, default=500, help="messages for --history")
    ap.add_argument("--listen", action="store_true", help="live-capture new messages")
    args = ap.parse_args()

    cfg = _load_cfg()
    if args.auth:
        return cmd_auth(cfg)
    if args.list:
        return cmd_list(cfg, args.query)
    if args.history:
        return cmd_history(cfg, args.limit)
    if args.listen:
        return cmd_listen(cfg)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
