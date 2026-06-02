"""Telegram notifications for the copier (send + reply). Send-only — no getUpdates
(so it never conflicts with the signal bot's command long-poll on the same token).

Uses configs/telegram.json: bot_token + (copier_chat_id or chat_id). Failures are
swallowed (a Telegram hiccup must never break trading).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

_CFG = Path("configs/telegram.json")
_API = "https://api.telegram.org/bot{t}/{m}"
_cache: dict = {}


def _creds() -> tuple[str, str] | None:
    if "tok" in _cache:
        return _cache["tok"], _cache["chat"]
    token = chat = ""
    if _CFG.exists():
        try:
            c = json.loads(_CFG.read_text(encoding="utf-8-sig"))
            token = str(c.get("bot_token") or "")
            chat = str(c.get("copier_chat_id") or c.get("chat_id") or "")
        except Exception:
            pass
    token = (os.environ.get("TELEGRAM_BOT_TOKEN", token) or "").strip()
    chat = (os.environ.get("TELEGRAM_CHAT_ID", chat) or "").strip()
    if not token or not chat:
        return None
    _cache["tok"], _cache["chat"] = token, chat
    return token, chat


def send(text: str, reply_to: int | None = None) -> int | None:
    """Send a message (optionally as a reply). Returns the new message_id or None."""
    creds = _creds()
    if creds is None:
        print("  [notify] no telegram creds — skipped")
        return None
    token, chat = creds
    payload = {"chat_id": chat, "text": text, "parse_mode": "HTML",
               "disable_web_page_preview": True}
    if reply_to:
        payload["reply_to_message_id"] = int(reply_to)
        payload["allow_sending_without_reply"] = True
    try:
        r = requests.post(_API.format(t=token, m="sendMessage"), json=payload, timeout=15)
        if not r.ok:
            print(f"  [notify] {r.status_code} {r.text[:150]}")
            return None
        return r.json().get("result", {}).get("message_id")
    except Exception as e:
        print(f"  [notify] exception {e}")
        return None
