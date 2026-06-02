"""Telegram for the UG copier — its OWN dedicated bot (separate from the signal
bot), so notifications + commands are fully independent (no getUpdates conflict).

Config: configs/copier_telegram.json  { "bot_token": "...", "chat_id": "..." }
Send-only here; the copier's command loop reuses creds() for getUpdates.
Failures are swallowed (a Telegram hiccup must never break trading).
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

_CFG = Path("configs/copier_telegram.json")
_API = "https://api.telegram.org/bot{t}/{m}"
_cache: dict = {}


def creds() -> tuple[str, str] | None:
    """(bot_token, chat_id) for the copier's dedicated bot, or None if not set."""
    if "tok" in _cache:
        return _cache["tok"], _cache["chat"]
    if not _CFG.exists():
        return None
    try:
        c = json.loads(_CFG.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    token = str(c.get("bot_token") or "").strip()
    chat = str(c.get("chat_id") or "").strip()
    if not token or not chat or "PUT_" in token:
        return None
    _cache["tok"], _cache["chat"] = token, chat
    return token, chat


_creds = creds       # backward-compat alias


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
