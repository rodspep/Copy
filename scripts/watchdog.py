"""External watchdog for the UG copier. Runs independently (Windows Task Scheduler,
session 0) so it survives even a hard-kill of the copier. Scans the per-account
heartbeat files the copier refreshes each poll; if a heartbeat goes STALE (copier hung
or killed), it sends ONE Telegram alert to that account's group (real → real config,
demo → demo config), and ONE 'recovered' alert when the heartbeat is fresh again.

Schedule (every ~3 min):  python -m scripts.watchdog
Stateless per run; uses a .alerted marker file to avoid spamming while down.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.exec import notify

HB_DIR = Path("data/ug")
STALE_SEC = 300                      # >5 min with no heartbeat = hung/dead (poll is ~2s)
DEMO_CFG = Path("configs/copier_telegram.json")
REAL_CFG = Path("configs/copier_telegram_real.json")


def _age_sec(ts_iso: str) -> float:
    t = datetime.fromisoformat(ts_iso)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - t).total_seconds()


def _alert(is_real: bool, text: str) -> bool:
    """Returns True only if Telegram actually delivered (notify.send returns the
    message_id, or None on any failure). The caller latches the marker ONLY on a
    confirmed send, so a failed/rate-limited send is retried next run (not suppressed)."""
    cfg = REAL_CFG if (is_real and REAL_CFG.exists()) else DEMO_CFG
    notify.set_config(cfg)
    return notify.send(text) is not None


def main() -> int:
    hbs = sorted(HB_DIR.glob("copier_heartbeat_*.json"))
    for hb in hbs:
        marker = hb.with_suffix(".alerted")
        try:
            d = json.loads(hb.read_text(encoding="utf-8"))
            age = _age_sec(d.get("ts", ""))
            label = d.get("label", "?")
            is_real = bool(d.get("real"))
        except Exception as e:
            print(f"[watchdog] cannot read {hb.name}: {e}")
            continue
        if age > STALE_SEC:
            if not marker.exists():
                ok = _alert(is_real, f"🚨 <b>[{label}] UG Copier KHÔNG phản hồi {age/60:.0f} phút</b> "
                                     f"(heartbeat cũ) — có thể đã treo/chết. Kiểm tra VPS ngay.")
                if ok:                       # latch only on confirmed delivery → retry if it failed
                    marker.write_text("down", encoding="utf-8")
                print(f"[watchdog] down {hb.name} age {age:.0f}s delivered={ok}")
        else:
            if marker.exists():
                ok = _alert(is_real, f"✅ <b>[{label}] UG Copier đã chạy lại</b> (heartbeat tươi).")
                if ok:
                    marker.unlink()
                print(f"[watchdog] recovered {hb.name} delivered={ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
