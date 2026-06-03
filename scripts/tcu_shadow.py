"""(b) Shadow-log of the TCU public signal API — an IMMUTABLE, receipt-time dataset so the
edge comparison is free of the survivorship/cherry-pick bias that made the 1-week Telegram
set look 13x better than the full API history (see scripts/tcu_replay_compare).

Each run (schedule every ~15 min, e.g. Windows Task Scheduler):
  - GET /api/signals?symbol=XAUUSD&limit=500
  - append any NEVER-SEEN signal id to shadow_receipts.jsonl with received_at = now
    (APPEND-ONLY — the receipt + the signal AS FIRST SEEN is never rewritten), and
  - refresh shadow_outcomes.json = latest {status, maxFavorablePrice, realizedPips, hit times}
    per id (mutable — captures resolution as trades close).
No MT5, no copier, no session — pure HTTP read; cannot affect live trading. Network/parse
errors are swallowed so a scheduled run never crash-loops.

Run: python -X utf8 -m scripts.tcu_shadow
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://tradecoinunderground.com/api/signals?symbol=XAUUSD&limit=500"
DIR = Path("data/ug/tcu")
RECEIPTS = DIR / "shadow_receipts.jsonl"
OUTCOMES = DIR / "shadow_outcomes.json"

# fields frozen at receipt time (the signal as first published — never rewritten)
RECEIPT_FIELDS = ("id", "symbol", "direction", "signalType", "entry", "entryRange",
                  "stopLoss", "takeProfit1", "takeProfit2", "takeProfit3", "tpPips",
                  "displayName", "isPrivate", "rawText", "sourceMessageId", "createdAt")
# fields tracked over time (resolution)
OUTCOME_FIELDS = ("status", "currentPrice", "maxFavorablePrice", "realizedPips",
                  "entryHitAt", "tp1HitAt", "tp2HitAt", "tp3HitAt", "slHitAt",
                  "voidedAt", "cancelledAt", "updatedAt")


def _now():
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(API, timeout=20)
        sigs = (r.json() or {}).get("signals", []) if r.ok else []
    except Exception as e:
        print(f"[shadow] fetch failed: {e}")
        return 0
    if not sigs:
        print("[shadow] no signals returned")
        return 0

    seen = set()
    if RECEIPTS.exists():
        for line in RECEIPTS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:                          # str(id) so re-runs never re-append (torn lines skipped)
                    seen.add(str(json.loads(line)["id"]))
                except Exception:
                    pass

    now = _now()
    new = 0
    with RECEIPTS.open("a", encoding="utf-8") as f:
        for s in sigs:
            sid = str(s.get("id") or "")
            if not sid or sid in seen:
                continue
            rec = {"received_at": now, **{k: s.get(k) for k in RECEIPT_FIELDS}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            seen.add(sid)
            new += 1

    # refresh outcomes snapshot for ALL ids we've ever seen (resolution may update post-receipt)
    outcomes = {}
    if OUTCOMES.exists():
        try:
            outcomes = json.loads(OUTCOMES.read_text(encoding="utf-8"))
        except Exception:
            outcomes = {}
    for s in sigs:
        sid = str(s.get("id") or "")
        if sid:
            outcomes[sid] = {"snapshot_at": now, **{k: s.get(k) for k in OUTCOME_FIELDS}}
    tmp = OUTCOMES.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(outcomes, ensure_ascii=False), encoding="utf-8")
    tmp.replace(OUTCOMES)

    print(f"[shadow] {now} polled {len(sigs)} | new receipts {new} | tracked {len(outcomes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
