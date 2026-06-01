"""Structured UG signal model + loader.

The Telethon capture (scripts/ug_reader.py) writes RAW messages. A parser (built
once we see UG's real format) turns those into the structured records this module
loads. Keeping the model separate means the feature engine never depends on how a
signal was parsed — only on these fields.

Structured signal JSONL record (one per line):
    {"ts": "2026-05-29T13:00:00+00:00", "direction": "long",
     "symbol": "XAUUSD", "tf": "M5",
     "entry": 2658.4, "sl": 2652.1, "tp": 2661.5, "raw": "..."}
Only `ts` and `direction` are required; price geometry is optional (some UG
messages may be entry-only).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_LONG = {"long", "buy", "mua", "enter_long"}
_SHORT = {"short", "sell", "ban", "bán", "enter_short"}


@dataclass(frozen=True)
class Signal:
    ts: pd.Timestamp           # tz-aware UTC, the message/entry time
    direction: str             # normalized 'long' | 'short'
    symbol: str | None = None
    tf: str | None = None
    entry: float | None = None
    sl: float | None = None
    tp: float | None = None
    raw: str = ""

    @property
    def has_geometry(self) -> bool:
        return self.entry is not None and self.sl is not None and self.tp is not None


def normalize_direction(value: str) -> str:
    v = str(value).strip().lower()
    if v in _LONG:
        return "long"
    if v in _SHORT:
        return "short"
    raise ValueError(f"unrecognized direction {value!r}")


def _to_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _f(v) -> float | None:
    return None if v is None or v == "" else float(v)


def signal_from_dict(d: dict) -> Signal:
    return Signal(
        ts=_to_utc(d["ts"]),
        direction=normalize_direction(d["direction"]),
        symbol=d.get("symbol"),
        tf=d.get("tf"),
        entry=_f(d.get("entry")),
        sl=_f(d.get("sl")),
        tp=_f(d.get("tp")),
        raw=d.get("raw", ""),
    )


def load_signals(path: str | Path) -> list[Signal]:
    """Load structured signals from a JSONL file, sorted oldest→newest."""
    path = Path(path)
    out: list[Signal] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(signal_from_dict(json.loads(line)))
    out.sort(key=lambda s: s.ts)
    return out
