"""Parse a Telegram Desktop chat export (messages.html) into UG signal records.

UG ("XAUUSD AI Sclaping UG") self-documents its analysis in every signal, e.g.:

    🔥 UG Trading Ai Signal Phương Pháp 2 🔥
    🔴 SELL XAUUSD
    📍 Entry: 4552 - 4555
    🛑 SL: 4565 (10.0 gia)
    ✅ TP1: 50 pip ... TP4: 200 pip
    📈 Elliott Wave: H1 song 5 xuong, D1 song C len...
    🔍 SMC: M15 quet thanh khoan mua, H1 quet, OB trung tinh
    📊 MA34/MA89:  M5 ⬇ | M15 ⬆ | M30 ⬆ | H1 ⬆  (with values)
    ⚡ Rui ro: 7/10   ⚠️ Khuyen nghi: CAUTION

So we extract the stated components directly. Outputs (data/ug/):
  messages.jsonl  — every message (ts_utc, from, text)         [audit trail]
  signals.jsonl   — structured signals (loadable by src.analysis.signals)

PIP CONVENTION: UG quotes SL in price ("gia") and TP in "pip". For XAU the MT5
convention is 1 pip = 0.1 price, which also reproduces UG's R:R (TP1 50pip=5.0 vs
SL ~10 → R:R 0.5; TP4 200pip=20 → 2.0) matching ug_methods.py. We convert TP pips
→ price with PIP=0.1 and FLAG it as an assumption to verify against the chart.

Usage:
  python -X utf8 -m scripts.parse_ug_export \
      --html "C:/Users/Admin/Downloads/Telegram Desktop/DataExport_2026-06-01/chats/chat_1/messages.html"
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
# NOTE: bs4 is imported lazily inside main() — only the HTML export path needs it.
# parse_signal() (used by the live listener) is pure-regex, so importing it must
# not require bs4 (the VPS venv doesn't have it).

PIP = 0.1                                  # XAU: 1 pip = 0.1 price (assumption — verify)
OUT_DIR = Path("data/ug")

_DIR = {"BUY": "long", "SELL": "short"}
_RE_METHOD = re.compile(r"Ph[uươ]+ng Ph[aá]p\s*(\d+)", re.IGNORECASE)
_RE_DIR = re.compile(r"\b(BUY|SELL)\s+XAUUSD\b")
_RE_ENTRY_RANGE = re.compile(r"Entry:\s*([\d.]+)\s*-\s*([\d.]+)")
_RE_ENTRY_ONE = re.compile(r"Entry:\s*([\d.]+)")
_RE_SL = re.compile(r"SL:\s*([\d.]+)\s*(?:\(\s*([\d.]+)\s*gia\s*\))?")
_RE_TP = re.compile(r"TP(\d):\s*([\d.]+)\s*pip", re.IGNORECASE)
_RE_RISK = re.compile(r"Rui ro:\s*(\d+)\s*/\s*10")
_RE_REC = re.compile(r"Khuyen nghi:\s*([A-Za-z_]+)")
_RE_MA = re.compile(r"\b(M5|M15|M30|H1|H4|D1):\s*([⬆⬇])\s*MA34=([\d.]+)\s*\|\s*MA89=([\d.]+)")
_RE_BLOCK = re.compile(r"Elliott Wave.*?:\s*(.*?)(?:\n\S|\Z)", re.DOTALL)


def _text_of(div) -> str:
    """div.text node → plain text with <br> as newlines, entities unescaped."""
    for br in div.find_all("br"):
        br.replace_with("\n")
    return div.get_text()


def _parse_date(title: str) -> str | None:
    """'25.05.2026 17:31:53 UTC+07:00' → UTC ISO8601."""
    if not title:
        return None
    t = title.replace("UTC", "").strip()           # '25.05.2026 17:31:53 +07:00'
    try:
        dt = datetime.strptime(t, "%d.%m.%Y %H:%M:%S %z")
    except ValueError:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def _parse_ma(text: str) -> dict:
    out = {}
    for tf, arrow, ma34, ma89 in _RE_MA.findall(text):
        out[tf] = {"dir": "up" if arrow == "⬆" else "down",
                   "ma34": float(ma34), "ma89": float(ma89)}
    return out


def parse_signal(ts_utc: str, text: str) -> dict | None:
    """Return a structured signal dict, or None if text isn't a signal."""
    md = _RE_DIR.search(text)
    if not md or "Entry:" not in text:
        return None
    direction = _DIR[md.group(1)]

    er = _RE_ENTRY_RANGE.search(text)
    if er:
        entry_low, entry_high = float(er.group(1)), float(er.group(2))
    else:
        e1 = _RE_ENTRY_ONE.search(text)
        entry_low = entry_high = float(e1.group(1)) if e1 else None
    entry = (entry_low + entry_high) / 2 if entry_low is not None else None

    sl_m = _RE_SL.search(text)
    sl = float(sl_m.group(1)) if sl_m else None
    sl_dist = float(sl_m.group(2)) if sl_m and sl_m.group(2) else (
        abs(entry - sl) if entry is not None and sl is not None else None)

    tps_pip = {int(n): float(p) for n, p in _RE_TP.findall(text)}
    # TP prices from the entry, in trade direction.
    sign = 1 if direction == "long" else -1
    tp_prices = {n: round(entry + sign * p * PIP, 3) for n, p in tps_pip.items()} \
        if entry is not None else {}
    tp1 = tp_prices.get(1)

    method = int(_RE_METHOD.search(text).group(1)) if _RE_METHOD.search(text) else None
    risk = int(_RE_RISK.search(text).group(1)) if _RE_RISK.search(text) else None
    rec_m = _RE_REC.search(text)
    rec = rec_m.group(1) if rec_m else None

    return {
        "ts": ts_utc,
        "direction": direction,
        "symbol": "XAUUSD",
        "tf": "M5",                       # UG scalping signals act on M5 (per ug_methods)
        "entry": entry,
        "sl": sl,
        "tp": tp1,                        # primary target = TP1 (loader uses this)
        # --- UG extras (for analysis) ---
        "method": method,
        "entry_low": entry_low, "entry_high": entry_high,
        "sl_dist": sl_dist,
        "tps_pip": tps_pip,
        "tp_prices": tp_prices,
        "ma": _parse_ma(text),
        "risk": risk,
        "recommendation": rec,
        "raw": text,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", required=True, help="Telegram export messages.html")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    from bs4 import BeautifulSoup        # lazy: only the HTML export path needs bs4
    soup = BeautifulSoup(Path(args.html).read_text(encoding="utf-8"), "html.parser")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_msgs, signals = [], []
    last_from = ""
    for msg in soup.select("div.message.default"):
        date_div = msg.select_one("div.date")
        ts = _parse_date(date_div.get("title", "")) if date_div else None
        from_div = msg.select_one("div.from_name")
        if from_div:                       # 'joined' messages omit from_name → carry forward
            last_from = from_div.get_text().strip()
        text_div = msg.select_one("div.text")
        if not text_div or ts is None:
            continue
        text = _text_of(text_div).strip()
        all_msgs.append({"ts": ts, "from": last_from, "text": text})
        sig = parse_signal(ts, text)
        if sig:
            signals.append(sig)

    msgs_path = out_dir / "messages.jsonl"
    sigs_path = out_dir / "signals.jsonl"
    with msgs_path.open("w", encoding="utf-8") as f:
        for m in all_msgs:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with sigs_path.open("w", encoding="utf-8") as f:
        for s in signals:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_long = sum(1 for s in signals if s["direction"] == "long")
    methods = {}
    miss = {"entry": 0, "sl": 0, "tp": 0, "ma": 0, "risk": 0}
    for s in signals:
        methods[s["method"]] = methods.get(s["method"], 0) + 1
        for k in ("entry", "sl", "tp"):
            if s[k] is None:
                miss[k] += 1
        if not s["ma"]:
            miss["ma"] += 1
        if s["risk"] is None:
            miss["risk"] += 1
    print(f"Parsed {len(all_msgs)} messages → {len(signals)} signals "
          f"(long {n_long} / short {len(signals)-n_long})")
    print(f"  methods: {methods} · missing fields: {miss}")
    print(f"  → {sigs_path}\n  → {msgs_path}")
    if signals:
        s = signals[-1]
        print(f"\nSample (latest): {s['ts']} {s['direction']} entry={s['entry']} "
              f"sl={s['sl']} tp1={s['tp']} risk={s['risk']} rec={s['recommendation']}")
        print(f"  MA: {s['ma']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
