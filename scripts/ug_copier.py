"""UG signal copier — filter, place limit at the deep edge, manage. MT5 / VPS.

Reads parsed UG signals from a file feed (data/ug/live_signals.jsonl), and for
each NEW signal: decide() (filter TP1∈{100,150}, deep-edge entry, TP from entry,
skip if price past TP1) → place a pending LIMIT (or DRY-log). Tracks placed
pendings and CANCELS any that hasn't filled once price reaches TP1 (don't chase)
or after an expiry.

SAFETY: DRY-RUN by default (real prices, no orders). Pass --live to actually
place orders. Volume defaults to 0.01. Only orders tagged with our MAGIC are ever
touched. Run inside the MT5 interactive session (like the signal bot).

Feed line = one parsed UG signal dict (see scripts/parse_ug_export.py), e.g.:
  {"ts":"...","direction":"long","entry_low":4468,"entry_high":4458,"sl":4448,
   "tps_pip":{"1":150,"2":200,"3":300,"4":400}}

Run (VPS, dry):  python -X utf8 -m scripts.ug_copier
Run (VPS, live): python -X utf8 -m scripts.ug_copier --live --symbol XAUUSDm
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from src.exec.ug_copier_logic import decide, should_cancel_pending, Order

FEED = Path("data/ug/live_signals.jsonl")
# State path is set per run-mode (live vs dry) so a dry-run can never poison the
# live dedup/exposure record. Assigned in main().
STATE = Path("data/ug/copier_state.json")


def _key(sig: dict) -> str:
    return "|".join(str(sig.get(k)) for k in ("ts", "direction", "entry_low", "entry_high", "sl"))


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"done": {}}        # key -> {ticket, order, placed_at} | {skipped: reason}


def _save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")        # atomic write: tmp + replace
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=0), encoding="utf-8")
    os.replace(tmp, STATE)




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
    args = ap.parse_args()

    global STATE
    STATE = Path(f"data/ug/copier_state_{'live' if args.live else 'dry'}.json")
    from src.exec.broker import Mt5Broker, DryRunBroker
    broker = Mt5Broker() if args.live else DryRunBroker()
    mode = "LIVE" if args.live else "DRY-RUN"
    print(f"UG copier {mode} · {args.symbol} · vol {args.volume} · poll {args.poll}s "
          f"· expiry {args.expiry_min}min")
    if args.live:
        print("  !! LIVE order placement enabled !!")

    st = _load_state()
    now_iso = lambda: pd.Timestamp.now(tz="UTC").isoformat()

    while True:
        try:
            px = broker.get_price(args.symbol)
            if not px:
                print(f"  [{now_iso()}] no price for {args.symbol}; retry")
                time.sleep(args.poll); continue
            mid = px["mid"]

            # 1) New signals → decide + place.
            for sig in _read_feed():
                k = _key(sig)
                if k in st["done"]:
                    continue
                d = decide(sig, mid, volume=args.volume)
                if d.action == "skip":
                    print(f"  [{now_iso()}] SKIP {sig.get('direction')} — {d.reason}")
                    st["done"][k] = {"skipped": d.reason, "at": now_iso()}
                elif (exp := broker.open_exposure(args.symbol)) is None:
                    print(f"  [{now_iso()}] HOLD {sig.get('direction')} — exposure unknown "
                          f"(broker query failed); not placing this cycle")
                    continue
                elif exp >= args.max_open:
                    print(f"  [{now_iso()}] HOLD {sig.get('direction')} — max-open "
                          f"{args.max_open} reached (exposure {exp}); reconsider next poll")
                    continue        # don't mark done → reconsider when exposure frees
                else:
                    o = d.order
                    # Pre-mark BEFORE order_send so a crash between send and save can
                    # never double-place on restart. If we then crash, the order (if
                    # it was sent) still carries its own SL/TP — safe, just unmanaged.
                    st["done"][k] = {"status": "placing", "at": now_iso(), "order": o.__dict__}
                    _save_state(st)
                    ticket = broker.place_limit(args.symbol, o)
                    if ticket is None:
                        print(f"  [{now_iso()}] place FAILED {o.order_type} @ {o.entry}")
                        st["done"][k] = {"status": "place_failed", "at": now_iso(),
                                         "order": o.__dict__}   # not retried (safety)
                        _save_state(st)
                        continue
                    print(f"  [{now_iso()}] PLACED {o.order_type} {args.symbol} {o.volume} "
                          f"@ {o.entry} sl={o.sl} tp={o.tp} ticket={ticket}")
                    st["done"][k] = {"ticket": ticket, "placed_at": now_iso(),
                                     "order": o.__dict__}
                    _save_state(st)

            # 2) Manage live pendings (cancel if price reached TP1 unfilled, or expired).
            live_tickets = broker.pending_tickets(args.symbol)
            if live_tickets is None:        # query FAILED — don't mistake for 'all gone'
                print(f"  [{now_iso()}] pending query failed; skip management this cycle")
                time.sleep(args.poll); continue
            for k, rec in st["done"].items():
                tk = rec.get("ticket")
                if tk is None or rec.get("closed"):
                    continue
                if tk not in live_tickets:         # filled or already gone → MT5 manages SL/TP now
                    rec["closed"] = "filled_or_gone"; _save_state(st); continue
                o = Order(**rec["order"])
                age_min = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(rec["placed_at"])).total_seconds() / 60
                if should_cancel_pending(o, mid) or age_min > args.expiry_min:
                    why = "TP1 reached unfilled" if should_cancel_pending(o, mid) else f"expired {age_min:.0f}min"
                    if broker.cancel(tk):
                        print(f"  [{now_iso()}] CANCEL ticket {tk} — {why}")
                        rec["closed"] = why; _save_state(st)
        except Exception as e:
            print(f"  [{now_iso()}] loop error {e}")
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
