"""MT5 (Exness) price sync — runs on a Windows VPS, keeps Railway calibrated.

The cloud signal bot generates levels from PAXG (a gold proxy). Your broker
(Exness) prices XAU slightly differently. This helper — run on a Windows machine
where the Exness **MetaTrader 5** terminal is installed and logged in — reads the
LIVE Exness XAU price, computes `offset = Exness_mid − PAXG_mid`, and pushes it to
the Railway bot's `PRICE_OFFSET` variable so every signal's entry/SL/TP matches
your Exness chart.

To avoid restarting the cloud bot constantly (a variable change triggers a
redeploy), the offset is only pushed when it drifts past a threshold.

`MetaTrader5` is Windows-only and needs the terminal running. Everything that
does NOT need MT5 (PAXG fetch, Railway read/write, offset math) is exercised by
`--selftest`, which runs anywhere.

Setup (Windows VPS):
  1. Install Exness MT5 terminal; log in (account/password/server from Exness
     Personal Area → your trading account → "MT5" credentials).
  2. pip install MetaTrader5 requests pandas
  3. Fill configs/mt5.json (copy from configs/mt5.example.json).
  4. python -X utf8 scripts/mt5_sync.py --selftest      # no MT5 needed
     python -X utf8 scripts/mt5_sync.py --price         # live Exness price
     python -X utf8 scripts/mt5_sync.py --loop 300      # sync every 5 min
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

CFG_PATH = Path("configs/mt5.json")
RW_URL = "https://backboard.railway.app/graphql/v2"
PAXG_KLINE = ("https://api.binance.com/api/v3/klines"
              "?symbol=PAXGUSDT&interval=1m&limit=1")


# ---------- config ----------
def load_cfg() -> dict:
    if not CFG_PATH.exists():
        raise SystemExit(f"missing {CFG_PATH} (copy configs/mt5.example.json)")
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


# ---------- prices ----------
def paxg_mid() -> float:
    r = requests.get(PAXG_KLINE, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    k = r.json()[0]
    return (float(k[2]) + float(k[3])) / 2.0          # (high+low)/2 of last 1m


def mt5_connect(cfg: dict):
    import MetaTrader5 as mt5                          # lazy: Windows-only
    m = cfg.get("mt5", {})
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize() failed: {mt5.last_error()}")
    if m.get("account"):
        ok = mt5.login(int(m["account"]), password=m.get("password", ""),
                       server=m.get("server", ""))
        if not ok:
            raise SystemExit(f"mt5.login failed: {mt5.last_error()}")
    return mt5


def exness_xau(cfg: dict) -> tuple[float, str]:
    """Return (mid_price, symbol) from the live Exness terminal."""
    mt5 = mt5_connect(cfg)
    candidates = [cfg.get("mt5", {}).get("symbol")] if cfg.get("mt5", {}).get("symbol") else []
    candidates += ["XAUUSD", "XAUUSDm", "XAUUSD.", "GOLD", "GOLDm"]
    for sym in [c for c in candidates if c]:
        if not mt5.symbol_select(sym, True):
            continue
        t = mt5.symbol_info_tick(sym)
        if t and t.bid > 0 and t.ask > 0:
            return (t.bid + t.ask) / 2.0, sym
    raise SystemExit("No XAU symbol found in the terminal (set mt5.symbol in config).")


# ---------- railway ----------
def _rw_headers(cfg: dict) -> dict:
    return {"Project-Access-Token": cfg["railway"]["token"],
            "Content-Type": "application/json"}


def rw_get_offset(cfg: dict) -> float | None:
    rw = cfg["railway"]
    q = ("query($p:String!,$e:String!,$s:String!){ variables("
         "projectId:$p, environmentId:$e, serviceId:$s) }")
    v = {"p": rw["project_id"], "e": rw["environment_id"], "s": rw["service_id"]}
    d = requests.post(RW_URL, headers=_rw_headers(cfg),
                      json={"query": q, "variables": v}, timeout=30).json()
    vars_ = (d.get("data") or {}).get("variables") or {}
    try:
        return float(vars_.get("PRICE_OFFSET"))
    except (TypeError, ValueError):
        return None


def rw_set_offset(cfg: dict, offset: float) -> bool:
    rw = cfg["railway"]
    m = ("mutation($i:VariableUpsertInput!){ variableUpsert(input:$i) }")
    i = {"projectId": rw["project_id"], "environmentId": rw["environment_id"],
         "serviceId": rw["service_id"], "name": "PRICE_OFFSET", "value": f"{offset:.2f}"}
    d = requests.post(RW_URL, headers=_rw_headers(cfg),
                      json={"query": m, "variables": {"i": i}}, timeout=30).json()
    return "errors" not in d


# ---------- calibrate ----------
def calibrate(cfg: dict, push: bool, mock_exness: float | None = None) -> dict:
    paxg = paxg_mid()
    if mock_exness is not None:
        ex, sym = mock_exness, "MOCK"
    else:
        ex, sym = exness_xau(cfg)
    offset = ex - paxg
    cur = rw_get_offset(cfg)
    thr = float(cfg.get("push_threshold", 0.5))
    changed = cur is None or abs(offset - cur) >= thr
    pushed = False
    if push and changed:
        pushed = rw_set_offset(cfg, offset)
    return {"paxg": paxg, "exness": ex, "symbol": sym, "offset": offset,
            "railway_offset": cur, "drift": None if cur is None else offset - cur,
            "threshold": thr, "would_push": changed, "pushed": pushed}


def _print(r: dict) -> None:
    print(f"  PAXG={r['paxg']:.2f}  Exness({r['symbol']})={r['exness']:.2f}  "
          f"offset={r['offset']:+.2f}")
    print(f"  Railway PRICE_OFFSET now={r['railway_offset']}  "
          f"drift={r['drift']}  thr={r['threshold']}  "
          f"{'PUSHED' if r['pushed'] else ('would push' if r['would_push'] else 'in range, no push')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="no MT5: PAXG + Railway + offset math with a mock price")
    ap.add_argument("--mock", type=float, help="mock Exness price for selftest")
    ap.add_argument("--price", action="store_true", help="print live Exness price")
    ap.add_argument("--calibrate", action="store_true", help="compute offset (no push)")
    ap.add_argument("--push", action="store_true", help="push offset to Railway if drifted")
    ap.add_argument("--loop", type=int, default=0, help="repeat every N seconds (+push)")
    args = ap.parse_args()
    cfg = load_cfg()

    if args.selftest:
        mock = args.mock if args.mock is not None else paxg_mid() + 1.23
        print("SELFTEST (mock Exness = PAXG + 1.23 unless --mock):")
        _print(calibrate(cfg, push=False, mock_exness=mock))
        print("  Railway read OK:", rw_get_offset(cfg) is not None or "offset unset")
        return 0
    if args.price:
        ex, sym = exness_xau(cfg)
        print(f"Exness {sym} mid = {ex:.2f}")
        return 0
    if args.loop > 0:
        print(f"syncing every {args.loop}s — Ctrl+C to stop")
        while True:
            try:
                _print(calibrate(cfg, push=True))
            except Exception as e:
                print(f"  ERROR {e}")
            time.sleep(args.loop)
        return 0
    _print(calibrate(cfg, push=args.push))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
