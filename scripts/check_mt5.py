"""Quick MT5 connectivity + data check — run on the VPS after installing MT5.

Verifies the terminal is reachable, finds your gold symbol, prints the live
price, and pulls a few H1/M30 bars so you know the bot's data feed works before
starting it. No orders are placed.

    python -X utf8 scripts\\check_mt5.py
    python -X utf8 scripts\\check_mt5.py --symbol XAUUSDm
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, help="force a gold symbol")
    args = ap.parse_args()

    try:
        import MetaTrader5 as mt5
    except Exception as e:
        print("FAIL: MetaTrader5 package not installed ->", e)
        print("Run: pip install MetaTrader5")
        return 1

    if not mt5.initialize():
        print("FAIL: mt5.initialize() ->", mt5.last_error())
        print("Is the MetaTrader 5 terminal installed and LOGGED IN?")
        return 1

    ti = mt5.terminal_info()
    ai = mt5.account_info()
    print("Terminal:", getattr(ti, "name", "?"), "| connected:", getattr(ti, "connected", "?"))
    if ai:
        print(f"Account: #{ai.login} {ai.server} | balance {ai.balance} {ai.currency}")

    candidates = ([args.symbol] if args.symbol else []) + \
                 ["XAUUSD", "XAUUSDm", "XAUUSD.", "GOLD", "GOLDm", "XAUUSDc"]
    found = None
    for s in [c for c in candidates if c]:
        if mt5.symbol_select(s, True):
            t = mt5.symbol_info_tick(s)
            if t and t.bid > 0:
                found = s
                print(f"\nSymbol OK: {s}  bid={t.bid}  ask={t.ask}  mid={(t.bid+t.ask)/2:.2f}")
                break
    if not found:
        print("\nFAIL: no gold symbol found. Open Market Watch in MT5, right-click ->")
        print("'Show All', find your gold symbol, then re-run with --symbol NAME")
        mt5.shutdown()
        return 1

    for tf_name, tf in [("H1", mt5.TIMEFRAME_H1), ("M30", mt5.TIMEFRAME_M30)]:
        rates = mt5.copy_rates_from_pos(found, tf, 0, 3)
        n = 0 if rates is None else len(rates)
        last = "" if not n else f" last close={rates[-1]['close']:.2f}"
        print(f"  {tf_name}: {n} bars fetched{last}")

    mt5.shutdown()
    print("\nALL GOOD — set MT5_SYMBOL =", found, "in run_bot_vps.bat if not XAUUSDm,")
    print("then start the bot with run_bot_vps.bat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
