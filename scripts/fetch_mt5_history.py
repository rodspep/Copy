"""Export historical OHLCV from the Exness MT5 terminal to CSV for analysis/backtest.

Runs on the VPS (where the MT5 terminal is logged in), inside the MT5 interactive
session via fetch_mt5.bat → run_fetch.bat. Pulls NATIVE per-timeframe bars (the
same bars the live bot reads via mt5_feed) so analysis/backtest use the exact live
price series — parity, no PAXG proxy / PRICE_OFFSET.

Writes CSV (no pyarrow dependency on the VPS venv): one file per timeframe,
columns [timestamp(UTC ISO), open, high, low, close, volume] — directly loadable
by src.data.tv_loader.load_tv_csv, e.g.:
    python -m scripts.analyze_ug --signals data/ug/signals.jsonl --tv data/xau/XAUUSD_M5.csv

History depth = whatever the terminal cached; the summary reports how far back
each TF actually goes.

Usage (on the VPS, via run_fetch.bat):
  python -X utf8 -m scripts.fetch_mt5_history --tfs M5 M30 H1
"""
from __future__ import annotations

import argparse
import os

from src.config import DATA_DIR
from src.data import mt5_feed

# Request size per TF. MT5 copy_rates_from_pos rejects counts above the
# terminal's max-bars setting with (-2 Invalid params), so we start modest
# (plenty for the recent UG window + MA89 warmup) and HALVE on failure.
DEFAULT_BARS = {"M1": 50_000, "M5": 30_000, "M15": 20_000, "M30": 15_000,
                "H1": 10_000, "H4": 5_000}
MIN_BARS = 2_000


def _fetch_with_fallback(symbol: str, tf: str, n: int):
    """bars(n); on 'Invalid params'/empty, retry halving n down to MIN_BARS."""
    while n >= MIN_BARS:
        try:
            return mt5_feed.bars(symbol, tf, n=n)
        except Exception as e:
            if "Invalid params" in str(e) or "no rates" in str(e):
                n //= 2
                print(f"    {tf}: retry with n={n} ({e})")
                continue
            raise
    raise RuntimeError(f"{tf}: could not fetch even {MIN_BARS} bars")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol-mt5", default=os.environ.get("MT5_SYMBOL", "XAUUSDm").strip(),
                    help="Broker symbol in the MT5 terminal (e.g. XAUUSDm)")
    ap.add_argument("--symbol-out", default="XAUUSD",
                    help="Output file prefix (match the backtest registry name)")
    ap.add_argument("--tfs", nargs="+", default=["M5", "M30", "H1"])
    ap.add_argument("--bars", type=int, default=None, help="override bars-per-TF")
    args = ap.parse_args()

    out_dir = DATA_DIR / "xau"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {args.symbol_mt5} → {out_dir} (tfs: {', '.join(args.tfs)})\n")

    for tf in args.tfs:
        n = args.bars or DEFAULT_BARS.get(tf, 10_000)
        try:
            df = _fetch_with_fallback(args.symbol_mt5, tf, n)
        except Exception as e:
            print(f"  {tf}: ERROR {e}")
            continue
        if df.empty:
            print(f"  {tf}: no bars")
            continue
        df = (df.sort_values("timestamp").drop_duplicates("timestamp")
              .reset_index(drop=True))
        out_file = out_dir / f"{args.symbol_out}_{tf}.csv"
        df.to_csv(out_file, index=False)
        first, last = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
        print(f"  {tf}: {len(df):>7} bars · {first.date()} → {last.date()} "
              f"(~{(last - first).days}d) → {out_file.name}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
