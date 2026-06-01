"""Export historical OHLCV from the Exness MT5 terminal to backtest parquet.

Runs on the VPS (where the MT5 terminal is logged in). Pulls NATIVE per-timeframe
bars — the same bars the live bot reads via mt5_feed — so the backtest trades on
the exact price series the live bot does (parity, no PAXG proxy / PRICE_OFFSET).

Writes data/xau/<symbol-out>/<tf>/<YYYY>-<MM>.parquet in the schema the backtest
loader expects: [timestamp(UTC), open, high, low, close, volume, trades]. That
makes it a drop-in for `python -m scripts.optimize_all --symbols XAUUSD`.

History DEPTH is whatever the terminal has cached. copy_rates_from_pos(0, n)
returns up to `n` most recent bars and however many the broker actually has —
so this is bounded by Exness + the terminal's "Max bars" setting. The summary
printed at the end tells you how far back you actually got per timeframe.

Usage (on the VPS):
  C:\\mt5-bot\\.venv\\Scripts\\python.exe -X utf8 -m scripts.fetch_mt5_history
  ... -m scripts.fetch_mt5_history --tfs M5 M30 H1 --symbol-mt5 XAUUSDm
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from src.config import DATA_DIR
from src.data import mt5_feed

# Default request size per TF — large enough to grab all the broker keeps. The
# call returns fewer if that's all there is; no error from over-asking.
DEFAULT_BARS = {"M5": 300_000, "M15": 200_000, "M30": 150_000,
                "H1": 100_000, "H4": 50_000}


def _write_sharded(df: pd.DataFrame, out_dir) -> int:
    """Write one parquet shard per (year, month), merging into any existing shard."""
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = df.copy()
    idx["_y"] = idx["timestamp"].dt.year
    idx["_m"] = idx["timestamp"].dt.month
    written = 0
    for (y, m), chunk in idx.groupby(["_y", "_m"], sort=True):
        out_file = out_dir / f"{y:04d}-{m:02d}.parquet"
        chunk = chunk.drop(columns=["_y", "_m"]).reset_index(drop=True)
        if out_file.exists():
            existing = pd.read_parquet(out_file)
            chunk = (pd.concat([existing, chunk], ignore_index=True)
                     .drop_duplicates("timestamp", keep="last")
                     .sort_values("timestamp").reset_index(drop=True))
        chunk.to_parquet(out_file, index=False)
        written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol-mt5", default=os.environ.get("MT5_SYMBOL", "XAUUSDm").strip(),
                    help="Broker symbol in the MT5 terminal (e.g. XAUUSDm)")
    ap.add_argument("--symbol-out", default="XAUUSD",
                    help="Output symbol dir name (must match the backtest registry)")
    ap.add_argument("--tfs", nargs="+", default=["M5", "M30", "H1"],
                    help="Timeframes to export")
    ap.add_argument("--bars", type=int, default=None,
                    help="Override bars-per-TF (default: per-TF table)")
    args = ap.parse_args()

    base = DATA_DIR / "xau" / args.symbol_out
    print(f"Exporting {args.symbol_mt5} → {base}  (tfs: {', '.join(args.tfs)})\n")

    summary = []
    for tf in args.tfs:
        n = args.bars or DEFAULT_BARS.get(tf, 100_000)
        try:
            df = mt5_feed.bars(args.symbol_mt5, tf, n=n)
        except Exception as e:
            print(f"  {tf}: ERROR {e}")
            summary.append((tf, 0, "-", "-"))
            continue
        if df.empty:
            print(f"  {tf}: no bars")
            summary.append((tf, 0, "-", "-"))
            continue
        df = df.assign(trades=0)                       # schema parity with loaders
        df = (df.sort_values("timestamp").drop_duplicates("timestamp")
              .reset_index(drop=True))
        shards = _write_sharded(df, base / tf)
        first, last = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
        span_days = (last - first).days
        print(f"  {tf}: {len(df):>7} bars · {first.date()} → {last.date()} "
              f"(~{span_days}d, {shards} shards)")
        summary.append((tf, len(df), str(first), str(last)))

    print("\nDone. Backtest with:  python -m scripts.optimize_all --symbols "
          f"{args.symbol_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
