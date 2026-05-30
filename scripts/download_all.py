"""Download 2-3 years of BTC + XAU data into data/ for backtesting.

Per `[[feedback-backtest-scope]]`: 1-2 months isn't enough; we need 2-3+ years.

BTC source: Binance monthly archive ZIPs at data.binance.vision (no auth).
XAU source: Dukascopy hourly tick .bi5 files (no auth).

Usage:
  python -m scripts.download_all                # downloads default 3yr window
  python -m scripts.download_all --start 2022-01-01 --end 2026-05-01
  python -m scripts.download_all --skip-btc     # XAU only
  python -m scripts.download_all --skip-xau     # BTC only

The script is idempotent — existing parquet files are reused unless --overwrite.
For Dukascopy, downloading every weekday hour over 3 years is SLOW (~25,000 HTTP
requests). The .bi5 cache makes the second run fast. Plan for ~2-4 hours on
first run for a full 3-year XAU pull.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

from src.data import binance_loader, histdata_loader


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_all")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BTC + XAU historical data.")
    default_end = datetime.now(timezone.utc).date()
    default_start = default_end.replace(year=default_end.year - 3)
    parser.add_argument("--start", default=default_start.isoformat(),
                        help="ISO date for start of download (default: 3 years ago)")
    parser.add_argument("--end", default=default_end.isoformat(),
                        help="ISO date for end (default: today)")
    parser.add_argument("--btc-symbol", default="BTCUSDT")
    parser.add_argument("--btc-intervals", nargs="+", default=["1m", "5m", "15m", "1h"],
                        help="Binance kline intervals to download")
    parser.add_argument("--xau-symbol", default="XAUUSD")
    parser.add_argument("--skip-btc", action="store_true")
    parser.add_argument("--skip-xau", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.skip_btc:
        for interval in args.btc_intervals:
            logger.info(f"BTC {args.btc_symbol} {interval}: {args.start} -> {args.end}")
            df = binance_loader.download(
                args.btc_symbol, interval=interval,
                start=args.start, end=args.end,
                overwrite=args.overwrite,
            )
            logger.info(f"  -> {len(df)} bars")

    if not args.skip_xau:
        # HistData M1 zips (full-year for past years, per-month for current year).
        # Loader resamples to M1/M5/M15/H1/H4 parquet shards.
        logger.info(f"XAU {args.xau_symbol}: {args.start} -> {args.end}")
        df = histdata_loader.download(
            args.xau_symbol, start=args.start, end=args.end, overwrite=args.overwrite,
        )
        logger.info(f"  -> {len(df)} M1 bars")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
