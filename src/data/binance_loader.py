"""
Binance public historical data loader.

Strategy:
- Bulk historical: download monthly ZIP from data.binance.vision (CSV inside, ~10MB/month for 1m).
- Recent (current month): fall back to REST /api/v3/klines (1000 candles per call).
- Output: parquet partitioned by year, stored under data/btc/<symbol>/<timeframe>/.

This is the official Binance public archive. No auth required.
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm

ARCHIVE_BASE = "https://data.binance.vision/data/spot/monthly/klines"
REST_BASE = "https://api.binance.com/api/v3/klines"

# Binance kline CSV columns
COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

VALID_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}


def _month_range(start: datetime, end: datetime) -> Iterable[tuple[int, int]]:
    """Yield (year, month) tuples from start to end inclusive."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def _download_monthly_zip(symbol: str, interval: str, year: int, month: int) -> pd.DataFrame | None:
    """Download one monthly ZIP file from Binance archive. Returns None if not yet available."""
    url = f"{ARCHIVE_BASE}/{symbol}/{interval}/{symbol}-{interval}-{year:04d}-{month:02d}.zip"
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, header=None, names=COLS)

    # Binance switched timestamp unit from ms to us in 2025-01. Detect and normalize.
    if df["open_time"].iloc[0] > 1e15:
        df["open_time"] = pd.to_datetime(df["open_time"], unit="us", utc=True)
    else:
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    # Keep `taker_buy_base` for true Cumulative-Volume-Delta (order-flow) indicators.
    df = df[["open_time", "open", "high", "low", "close", "volume", "trades", "taker_buy_base"]]
    df = df.rename(columns={"open_time": "timestamp"})
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


def _fetch_rest(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fall back to REST API for the current month or very recent gaps."""
    all_rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(REST_BASE, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_rows.extend(batch)
        last_open = batch[-1][0]
        cursor = last_open + 1
        if len(batch) < 1000:
            break

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "trades", "taker_buy_base"])

    df = pd.DataFrame(all_rows, columns=COLS)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume", "trades", "taker_buy_base"]]
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def download(
    symbol: str,
    interval: str,
    start: str | datetime,
    end: str | datetime | None = None,
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Download Binance klines for [start, end].

    Args:
        symbol: e.g. "BTCUSDT"
        interval: one of "1m", "5m", "15m", "1h", "4h", "1d"
        start, end: ISO date strings or datetime. end defaults to today (UTC).
        out_dir: where to save parquet. Defaults to data/btc/<symbol>/<interval>/.
        overwrite: if False, skip months that already have parquet files.

    Returns the concatenated DataFrame across the full range.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid interval {interval}. Must be one of {VALID_INTERVALS}.")

    if isinstance(start, str):
        start = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    if end is None:
        end = datetime.now(tz=timezone.utc)
    elif isinstance(end, str):
        end = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    if out_dir is None:
        from src.config import DATA_DIR
        out_dir = DATA_DIR / "btc" / symbol / interval
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(tz=timezone.utc)
    current_month = (now.year, now.month)

    frames: list[pd.DataFrame] = []
    months = list(_month_range(start, end))

    for year, month in tqdm(months, desc=f"{symbol} {interval}"):
        out_file = out_dir / f"{year:04d}-{month:02d}.parquet"

        if out_file.exists() and not overwrite and (year, month) != current_month:
            df = pd.read_parquet(out_file)
            frames.append(df)
            continue

        if (year, month) == current_month:
            # Use REST API for current month (archive not yet available)
            month_start = datetime(year, month, 1, tzinfo=timezone.utc)
            df = _fetch_rest(
                symbol, interval,
                int(month_start.timestamp() * 1000),
                int(now.timestamp() * 1000),
            )
        else:
            df = _download_monthly_zip(symbol, interval, year, month)
            if df is None:
                # Archive 404 — too recent. Try REST.
                month_start = datetime(year, month, 1, tzinfo=timezone.utc)
                if month == 12:
                    month_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                else:
                    month_end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
                df = _fetch_rest(
                    symbol, interval,
                    int(month_start.timestamp() * 1000),
                    int(month_end.timestamp() * 1000),
                )

        if df is None or df.empty:
            continue

        # Clip to requested range
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        if df.empty:
            continue

        df.to_parquet(out_file, index=False)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    full = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    return full.reset_index(drop=True)


def load(symbol: str, interval: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load all parquet files for a symbol/interval from disk."""
    if data_dir is None:
        from src.config import DATA_DIR
        data_dir = DATA_DIR / "btc" / symbol / interval
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values("timestamp").reset_index(drop=True)


# Smoke tests live in tests/data/test_binance_loader.py — do not embed test code here.
