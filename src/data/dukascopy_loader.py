"""
Dukascopy XAU/USD historical tick data loader.

Strategy:
- Dukascopy publishes free historical tick data as hourly .bi5 files (LZMA-compressed).
- URL: https://datafeed.dukascopy.com/datafeed/XAUUSD/{year}/{month_zero_indexed:02d}/{day:02d}/{hour:02d}h_ticks.bi5
  NOTE: month is zero-indexed in the URL (January = 00, December = 11).
- Each .bi5 decompresses to a stream of 20-byte big-endian records:
    >IIIff  ->  (ms_offset_from_hour, ask_price_int, bid_price_int, ask_volume, bid_volume)
  For XAUUSD the price divisor is 1000 (e.g. integer 1234567 -> 1234.567 USD).
- Resample ticks to M1/M5/M15/H1/H4 OHLCV with mid price = (bid+ask)/2.
- Output schema mirrors src/data/binance_loader.py:
    columns = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
    timestamp is UTC tz-aware.

Cache layout:
- Raw .bi5 files cached under data/xau/_cache/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}.bi5
- Parquet bars under data/xau/{symbol}/{interval}/{year}-{month:02d}.parquet

No auth required.
"""
from __future__ import annotations

import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# How many concurrent hourly .bi5 fetches. Dukascopy's free public endpoint is
# tolerant of moderate parallelism; tune down if you start seeing rate-limit 5xx.
DEFAULT_WORKERS = 8
# NOTE: We initially tried 16 then 32 workers, which gave great throughput but
# triggered a Dukascopy IP-level rate-limit (TCP connections actively refused for
# extended periods). Default is now 8 workers. If you see "Connection refused" /
# "Max retries exceeded", drop to 4 workers and add jitter, OR wait 30-60 min
# for the rate-limit to lift.

DUKASCOPY_BASE = "https://datafeed.dukascopy.com/datafeed"
USER_AGENT = {"User-Agent": "Mozilla/5.0"}

# XAUUSD price divisor: Dukascopy stores prices as integers scaled by 1000.
PRICE_DIVISOR = {
    "XAUUSD": 1000.0,
}

# Map our timeframe names to pandas resample rules.
INTERVALS = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
}

TICK_STRUCT = struct.Struct(">IIIff")  # 20 bytes per tick
TICK_SIZE = TICK_STRUCT.size


def _hour_url(symbol: str, dt: datetime) -> str:
    """Build the Dukascopy URL for a given UTC hour. Month is zero-indexed."""
    return (
        f"{DUKASCOPY_BASE}/{symbol}/"
        f"{dt.year:04d}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"
    )


def _cache_path(cache_root: Path, symbol: str, dt: datetime) -> Path:
    return (
        cache_root
        / symbol
        / f"{dt.year:04d}"
        / f"{dt.month:02d}"
        / f"{dt.day:02d}"
        / f"{dt.hour:02d}.bi5"
    )


def _fetch_hour(symbol: str, dt: datetime, cache_root: Path, overwrite: bool = False) -> bytes | None:
    """
    Fetch one hourly .bi5 file. Returns raw compressed bytes, or None on 404.
    Caches raw bytes to disk for re-use.
    """
    cache_file = _cache_path(cache_root, symbol, dt)
    if cache_file.exists() and not overwrite:
        return cache_file.read_bytes()

    url = _hour_url(symbol, dt)
    for attempt in range(2):
        try:
            resp = requests.get(url, headers=USER_AGENT, timeout=30)
        except requests.RequestException:
            if attempt == 0:
                time.sleep(2.0)
                continue
            return None

        if resp.status_code == 404:
            return None
        if 500 <= resp.status_code < 600:
            if attempt == 0:
                time.sleep(2.0)
                continue
            return None
        if resp.status_code != 200:
            return None

        data = resp.content
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(data)
        return data

    return None


def _decompress_bi5(data: bytes) -> bytes:
    """Decompress a .bi5 LZMA blob. Empty bytes if the file is empty."""
    if not data:
        return b""
    try:
        return lzma.decompress(data, format=lzma.FORMAT_ALONE)
    except lzma.LZMAError:
        return lzma.decompress(data, format=lzma.FORMAT_AUTO)


def _parse_ticks(decompressed: bytes, hour_dt: datetime, divisor: float) -> pd.DataFrame:
    """
    Parse decompressed tick stream into a DataFrame:
      columns = [timestamp, bid, ask, bid_volume, ask_volume, mid, volume]
    """
    if not decompressed:
        return pd.DataFrame(
            columns=["timestamp", "bid", "ask", "bid_volume", "ask_volume", "mid", "volume"]
        )

    n = len(decompressed) // TICK_SIZE
    if n == 0:
        return pd.DataFrame(
            columns=["timestamp", "bid", "ask", "bid_volume", "ask_volume", "mid", "volume"]
        )

    # Vectorized parse via numpy structured dtype (big-endian).
    dtype = np.dtype([
        ("ms", ">u4"),
        ("ask_i", ">u4"),
        ("bid_i", ">u4"),
        ("ask_v", ">f4"),
        ("bid_v", ">f4"),
    ])
    arr = np.frombuffer(decompressed[: n * TICK_SIZE], dtype=dtype)

    hour_ms = int(hour_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    ts_ms = hour_ms + arr["ms"].astype(np.int64)
    timestamps = pd.to_datetime(ts_ms, unit="ms", utc=True)

    bid = arr["bid_i"].astype(np.float64) / divisor
    ask = arr["ask_i"].astype(np.float64) / divisor
    bid_v = arr["bid_v"].astype(np.float64)
    ask_v = arr["ask_v"].astype(np.float64)
    mid = (bid + ask) / 2.0
    vol = bid_v + ask_v

    return pd.DataFrame({
        "timestamp": timestamps,
        "bid": bid,
        "ask": ask,
        "bid_volume": bid_v,
        "ask_volume": ask_v,
        "mid": mid,
        "volume": vol,
    })


def _resample(ticks: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample tick frame -> OHLCV bars matching Binance loader schema."""
    if ticks.empty:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "trades"]
        )

    df = ticks.set_index("timestamp")
    agg = df["mid"].resample(rule).agg(["first", "max", "min", "last"])
    agg.columns = ["open", "high", "low", "close"]
    agg["volume"] = df["volume"].resample(rule).sum()
    agg["trades"] = df["mid"].resample(rule).count()

    # Drop empty bars (gaps / weekend / holidays).
    agg = agg.dropna(subset=["open"]).reset_index()
    agg = agg.rename(columns={"timestamp": "timestamp"})
    return agg[["timestamp", "open", "high", "low", "close", "volume", "trades"]]


def _hour_range(start: datetime, end: datetime):
    """Yield UTC-aware datetimes at hourly resolution from start to end-1h inclusive."""
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur < end:
        yield cur
        cur += timedelta(hours=1)


def download(
    symbol: str,
    start,
    end=None,
    out_dir: Path | None = None,
    overwrite: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> pd.DataFrame:
    """
    Download Dukascopy ticks for [start, end), resample to M1/M5/M15/H1/H4,
    write parquet per (interval, year, month), and return the M1 DataFrame.

    Args:
        symbol: e.g. "XAUUSD"
        start: ISO date string or datetime (UTC).
        end:   ISO date string or datetime (UTC). Defaults to start + 1 day.
        out_dir: base directory; defaults to DATA_DIR / "xau". Parquet goes under
                 {out_dir}/{symbol}/{interval}/{year}-{month:02d}.parquet.
        overwrite: ignore existing parquet (raw .bi5 cache still re-used).

    Returns the M1 DataFrame across the full range.
    """
    if symbol not in PRICE_DIVISOR:
        raise ValueError(f"Unknown symbol {symbol}. Known: {list(PRICE_DIVISOR)}")

    if isinstance(start, str):
        start = datetime.fromisoformat(start)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    else:
        start = start.astimezone(timezone.utc)

    if end is None:
        end = start + timedelta(days=1)
    elif isinstance(end, str):
        end = datetime.fromisoformat(end)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    else:
        end = end.astimezone(timezone.utc)

    if out_dir is None:
        from src.config import DATA_DIR
        out_dir = DATA_DIR / "xau"
    out_dir = Path(out_dir)
    cache_root = out_dir / "_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    divisor = PRICE_DIVISOR[symbol]
    hours = list(_hour_range(start, end))

    # Pre-pass: parallel-fetch all hours into the .bi5 cache. We materialize raw
    # bytes via threaded HTTP (16 workers by default) — Dukascopy's free public
    # endpoint has ~10s response latency, so serial download takes ~73h for 3 years.
    # Parallel typically drops that to ~5-8h.
    raw_by_dt: dict[datetime, bytes | None] = {}
    fetched = 0
    missing = 0

    def _job(dt: datetime) -> tuple[datetime, bytes | None]:
        return dt, _fetch_hour(symbol, dt, cache_root, overwrite=False)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_job, dt) for dt in hours]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{symbol} ticks (fetch)"):
                dt, raw = fut.result()
                raw_by_dt[dt] = raw
    else:
        for dt in tqdm(hours, desc=f"{symbol} ticks (fetch)"):
            raw_by_dt[dt] = _fetch_hour(symbol, dt, cache_root, overwrite=False)

    # Parse in chronological order so ticks_frames stays sorted.
    tick_frames: list[pd.DataFrame] = []
    for dt in tqdm(hours, desc=f"{symbol} ticks (parse)"):
        raw = raw_by_dt.get(dt)
        if raw is None:
            missing += 1
            continue
        try:
            decompressed = _decompress_bi5(raw)
        except lzma.LZMAError:
            missing += 1
            continue
        df = _parse_ticks(decompressed, dt, divisor)
        if df.empty:
            fetched += 1
            continue
        tick_frames.append(df)
        fetched += 1

    print(f"Hours fetched -> {fetched}, Hours 404/empty -> {missing}")

    if not tick_frames:
        print("No ticks downloaded.")
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "trades"]
        )

    ticks = pd.concat(tick_frames, ignore_index=True)
    ticks = ticks.sort_values("timestamp").reset_index(drop=True)
    print(f"Total ticks -> {len(ticks)}")

    m1_full: pd.DataFrame | None = None

    for interval, rule in INTERVALS.items():
        bars = _resample(ticks, rule)
        if bars.empty:
            continue

        interval_dir = out_dir / symbol / interval
        interval_dir.mkdir(parents=True, exist_ok=True)

        # Persist per (year, month).
        bars_idx = bars.copy()
        bars_idx["_y"] = bars_idx["timestamp"].dt.year
        bars_idx["_m"] = bars_idx["timestamp"].dt.month
        for (year, month), chunk in bars_idx.groupby(["_y", "_m"], sort=True):
            out_file = interval_dir / f"{year:04d}-{month:02d}.parquet"
            chunk = chunk.drop(columns=["_y", "_m"])

            if out_file.exists() and not overwrite:
                existing = pd.read_parquet(out_file)
                merged = (
                    pd.concat([existing, chunk], ignore_index=True)
                    .drop_duplicates("timestamp", keep="last")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
                merged.to_parquet(out_file, index=False)
            else:
                chunk.reset_index(drop=True).to_parquet(out_file, index=False)

        if interval == "M1":
            m1_full = bars.reset_index(drop=True)

    if m1_full is None:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "trades"]
        )
    return m1_full


def load(symbol: str, interval: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load all parquet files for a symbol/interval from disk."""
    if interval not in INTERVALS:
        raise ValueError(f"Invalid interval {interval}. Must be one of {list(INTERVALS)}")
    if data_dir is None:
        from src.config import DATA_DIR
        data_dir = DATA_DIR / "xau" / symbol / interval
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "trades"]
        )
    return (
        pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# Smoke tests live in tests/data/test_dukascopy_loader.py — do not embed test code here.
