"""Binance PAXGUSDT loader — fresh, clean gold proxy (24/7, no lag).

PAXG (Paxos Gold) is an ERC-20 token where 1 PAXG = 1 fine troy ounce of London
Good Delivery gold. It trades on Binance against USDT around the clock and
tracks XAU spot within a small, slowly-varying basis.

Why this matters for THIS project:
  - HistData XAU M5 (our backtest source) has unreliable intrabar OHLC: its 5-min
    return correlation with TWO independent fresh feeds (Yahoo GC=F and Binance
    PAXG) is only ~0.10, whereas PAXG vs GC=F is ~0.91. So HistData's bar
    high/low — which decide TP/SL hits — are noisy.
  - PAXG is Binance-grade klines (clean OHLC), always current, free, no key.
  - For verifying signal-time windows (UG-style TP/SL outcomes) PAXG is the most
    trustworthy free source: it covers up to "now" and its intrabar path matches
    an independent feed at 0.91.

Caveat: PAXG carries a basis vs XAU spot (a few to a few tens of dollars, slowly
varying). For RELATIVE measurements (pip distance from entry to TP/SL) the basis
is irrelevant — use PAXG's own price as the entry reference. For mapping an
absolute XAU spot level onto PAXG, subtract the basis first.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR

_BINANCE = "https://api.binance.com/api/v3/klines"
_INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}
CACHE_DIR = DATA_DIR / "paxg"


def _fetch_chunk(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    url = (f"{_BINANCE}?symbol={symbol}&interval={interval}"
           f"&startTime={start_ms}&endTime={end_ms}&limit=1000")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def download(symbol: str = "PAXGUSDT", interval: str = "5m",
             start: str = "2026-04-01", end: str | None = None,
             *, overwrite: bool = False) -> pd.DataFrame:
    """Download PAXGUSDT klines, paginating Binance's 1000-bar limit. Cache to parquet.

    Returns DataFrame [timestamp, open, high, low, close, volume].
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{symbol}_{interval}.parquet"
    if cache.exists() and not overwrite:
        return pd.read_parquet(cache)

    step = _INTERVAL_MS[interval]
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    if end:
        end_ts = pd.Timestamp(end, tz="UTC")
    else:
        end_ts = pd.Timestamp.now(tz="UTC")
    end_ms = int(end_ts.timestamp() * 1000)

    rows = []
    cur = start_ms
    while cur < end_ms:
        chunk_end = min(cur + 1000 * step, end_ms)
        data = _fetch_chunk(symbol, interval, cur, chunk_end)
        if not data:
            cur = chunk_end
            continue
        rows.extend(data)
        last_open = data[-1][0]
        cur = last_open + step
        time.sleep(0.25)  # be gentle with the public endpoint

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["ot", "o", "h", "l", "c", "v", "ct",
                                     "qv", "n", "tb", "tq", "ig"])
    df = df.drop_duplicates("ot")
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["ot"], unit="ms", utc=True),
        "open": df["o"].astype(float), "high": df["h"].astype(float),
        "low": df["l"].astype(float), "close": df["c"].astype(float),
        "volume": df["v"].astype(float),
    }).sort_values("timestamp").reset_index(drop=True)
    out.to_parquet(cache, index=False)
    return out


def load(symbol: str = "PAXGUSDT", interval: str = "5m") -> pd.DataFrame:
    cache = CACHE_DIR / f"{symbol}_{interval}.parquet"
    if not cache.exists():
        raise FileNotFoundError(f"{cache} not found — run paxg_loader.download() first")
    return pd.read_parquet(cache)
