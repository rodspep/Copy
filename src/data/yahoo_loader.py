"""Yahoo Finance loader for cross-asset daily data (DXY, US10Y, SPX, etc.).

Pull free OHLCV from Yahoo Finance v7 chart endpoint, cache to parquet.

Used as auxiliary context for XAU/BTC strategies:
  - DXY (DX-Y.NYB): inverse-correlated with XAU; trade gold long only when DXY weak.
  - US10Y yield (^TNX): real-yield context for gold.
  - SPX (^GSPC): risk-on/off regime.

Loader is intentionally minimal — daily resolution is enough for regime filtering.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR


SYMBOL_MAP = {
    "DXY": "DX-Y.NYB",       # US Dollar Index (ICE)
    "US10Y": "^TNX",          # 10-Year Treasury Note Yield
    "SPX": "^GSPC",           # S&P 500
    "OIL": "CL=F",            # Crude Oil futures
    "VIX": "^VIX",            # Volatility Index
}

CACHE_DIR = DATA_DIR / "yahoo"


def _yahoo_chart(yahoo_symbol: str, start: str, end: str,
                 interval: str = "1d") -> pd.DataFrame:
    """Fetch from Yahoo Finance v7 chart API. Returns DataFrame or empty."""
    p1 = int(time.mktime(time.strptime(start, "%Y-%m-%d")))
    p2 = int(time.mktime(time.strptime(end, "%Y-%m-%d")))
    sym_enc = urllib.parse.quote(yahoo_symbol)
    url = f"https://query1.finance.yahoo.com/v7/finance/chart/{sym_enc}?period1={p1}&period2={p2}&interval={interval}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    r = data.get("chart", {}).get("result", [None])[0]
    if r is None or not r.get("timestamp"):
        return pd.DataFrame()
    ts = r["timestamp"]
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(ts, unit="s", utc=True),
        "open": q.get("open"),
        "high": q.get("high"),
        "low": q.get("low"),
        "close": q.get("close"),
        "volume": q.get("volume", [0] * len(ts)),
    })
    # Drop rows with all-null OHLCV
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def download(symbol: str, start: str, end: str, *, overwrite: bool = False) -> pd.DataFrame:
    """Download a Yahoo symbol and cache to parquet.

    Args:
        symbol: friendly name from SYMBOL_MAP ('DXY', 'US10Y', etc.) OR raw Yahoo ticker
        start: 'YYYY-MM-DD'
        end:   'YYYY-MM-DD'
        overwrite: re-download even if cache exists

    Returns daily DataFrame with columns [timestamp, open, high, low, close, volume].
    """
    yahoo_sym = SYMBOL_MAP.get(symbol.upper(), symbol)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol.upper()}_1d.parquet"
    if cache_path.exists() and not overwrite:
        return pd.read_parquet(cache_path)
    df = _yahoo_chart(yahoo_sym, start, end, interval="1d")
    if df.empty:
        return df
    df.to_parquet(cache_path, index=False)
    return df


def load(symbol: str) -> pd.DataFrame:
    """Load daily data from cache. Raises FileNotFoundError if not downloaded."""
    cache_path = CACHE_DIR / f"{symbol.upper()}_1d.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} not found — run yahoo_loader.download({symbol!r}, ...) first"
        )
    return pd.read_parquet(cache_path)


def align_daily_to_ltf(ltf: pd.DataFrame, daily: pd.DataFrame,
                       cols: list[str], suffix: str = "") -> pd.DataFrame:
    """Forward-fill daily Yahoo data onto LTF (M5) timestamps with no-lookahead.

    A daily bar dated D becomes "available" at D + 1 day (since daily close at
    21:00 UTC is the close-of-day; we conservatively make it available the next
    UTC day at midnight). The merge uses pd.merge_asof with that availability
    timestamp.
    """
    if "timestamp" not in ltf.columns or "timestamp" not in daily.columns:
        raise ValueError("both ltf and daily must have 'timestamp' column")
    d = daily.copy()
    # availability = day after the daily bar's close, at 00:00 UTC
    d["__avail"] = d["timestamp"].dt.normalize() + pd.Timedelta(days=1)
    d = d.sort_values("__avail")

    out = ltf[["timestamp"]].copy().sort_values("timestamp").reset_index(drop=False)
    merged = pd.merge_asof(
        out, d[["__avail"] + cols],
        left_on="timestamp", right_on="__avail",
        direction="backward", allow_exact_matches=True,
    ).sort_values("index").reset_index(drop=True)
    if suffix:
        rename = {c: f"{c}{suffix}" for c in cols}
        merged = merged.rename(columns=rename)
        cols = [f"{c}{suffix}" for c in cols]
    aligned = merged[cols].copy()
    aligned.index = ltf.index
    return aligned
