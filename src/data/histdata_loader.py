"""HistData.com XAU/USD M1 historical loader.

HistData publishes free M1 OHLC bar data per month/year as ZIP downloads. The
download flow requires a security token (`tk`) that's set by JavaScript on the
download page — direct scraping fails. We delegate the token-and-download dance
to the `histdata` PyPI package (which knows the right HTML extraction), then
parse the resulting ZIPs ourselves.

Per-year ZIPs (~4-5 MB each) cover full M1 for past years; for the current year
HistData publishes per-month ZIPs instead. We handle both transparently.

Timezone: HistData ASCII M1 timestamps are EST (UTC-5, no DST). We convert to
UTC for parity with the other loaders.

Output schema mirrors `src/data/binance_loader.py` and `src/data/dukascopy_loader.py`:
    columns = ["timestamp", "open", "high", "low", "close", "volume", "trades"]
    timestamp is UTC tz-aware.

Cache layout:
    data/xau/_cache/histdata/{symbol}/{year}_M1.zip                — past year ZIP
    data/xau/_cache/histdata/{symbol}/{year}-{month:02d}_M1.zip    — current year monthly ZIP
    data/xau/{symbol}/{interval}/{year}-{month:02d}.parquet         — resampled bars
"""
from __future__ import annotations

import io
import shutil
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

# Third-party — handles HistData's bot-protected download dance.
from histdata import download_hist_data
from histdata.api import Platform, TimeFrame


# HistData ASCII M1 timestamps are EST (UTC-5) year-round (no DST). Per their FAQ.
HISTDATA_TZ_OFFSET = pd.Timedelta(hours=5)  # add to convert EST → UTC.

# Resample rules used to derive M5/M15/H1/H4 from M1.
RESAMPLE_RULES = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
}


def _parse_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    """Parse a HistData ASCII M1 ZIP and return UTC OHLCV.

    HistData CSV format (semicolon-separated, no header):
        20240101 180000;2062.598000;2064.525000;2062.405000;2064.235000;0
    Columns: dt;open;high;low;close;volume (volume is 0 for FX/CFD).
    """
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "trades"])
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(
                f, sep=";", header=None,
                names=["dt_str", "open", "high", "low", "close", "volume"],
                dtype={"dt_str": str},
            )
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "trades"])

    est_naive = pd.to_datetime(df["dt_str"], format="%Y%m%d %H%M%S")
    utc_ts = (est_naive + HISTDATA_TZ_OFFSET).dt.tz_localize("UTC")

    out = pd.DataFrame({
        "timestamp": utc_ts,
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce"),
        "trades": 0,  # HistData doesn't publish trade counts; placeholder for schema parity.
    })
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return out


def _resample_to_tfs(m1: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Resample the M1 frame to all configured higher TFs (and pass through M1)."""
    if m1.empty:
        return {tf: pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "trades"])
                for tf in RESAMPLE_RULES}
    indexed = m1.set_index("timestamp")
    out: dict[str, pd.DataFrame] = {}
    for tf, rule in RESAMPLE_RULES.items():
        if tf == "M1":
            out[tf] = m1.copy()
            continue
        agg = indexed.resample(rule, label="left", closed="left").agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
            "volume": "sum",
        })
        agg = agg.dropna(subset=["open"]).reset_index()
        agg["trades"] = 0
        out[tf] = agg[["timestamp", "open", "high", "low", "close", "volume", "trades"]]
    return out


def _download_year_zip(year: int, symbol: str, cache_root: Path, overwrite: bool = False) -> Path | None:
    """Download a full-year HistData M1 ZIP (used for past years). Returns cached path."""
    target = cache_root / f"{year}_M1.zip"
    if target.exists() and not overwrite:
        return target
    try:
        downloaded = download_hist_data(
            year=year, month=None, pair=symbol.lower(),
            platform=Platform.GENERIC_ASCII, time_frame=TimeFrame.ONE_MINUTE,
            output_directory=str(cache_root),
        )
    except Exception as e:
        print(f"HistData {symbol} {year}: download failed ({type(e).__name__}: {e})")
        return None
    downloaded = Path(downloaded)
    if downloaded != target:
        shutil.move(str(downloaded), str(target))
    return target


def _download_month_zip(year: int, month: int, symbol: str, cache_root: Path, overwrite: bool = False) -> Path | None:
    """Download a single-month HistData M1 ZIP (used for the current year)."""
    target = cache_root / f"{year}-{month:02d}_M1.zip"
    if target.exists() and not overwrite:
        return target
    try:
        downloaded = download_hist_data(
            year=year, month=month, pair=symbol.lower(),
            platform=Platform.GENERIC_ASCII, time_frame=TimeFrame.ONE_MINUTE,
            output_directory=str(cache_root),
        )
    except Exception as e:
        print(f"HistData {symbol} {year}-{month:02d}: download failed ({type(e).__name__}: {e})")
        return None
    downloaded = Path(downloaded)
    if downloaded != target:
        shutil.move(str(downloaded), str(target))
    return target


def download(
    symbol: str,
    start,
    end=None,
    out_dir: Path | None = None,
    overwrite: bool = False,
    pause_seconds: float = 1.0,
) -> pd.DataFrame:
    """Download HistData M1 for [start, end], resample to M1/M5/M15/H1/H4, write parquet.

    Args:
        symbol         — e.g. "XAUUSD"
        start, end     — ISO date strings or datetimes (UTC)
        out_dir        — defaults to DATA_DIR/xau
        overwrite      — re-download even if zip is cached
        pause_seconds  — polite delay between HTTP requests

    Returns the concatenated M1 DataFrame across the whole range (UTC-aware).
    """
    if isinstance(start, str):
        start = datetime.fromisoformat(start)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    else:
        start = start.astimezone(timezone.utc)
    if end is None:
        end = datetime.now(tz=timezone.utc)
    elif isinstance(end, str):
        end = datetime.fromisoformat(end)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    if out_dir is None:
        from src.config import DATA_DIR
        out_dir = DATA_DIR / "xau"
    out_dir = Path(out_dir)
    cache_root = out_dir / "_cache" / "histdata" / symbol
    cache_root.mkdir(parents=True, exist_ok=True)

    # Decide whether to use year-based or month-based downloads.
    # HistData publishes per-year ZIPs for past years and per-month for the current year.
    today = datetime.now(timezone.utc)
    current_year = today.year

    zip_paths: list[Path] = []
    # Year iteration.
    for year in range(start.year, end.year + 1):
        if year < current_year:
            print(f"HistData {symbol} {year}: full-year ZIP")
            path = _download_year_zip(year, symbol, cache_root, overwrite=overwrite)
            if path is not None:
                zip_paths.append(path)
            if pause_seconds > 0:
                time.sleep(pause_seconds)
        else:
            # Current year: month-by-month, only months covered by [start, end].
            first_m = start.month if year == start.year else 1
            last_m = end.month if year == end.year else 12
            # Cap last_m so we don't request future months
            last_m = min(last_m, today.month)
            for month in range(first_m, last_m + 1):
                print(f"HistData {symbol} {year}-{month:02d}: month ZIP")
                path = _download_month_zip(year, month, symbol, cache_root, overwrite=overwrite)
                if path is not None:
                    zip_paths.append(path)
                if pause_seconds > 0:
                    time.sleep(pause_seconds)

    if not zip_paths:
        print("HistData: no ZIPs downloaded.")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "trades"])

    m1_frames: list[pd.DataFrame] = []
    for z in tqdm(zip_paths, desc=f"{symbol} HistData parse"):
        df = _parse_csv_from_zip(z)
        if not df.empty:
            m1_frames.append(df)

    if not m1_frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "trades"])

    full_m1 = pd.concat(m1_frames, ignore_index=True)
    full_m1 = full_m1.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    full_m1 = full_m1[(full_m1["timestamp"] >= start) & (full_m1["timestamp"] <= end)].reset_index(drop=True)

    # Resample + persist to parquet (one shard per [interval, year, month]).
    by_tf = _resample_to_tfs(full_m1)
    for tf, frame in by_tf.items():
        if frame.empty:
            continue
        interval_dir = out_dir / symbol / tf
        interval_dir.mkdir(parents=True, exist_ok=True)
        idx = frame.copy()
        idx["_y"] = idx["timestamp"].dt.year
        idx["_m"] = idx["timestamp"].dt.month
        for (y, m), chunk in idx.groupby(["_y", "_m"], sort=True):
            out_file = interval_dir / f"{y:04d}-{m:02d}.parquet"
            chunk = chunk.drop(columns=["_y", "_m"]).reset_index(drop=True)
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
                chunk.to_parquet(out_file, index=False)

    return full_m1


def load(symbol: str, interval: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load all parquet files for symbol/interval (same API as other loaders)."""
    if interval not in RESAMPLE_RULES:
        raise ValueError(f"interval {interval} not in {list(RESAMPLE_RULES)}")
    if data_dir is None:
        from src.config import DATA_DIR
        data_dir = DATA_DIR / "xau" / symbol / interval
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "trades"])
    return (
        pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
