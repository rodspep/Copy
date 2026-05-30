"""
Smoke + sanity tests for the Binance loader.

Run with: python -m tests.data.test_binance_loader
or:        python -m pytest tests/data/test_binance_loader.py
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.data.binance_loader import download, load


def test_download_recent_btc_5m(days: int = 35) -> pd.DataFrame:
    """Download the last `days` of BTCUSDT 5m and assert basic invariants."""
    start = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    df = download("BTCUSDT", "5m", start=start)

    assert not df.empty, "Download returned empty frame"
    assert df["timestamp"].is_monotonic_increasing, "Timestamps not sorted"
    assert df["timestamp"].dt.tz is not None, "Timestamps must be tz-aware (UTC)"
    assert not df[["open", "high", "low", "close", "volume"]].isna().any().any(), "NaNs present"

    # Spot-check: 5m bars over `days` should be close to days*288 with no large gaps
    diffs_min = df["timestamp"].diff().dt.total_seconds().div(60).dropna()
    most_common = diffs_min.mode().iloc[0]
    assert most_common == 5.0, f"Expected 5-minute spacing, got mode {most_common}"
    assert diffs_min.max() <= 60.0, f"Suspicious gap of {diffs_min.max()} minutes — Binance should be continuous"

    print(f"PASS test_download_recent_btc_5m -> {len(df)} rows, range {df['timestamp'].min()} -> {df['timestamp'].max()}")
    return df


def test_load_roundtrip() -> None:
    """After download, load() must reconstitute the same data from parquet."""
    df = load("BTCUSDT", "5m")
    assert not df.empty, "load() returned empty — download() must run first"
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].is_unique
    print(f"PASS test_load_roundtrip -> {len(df)} rows from parquet")


if __name__ == "__main__":
    test_download_recent_btc_5m()
    test_load_roundtrip()
    print("\nAll Binance loader tests passed.")
