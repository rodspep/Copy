"""
Smoke + sanity tests for the Dukascopy XAU/USD loader.

Run with: python -m tests.data.test_dukascopy_loader
or:       python -m pytest tests/data/test_dukascopy_loader.py
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.data.dukascopy_loader import download, load


# Two consecutive XAU weekdays in a recent past month. Avoid weekends/holidays.
SMOKE_START = datetime(2026, 4, 21, tzinfo=timezone.utc)
SMOKE_END = datetime(2026, 4, 23, tzinfo=timezone.utc)
SMOKE_SYMBOL = "XAUUSD"


def test_download_xau_two_weekdays() -> pd.DataFrame:
    """Download two weekdays of XAU ticks and verify M1 invariants."""
    m1 = download(SMOKE_SYMBOL, SMOKE_START, SMOKE_END)

    assert not m1.empty, "Download returned empty M1 frame"
    assert m1["timestamp"].is_monotonic_increasing, "M1 timestamps not sorted"
    assert str(m1["timestamp"].dt.tz) == "UTC", f"tz != UTC: {m1['timestamp'].dt.tz}"
    assert not m1[["open", "high", "low", "close", "volume"]].isna().any().any(), "NaNs in M1 bars"

    # OHLC integrity: high >= max(open, close), low <= min(open, close)
    assert (m1["high"] >= m1[["open", "close"]].max(axis=1) - 1e-9).all(), "high < max(open, close)"
    assert (m1["low"] <= m1[["open", "close"]].min(axis=1) + 1e-9).all(), "low > min(open, close)"

    # Plausibility for XAU/USD (rough bound — 1000-5000 USD per oz covers any realistic regime).
    assert 1000.0 < m1["low"].min() < 5000.0, f"XAU low out of plausible range: {m1['low'].min()}"
    assert 1000.0 < m1["high"].max() < 5000.0, f"XAU high out of plausible range: {m1['high'].max()}"

    print(
        f"PASS test_download_xau_two_weekdays -> {len(m1)} M1 rows, "
        f"range {m1['timestamp'].min()} -> {m1['timestamp'].max()}, "
        f"price [{m1['low'].min():.3f}, {m1['high'].max():.3f}]"
    )
    return m1


def test_resampled_timeframes_present() -> None:
    """After download, every configured timeframe must be loadable from parquet."""
    counts: dict[str, int] = {}
    for interval in ("M1", "M5", "M15", "H1", "H4"):
        df = load(SMOKE_SYMBOL, interval)
        assert not df.empty, f"{interval} parquet missing — download() must run first"
        assert df["timestamp"].is_monotonic_increasing, f"{interval} timestamps not sorted"
        assert df["timestamp"].is_unique, f"{interval} has duplicate timestamps"
        counts[interval] = len(df)

    # Higher timeframes must have fewer (or equal) bars than lower ones.
    assert counts["M1"] >= counts["M5"] >= counts["M15"] >= counts["H1"] >= counts["H4"], (
        f"Bar counts not monotonically decreasing across TFs: {counts}"
    )
    print(f"PASS test_resampled_timeframes_present -> {counts}")


if __name__ == "__main__":
    test_download_xau_two_weekdays()
    test_resampled_timeframes_present()
    print("\nAll Dukascopy loader tests passed.")
