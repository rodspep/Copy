"""Loader for TradingView-exported CSV (real broker XAUUSD spot).

TradingView Premium lets you export chart data to CSV ("Export chart data..."),
giving the ACTUAL broker feed (e.g. OANDA:XAUUSD) at any timeframe — fresh to
now, clean intrabar OHLC. This is the authoritative source for verifying UG
signals (vs. our PAXG/GC=F proxies).

Robust to TradingView's export variations:
  - time column named 'time' (ISO8601 like 2026-05-26T00:00:00Z, or unix seconds)
  - OHLC columns: open/high/low/close (any case)
  - optional volume

Usage:
  from src.data.tv_loader import load_tv_csv
  df = load_tv_csv('data/tv/OANDA_XAUUSD_5m.csv')
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_tv_csv(path: str | Path) -> pd.DataFrame:
    """Parse a TradingView CSV export → standard [timestamp, open, high, low, close, volume].

    timestamp is tz-aware UTC. Rows are de-duplicated and sorted ascending.
    """
    path = Path(path)
    df = pd.read_csv(path)
    # Normalize column names
    cols = {c.lower().strip(): c for c in df.columns}

    # Time column
    time_col = None
    for cand in ("time", "date", "datetime", "timestamp"):
        if cand in cols:
            time_col = cols[cand]
            break
    if time_col is None:
        time_col = df.columns[0]  # fall back to first column

    raw_t = df[time_col]
    # Detect unix seconds vs ISO string
    if pd.api.types.is_numeric_dtype(raw_t):
        unit = "s"
        # If values look like ms (13 digits), use ms
        if raw_t.dropna().iloc[0] > 1e12:
            unit = "ms"
        ts = pd.to_datetime(raw_t, unit=unit, utc=True)
    else:
        ts = pd.to_datetime(raw_t, utc=True)

    def pick(*names):
        for n in names:
            if n in cols:
                return df[cols[n]]
        raise ValueError(f"CSV missing any of columns {names}; has {list(df.columns)}")

    out = pd.DataFrame({
        "timestamp": ts,
        "open": pick("open").astype(float),
        "high": pick("high").astype(float),
        "low": pick("low").astype(float),
        "close": pick("close").astype(float),
    })
    # Volume optional
    for vname in ("volume", "vol"):
        if vname in cols:
            out["volume"] = pd.to_numeric(df[cols[vname]], errors="coerce").fillna(0.0)
            break
    else:
        out["volume"] = 0.0

    out = (out.dropna(subset=["close"])
              .drop_duplicates("timestamp", keep="last")
              .sort_values("timestamp")
              .reset_index(drop=True))
    return out


def validate_against_proxy(tv_df: pd.DataFrame, proxy_df: pd.DataFrame) -> dict:
    """Quick sanity: correlation + basis of a TradingView feed vs a proxy (PAXG/GC=F).

    Returns dict with overlap count, level corr, return corr, basis stats.
    """
    import numpy as np
    m = pd.merge(
        tv_df[["timestamp", "close"]].rename(columns={"close": "tv"}),
        proxy_df[["timestamp", "close"]].rename(columns={"close": "px"}),
        on="timestamp",
    )
    if len(m) < 30:
        return {"overlap": len(m), "note": "insufficient overlap to validate"}
    lvl = float(np.corrcoef(m["tv"], m["px"])[0, 1])
    r1, r2 = m["tv"].pct_change(), m["px"].pct_change()
    msk = r1.notna() & r2.notna()
    rc = float(np.corrcoef(r1[msk], r2[msk])[0, 1])
    basis = m["tv"] - m["px"]
    return {
        "overlap": len(m), "level_corr": round(lvl, 4), "return_corr": round(rc, 4),
        "basis_mean": round(float(basis.mean()), 2), "basis_std": round(float(basis.std()), 2),
    }
