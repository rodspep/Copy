"""Live OHLCV from a MetaTrader 5 terminal (Exness) — real broker prices.

Windows-only (`MetaTrader5` lib + running terminal). Imported lazily so the rest
of the codebase still works on Linux / without MT5. When the bot runs on the VPS
with DATA_SOURCE=mt5, bars come straight from the broker — no PAXG proxy, no
price offset.

Bar times are the terminal's server time as a Unix epoch; treated as UTC and
used consistently for signal timestamps, dedup, and outcome resolution.
"""
from __future__ import annotations

import os

import pandas as pd

_INITED = False


def _mt5():
    import MetaTrader5 as mt5          # lazy, Windows-only
    return mt5


def init(account: int | None = None, password: str = "", server: str = "") -> None:
    """Initialize + (optionally) log in to the terminal. Idempotent."""
    global _INITED
    if _INITED:
        return
    mt5 = _mt5()
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")
    account = account or (int(os.environ["MT5_ACCOUNT"])
                          if os.environ.get("MT5_ACCOUNT") else None)
    if account:
        password = password or os.environ.get("MT5_PASSWORD", "")
        server = server or os.environ.get("MT5_SERVER", "")
        if not mt5.login(int(account), password=password, server=server):
            raise RuntimeError(f"mt5.login failed: {mt5.last_error()}")
    _INITED = True


def _tf_const(tf: str):
    mt5 = _mt5()
    m = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
         "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}
    if tf not in m:
        raise ValueError(f"unsupported MT5 timeframe {tf}")
    return m[tf]


def bars(symbol: str, tf: str, n: int = 3000) -> pd.DataFrame:
    """Return the last `n` bars as [timestamp, open, high, low, close, volume]."""
    mt5 = _mt5()
    init()
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"symbol_select({symbol}) failed: {mt5.last_error()}")
    rates = mt5.copy_rates_from_pos(symbol, _tf_const(tf), 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"no rates for {symbol} {tf}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    vol = "real_volume" if df.get("real_volume", pd.Series([0])).astype(float).sum() > 0 else "tick_volume"
    out = df[["timestamp", "open", "high", "low", "close", vol]].copy()
    out = out.rename(columns={vol: "volume"})
    return out.reset_index(drop=True)
