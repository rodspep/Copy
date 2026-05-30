"""Volume-based / order-flow indicators.

Functions:
- `vwap()`                 — session-anchored VWAP with configurable σ bands (default 1σ/2σ/3σ)
- `obv()`                  — On-Balance Volume (signed cumulative volume by bar direction)
- `cvd()`                  — Cumulative Volume Delta (true aggressor-side if `taker_buy_base`
                             column is present; otherwise tick-rule proxy via bar direction)

Session anchors for VWAP:
- 'daily_utc'    — reset at 00:00 UTC every day (default for BTC, 24/7)
- 'daily_2200'   — reset at 22:00 UTC (CME Globex daily close; common XAU session)
- Custom anchors via callable returning a session-id Series

All functions are lookahead-safe: at bar i, only bars 0..i contribute. VWAP uses
`groupby().cumsum()` which is strictly causal in row order, with a monotonic-
timestamp guard to prevent any unsorted-input row reordering.
"""
from __future__ import annotations

from typing import Callable, Iterable, Literal, Sequence

import numpy as np
import pandas as pd


SessionAnchor = Literal["daily_utc", "daily_2200"]
PriceCol = Literal["close", "typical", "hlc3", "ohlc4"]


# -----------------------------------------------------------------------------
# Session id helpers
# -----------------------------------------------------------------------------

def _session_id_daily_utc(timestamps: pd.Series) -> pd.Series:
    """Session id = UTC date. Resets at 00:00 UTC."""
    return timestamps.dt.tz_convert("UTC").dt.normalize()


def _session_id_daily_2200(timestamps: pd.Series) -> pd.Series:
    """Session id resets at 22:00 UTC each day (CME Globex daily close).

    A timestamp t belongs to session = floor((t - 22h) to nearest day) + 22h.
    """
    ts = timestamps.dt.tz_convert("UTC")
    shifted = ts - pd.Timedelta(hours=22)
    return shifted.dt.normalize() + pd.Timedelta(hours=22)


def _resolve_anchor(anchor) -> Callable[[pd.Series], pd.Series]:
    if callable(anchor):
        return anchor
    if anchor == "daily_utc":
        return _session_id_daily_utc
    if anchor == "daily_2200":
        return _session_id_daily_2200
    raise ValueError(
        f"Unknown anchor {anchor!r}. Use 'daily_utc', 'daily_2200', or pass a callable."
    )


def _resolve_price(df: pd.DataFrame, price_col: PriceCol) -> pd.Series:
    if price_col in ("typical", "hlc3"):
        return (df["high"] + df["low"] + df["close"]) / 3.0
    if price_col == "ohlc4":
        if "open" not in df.columns:
            raise ValueError("ohlc4 requires column 'open'")
        return (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    if price_col == "close":
        return df["close"]
    raise ValueError(f"Unknown price_col {price_col!r}")


# -----------------------------------------------------------------------------
# VWAP with multiple σ bands
# -----------------------------------------------------------------------------

DEFAULT_VWAP_BANDS: tuple[float, ...] = (1.0, 2.0, 3.0)


def vwap(
    df: pd.DataFrame,
    anchor: SessionAnchor | Callable[[pd.Series], pd.Series] = "daily_utc",
    price_col: PriceCol = "typical",
    bands: Sequence[float] = DEFAULT_VWAP_BANDS,
) -> pd.DataFrame:
    """Session-anchored VWAP with configurable σ bands.

    Args:
        df         — OHLCV DataFrame with 'timestamp' (UTC tz-aware) + ['high','low','close','volume'].
        anchor     — 'daily_utc' (BTC, 24/7) or 'daily_2200' (XAU CME session), or a callable.
        price_col  — 'close', 'typical' / 'hlc3' (default), or 'ohlc4'.
        bands      — sequence of POSITIVE σ multipliers; default (1.0, 2.0, 3.0) for 1σ/2σ/3σ.
                     Each value k produces `vwap_upper_{k}` and `vwap_lower_{k}` columns.

    Returns DataFrame with columns:
        - 'vwap'               session-cumulative VWAP
        - 'vwap_std'           session-cumulative volume-weighted stdev of price
        - 'vwap_upper_{k}'     vwap + k * vwap_std  (one per k in `bands`)
        - 'vwap_lower_{k}'     vwap - k * vwap_std  (one per k in `bands`)

    Column naming uses `g` formatting so 1.0 → "1", 1.5 → "1.5". This keeps
    the common 1σ/2σ/3σ case readable while supporting half-σ bands if needed.
    """
    if "timestamp" not in df.columns:
        raise ValueError("vwap requires a 'timestamp' column")
    required = {"high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"vwap requires columns {required}")

    ts = df["timestamp"]
    if ts.dt.tz is None:
        raise ValueError("'timestamp' column must be tz-aware (UTC)")
    if not ts.is_monotonic_increasing:
        raise ValueError("vwap requires timestamps sorted ascending")

    bands = tuple(float(b) for b in bands)
    if not bands:
        raise ValueError("bands must be non-empty")
    if any(b <= 0 for b in bands):
        raise ValueError(f"all bands must be > 0, got {bands}")

    price = _resolve_price(df, price_col)
    vol = df["volume"].astype("float64")
    pv = price * vol
    p2v = (price * price) * vol

    session_id = _resolve_anchor(anchor)(ts)
    grouper = session_id.values

    # Per-session causal cumulative sums.
    g = pd.DataFrame({"pv": pv, "p2v": p2v, "v": vol}).groupby(grouper, sort=False)
    cum_pv = g["pv"].cumsum()
    cum_p2v = g["p2v"].cumsum()
    cum_v = g["v"].cumsum()

    safe_v = cum_v.where(cum_v > 0)
    vwap_line = cum_pv / safe_v

    mean_p = vwap_line
    mean_p2 = cum_p2v / safe_v
    var = (mean_p2 - mean_p * mean_p).clip(lower=0.0)  # clip tiny negatives from float error
    std = np.sqrt(var)

    out_cols = {"vwap": vwap_line, "vwap_std": std}
    for k in bands:
        label = f"{k:g}"  # 1.0 → '1', 2.5 → '2.5'
        out_cols[f"vwap_upper_{label}"] = vwap_line + k * std
        out_cols[f"vwap_lower_{label}"] = vwap_line - k * std
    return pd.DataFrame(out_cols, index=df.index)


# -----------------------------------------------------------------------------
# Order flow: OBV
# -----------------------------------------------------------------------------

def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (Granville).

    OBV_t = OBV_{t-1} + sign(close_t - close_{t-1}) * volume_t
    OBV_0 = 0 (convention; the first bar has no previous close).

    Lookahead-safe: uses `close.diff()` (causal) and `cumsum()` (row-order causal).
    Requires monotonic timestamps if 'timestamp' column exists; otherwise relies on
    caller to have sorted.
    """
    if "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("obv requires columns ['close','volume']")
    if "timestamp" in df.columns:
        if df["timestamp"].dt.tz is None:
            raise ValueError("'timestamp' must be tz-aware")
        if not df["timestamp"].is_monotonic_increasing:
            raise ValueError("obv requires timestamps sorted ascending")

    signed = np.sign(df["close"].diff().fillna(0.0)) * df["volume"].astype("float64")
    return signed.cumsum().rename("obv")


# -----------------------------------------------------------------------------
# Order flow: CVD (Cumulative Volume Delta)
# -----------------------------------------------------------------------------

def cvd(
    df: pd.DataFrame,
    anchor: SessionAnchor | Callable[[pd.Series], pd.Series] | None = None,
) -> pd.Series:
    """Cumulative Volume Delta.

    If the DataFrame has a `taker_buy_base` column (Binance kline data preserves
    aggressor-side buy volume), CVD uses TRUE aggressor delta:
        delta_t = taker_buy_base_t - (volume_t - taker_buy_base_t)
                = 2 * taker_buy_base_t - volume_t

    Otherwise falls back to a bar-direction tick-rule proxy:
        delta_t = sign(close_t - open_t) * volume_t

    Args:
        anchor: if provided, CVD resets per session (same semantics as VWAP).
                None → strictly cumulative across the full DataFrame.

    Lookahead-safe: same constraints as OBV.
    """
    if "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("cvd requires columns ['close','volume']")
    if "timestamp" in df.columns:
        if df["timestamp"].dt.tz is None:
            raise ValueError("'timestamp' must be tz-aware")
        if not df["timestamp"].is_monotonic_increasing:
            raise ValueError("cvd requires timestamps sorted ascending")

    vol = df["volume"].astype("float64")
    if "taker_buy_base" in df.columns:
        taker_buy = df["taker_buy_base"].astype("float64")
        delta = 2.0 * taker_buy - vol
    else:
        # Bar-direction proxy. Doji bars (open == close) contribute zero — neutral.
        if "open" not in df.columns:
            raise ValueError("cvd fallback requires column 'open' when taker_buy_base is absent")
        bar_sign = np.sign(df["close"] - df["open"])
        delta = bar_sign * vol

    if anchor is None:
        cum = delta.cumsum()
    else:
        if "timestamp" not in df.columns:
            raise ValueError("session-anchored cvd requires a 'timestamp' column")
        session_id = _resolve_anchor(anchor)(df["timestamp"])
        cum = pd.Series(delta).groupby(session_id.values, sort=False).cumsum()
    return cum.rename("cvd")
