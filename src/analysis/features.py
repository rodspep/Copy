"""Feature/validator engine: indicator + structure conditions at each UG signal.

Design:
  - Indicator SERIES are computed ONCE on the full OHLC frame. Every indicator in
    src.indicators is causal (value at bar i uses only bars ≤ i), so reading a
    series at the signal's bar index is lookahead-safe.
  - For a signal at time T we use the LAST CLOSED bar at-or-before T (asof), never
    a bar that hadn't closed when UG fired.
  - Each feature group is a small function (df, ind, i) -> dict, composed into one
    row per signal. Adding a hypothesis = adding a group.

The feature matrix (signals × conditions, + direction + geometry) is what the
downstream stats use to find which combination UG consistently acts on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import (
    ema, macd, adx, rsi, stochastic, atr, bollinger,
    swings, structure_breaks, fair_value_gaps, order_blocks, liquidity_sweeps,
)
from src.analysis.signals import Signal

# Defaults — overridable via build_feature_matrix(cfg=...).
DEFAULT_CFG = {
    "ema_periods": (20, 50, 100, 200),
    "slope_lookback": 5,        # bars to measure EMA/RSI slope
    "rsi_period": 14,
    "atr_period": 14,
    "swing_left": 5, "swing_right": 5,
    "recent_break_bars": 12,    # window for "recent BOS / liquidity sweep"
    "vol_ma": 20,
}


# ---------- as-of bar lookup (no lookahead) ----------
def asof_index(times: pd.Series, ts: pd.Timestamp) -> int | None:
    """Index of the last entry in `times` with value <= ts, or None if ts is
    before all. `times` must be sorted ascending. Generic: pass bar CLOSE times
    for a strict no-lookahead lookup (see closed_bar_index)."""
    pos = int(times.searchsorted(ts, side="right")) - 1
    return pos if pos >= 0 else None


def infer_bar_seconds(timestamps: pd.Series) -> float:
    """Bar duration = modal gap between consecutive bar-open times (robust to
    weekend/session gaps)."""
    diffs = timestamps.diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]                       # ignore duplicate-timestamp gaps
    if diffs.empty:
        raise ValueError("need >=2 distinct bar times to infer timeframe")
    return float(diffs.mode().iloc[0])


def closed_bar_index(timestamps: pd.Series, ts: pd.Timestamp,
                     bar_seconds: float) -> int | None:
    """Index of the last bar that had FULLY CLOSED at-or-before `ts`.

    Bars are stamped at their OPEN; bar [T, T+Δ) is only known at T+Δ. So the
    last usable bar is the latest with open_time + Δ <= ts. A signal firing
    mid-bar therefore reads the PREVIOUS (closed) bar — never the forming one.
    """
    close_times = timestamps + pd.Timedelta(seconds=bar_seconds)
    return asof_index(close_times, ts)


# ---------- indicator precompute (once per frame) ----------
def precompute(df: pd.DataFrame, cfg: dict) -> dict:
    """Compute all indicator series on the full frame. Returns a dict of Series."""
    close = df["close"]
    ind: dict = {}
    for n in cfg["ema_periods"]:
        ind[f"ema{n}"] = ema(close, n)
    ind["rsi"] = rsi(close, cfg["rsi_period"])
    ind["atr"] = atr(df, cfg["atr_period"])
    m = macd(close)
    ind["macd"], ind["macd_signal"], ind["macd_hist"] = m["macd"], m["signal"], m["hist"]
    a = adx(df)
    ind["adx"], ind["plus_di"], ind["minus_di"] = a["adx"], a["plus_di"], a["minus_di"]
    st = stochastic(df)
    ind["stoch_k"], ind["stoch_d"] = st["k"], st["d"]
    bb = bollinger(close)
    ind["bb_percent_b"], ind["bb_bandwidth"] = bb["percent_b"], bb["bandwidth"]
    sw = swings(df, left=cfg["swing_left"], right=cfg["swing_right"])
    ind["swing_high"], ind["swing_low"] = sw["swing_high_price"], sw["swing_low_price"]
    stx = structure_breaks(df, sw)
    ind["trend"], ind["bos"], ind["choch"] = stx["trend"], stx["bos"], stx["choch"]
    fv = fair_value_gaps(df)
    ind["bull_fvg_top"], ind["bull_fvg_bot"] = fv["bull_fvg_top"], fv["bull_fvg_bot"]
    ind["bear_fvg_top"], ind["bear_fvg_bot"] = fv["bear_fvg_top"], fv["bear_fvg_bot"]
    ob = order_blocks(df, stx)
    ind["bull_ob_top"], ind["bull_ob_bot"] = ob["bull_ob_top"], ob["bull_ob_bot"]
    ind["bear_ob_top"], ind["bear_ob_bot"] = ob["bear_ob_top"], ob["bear_ob_bot"]
    ls = liquidity_sweeps(df, sw)
    ind["bull_sweep"], ind["bear_sweep"] = ls["bull_sweep"], ls["bear_sweep"]
    ind["vol_ma"] = df["volume"].rolling(cfg["vol_ma"], min_periods=cfg["vol_ma"]).mean()
    return ind


# ---------- helpers ----------
def _val(s: pd.Series, i: int) -> float:
    """Scalar value at positional index i, NaN-safe → float or np.nan."""
    v = s.iloc[i]
    return float(v) if pd.notna(v) else np.nan


def _recent_any(s: pd.Series, i: int, lookback: int) -> bool:
    lo = max(0, i - lookback + 1)
    window = s.iloc[lo:i + 1]
    return bool(window.fillna(False).astype(bool).any())


def _gt(a: float, b: float) -> float:
    """a > b as 0/1, but NaN if either side is undefined (don't fabricate 0)."""
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return float(a > b)


# ---------- feature groups (each: df, ind, i, atr_i -> dict) ----------
def _f_trend(df, ind, i, atr_i, cfg) -> dict:
    close = _val(df["close"], i)
    out: dict = {}
    emas = {n: _val(ind[f"ema{n}"], i) for n in cfg["ema_periods"]}
    ns = list(cfg["ema_periods"])
    # fast-above-slow for each adjacent pair (trend stack)
    for a_, b_ in zip(ns, ns[1:]):
        out[f"ema{a_}_gt_ema{b_}"] = float(emas[a_] > emas[b_]) if not (np.isnan(emas[a_]) or np.isnan(emas[b_])) else np.nan
    fastest, slowest = ns[0], ns[-1]
    out["price_gt_ema_fastest"] = float(close > emas[fastest]) if not np.isnan(emas[fastest]) else np.nan
    out["price_gt_ema_slowest"] = float(close > emas[slowest]) if not np.isnan(emas[slowest]) else np.nan
    # slope of slowest EMA over lookback (in ATR units, sign + magnitude)
    k = cfg["slope_lookback"]
    if i - k >= 0 and atr_i and not np.isnan(atr_i):
        prev = _val(ind[f"ema{slowest}"], i - k)
        out["ema_slowest_slope_atr"] = (emas[slowest] - prev) / atr_i if not np.isnan(prev) else np.nan
    else:
        out["ema_slowest_slope_atr"] = np.nan
    out["adx"] = _val(ind["adx"], i)
    out["di_bullish"] = _gt(_val(ind["plus_di"], i), _val(ind["minus_di"], i))
    out["struct_trend"] = _val(ind["trend"], i)            # -1 / 0 / +1
    return out


def _f_momentum(df, ind, i, atr_i, cfg) -> dict:
    out = {"rsi": _val(ind["rsi"], i),
           "stoch_k": _val(ind["stoch_k"], i),
           "macd_hist": _val(ind["macd_hist"], i),
           "macd_above_signal": _gt(_val(ind["macd"], i), _val(ind["macd_signal"], i))}
    k = cfg["slope_lookback"]
    out["rsi_rising"] = _gt(_val(ind["rsi"], i), _val(ind["rsi"], i - k)) if i - k >= 0 else np.nan
    return out


def _f_volatility(df, ind, i, atr_i, cfg) -> dict:
    return {"atr": atr_i,
            "bb_percent_b": _val(ind["bb_percent_b"], i),
            "bb_bandwidth": _val(ind["bb_bandwidth"], i)}


def _f_structure(df, ind, i, atr_i, cfg) -> dict:
    close = _val(df["close"], i)
    out: dict = {}
    # distance to the LAST CONFIRMED swing high/low (carried forward by swings()),
    # not necessarily the nearest price level.
    sh, sl = _val(ind["swing_high"], i), _val(ind["swing_low"], i)
    if atr_i and not np.isnan(atr_i):
        out["dist_last_swing_low_atr"] = (close - sl) / atr_i if not np.isnan(sl) else np.nan
        out["dist_last_swing_high_atr"] = (sh - close) / atr_i if not np.isnan(sh) else np.nan
    else:
        out["dist_last_swing_low_atr"] = out["dist_last_swing_high_atr"] = np.nan
    lb = cfg["recent_break_bars"]
    out["recent_bos"] = float(_recent_any(ind["bos"], i, lb))
    out["recent_bull_sweep"] = float(_recent_any(ind["bull_sweep"], i, lb))
    out["recent_bear_sweep"] = float(_recent_any(ind["bear_sweep"], i, lb))
    # inside an active order block / FVG zone (price within the zone)
    out["in_bull_ob"] = _in_zone(close, _val(ind["bull_ob_bot"], i), _val(ind["bull_ob_top"], i))
    out["in_bear_ob"] = _in_zone(close, _val(ind["bear_ob_bot"], i), _val(ind["bear_ob_top"], i))
    out["in_bull_fvg"] = _in_zone(close, _val(ind["bull_fvg_bot"], i), _val(ind["bull_fvg_top"], i))
    out["in_bear_fvg"] = _in_zone(close, _val(ind["bear_fvg_bot"], i), _val(ind["bear_fvg_top"], i))
    return out


def _in_zone(price: float, bot: float, top: float) -> float:
    if np.isnan(bot) or np.isnan(top):
        return np.nan
    return float(bot <= price <= top)


def _f_candle(df, ind, i, atr_i, cfg) -> dict:
    o, h, l, c = (_val(df[col], i) for col in ("open", "high", "low", "close"))
    rng = h - l
    out = {"bull_candle": _gt(c, o)}
    if rng > 0:
        out["body_frac"] = abs(c - o) / rng
        out["upper_wick_frac"] = (h - max(o, c)) / rng
        out["lower_wick_frac"] = (min(o, c) - l) / rng
    else:
        out["body_frac"] = out["upper_wick_frac"] = out["lower_wick_frac"] = np.nan
    return out


def _f_volume(df, ind, i, atr_i, cfg) -> dict:
    vma = _val(ind["vol_ma"], i)
    vol = _val(df["volume"], i)
    return {"vol_vs_ma": vol / vma if vma and not np.isnan(vma) and vma > 0 else np.nan}


def _f_time(df, ind, i, atr_i, cfg) -> dict:
    ts = df["timestamp"].iloc[i]
    hour = int(ts.hour)                               # UTC
    # Rough FX sessions (UTC): Asia 0-7, London 7-12, overlap 12-16, NY 16-21.
    if hour < 7:
        sess = "asia"
    elif hour < 12:
        sess = "london"
    elif hour < 16:
        sess = "overlap"
    elif hour < 21:
        sess = "ny"
    else:
        sess = "late"
    return {"hour_utc": hour, "session": sess, "dow": int(ts.dayofweek)}


def _f_geometry(signal: Signal, df, ind, i, atr_i, cfg) -> dict:
    """UG's own entry/SL/TP geometry — directly tests Method A (fade, tiny R:R)
    vs Method B (deep pullback, wide R:R)."""
    out: dict = {}
    if not signal.has_geometry:
        return out
    risk = abs(signal.entry - signal.sl)
    reward = abs(signal.tp - signal.entry)
    out["rr"] = reward / risk if risk > 0 else np.nan
    if atr_i and not np.isnan(atr_i) and atr_i > 0:
        out["sl_dist_atr"] = risk / atr_i
        out["tp_dist_atr"] = reward / atr_i
        # how far the entry sits from the bar close, in ATR (limit vs market)
        out["entry_minus_close_atr"] = (signal.entry - _val(df["close"], i)) / atr_i
    return out


_GROUPS = (_f_trend, _f_momentum, _f_volatility, _f_structure, _f_candle, _f_volume, _f_time)


def compute_row(signal: Signal, df: pd.DataFrame, ind: dict, i: int, cfg: dict) -> dict:
    atr_i = _val(ind["atr"], i)
    row: dict = {
        "ts": df["timestamp"].iloc[i].isoformat(),
        "signal_ts": signal.ts.isoformat(),
        "direction": signal.direction,
        "dir_sign": 1 if signal.direction == "long" else -1,
        "close": _val(df["close"], i),
    }
    for g in _GROUPS:
        row.update(g(df, ind, i, atr_i, cfg))
    row.update(_f_geometry(signal, df, ind, i, atr_i, cfg))
    return row


def build_feature_matrix(signals: list[Signal], df: pd.DataFrame,
                         cfg: dict | None = None) -> pd.DataFrame:
    """One feature row per signal. Signals before the data start are dropped
    (reported via the 'skipped' attr on the returned frame)."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts_col = df["timestamp"]
    if ts_col.dt.tz is None:
        raise ValueError("df['timestamp'] must be tz-aware UTC")
    if str(ts_col.dt.tz) != "UTC":                 # convert so hour_utc et al. are UTC
        ts_col = ts_col.dt.tz_convert("UTC")
        df = df.assign(timestamp=ts_col)
    bar_seconds = float(cfg.get("bar_seconds") or infer_bar_seconds(ts_col))
    ind = precompute(df, cfg)
    rows, skipped = [], 0
    for s in signals:
        i = closed_bar_index(ts_col, s.ts, bar_seconds)   # strict: bar closed before signal
        if i is None:
            skipped += 1
            continue
        rows.append(compute_row(s, df, ind, i, cfg))
    out = pd.DataFrame(rows)
    out.attrs["skipped"] = skipped
    out.attrs["n_signals"] = len(signals)
    out.attrs["bar_seconds"] = bar_seconds
    return out
