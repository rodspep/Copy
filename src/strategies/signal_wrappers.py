"""Pure post-processing wrappers on raw signal DataFrames.

These convert a "market-on-close → fill at next open" signal stream into
alternative execution semantics, WITHOUT touching the backtest engine. The
engine still sees a standard ('action','sl','tp') signals frame; the wrapper
just rewrites WHEN actions fire and WHERE SL/TP land.

Pattern: every wrapper is `(ltf, sigs, **config) -> sigs_new` — pure, no
hidden state across calls. Engines and tests can compose them freely.

Two wrappers implemented:

  limit_entry_wrapper(ltf, sigs, offset_atr, expire_bars, atr_period=14)
      Convert market entries into pending-limit entries:
        - At bar i where `sigs.action == enter_long`, plan a long limit at
          `close_i - offset_atr * ATR(i)`.
        - Watch bars i+1 .. i+expire_bars. If any bar's LOW <= limit_price,
          THAT bar's signal becomes 'enter_long' with the limit_price as the
          implicit fill anchor (engine still uses bar.open for fill, but our
          'enter' is deferred to the fill bar). SL/TP carry forward unchanged.
        - If no bar hits the limit within window → drop the signal entirely
          (the false-breakout filter UG-style relies on).
      Mirror for shorts (limit ABOVE close, watch HIGH).

  partial_tp_wrapper(ltf, sigs, tp1_frac, atr_period=14)
      Encode a "TP1 at fraction × (entry → final TP) distance, close 50%, move
      SL to BE" exit by SHRINKING the position's effective TP and SL to the
      tighter TP1 level. The current engine doesn't support partial close, so
      this wrapper is an upper-bound approximation: it replaces the original
      TP with TP1 (close 100% at TP1). Use it to bound the value of partial
      exits without an engine change. NOT exact partial-close math.

All wrappers preserve the contract:
  - len(sigs_new) == len(sigs) == len(ltf)
  - actions ∈ {enter_long, enter_short, exit, hold}
  - SL/TP finite on enter rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import atr, ema
from src.strategies.base import empty_signals


def limit_entry_wrapper(
    ltf: pd.DataFrame,
    sigs: pd.DataFrame,
    *,
    offset_atr: float = 0.3,
    expire_bars: int = 5,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Convert market entries → pending-limit entries with expire window.

    Args:
        ltf:          OHLCV.
        sigs:         Original signals DataFrame.
        offset_atr:   Limit price offset from the trigger bar's close, in ATRs.
                      For longs: limit_price = close - offset_atr * ATR.
                      For shorts: limit_price = close + offset_atr * ATR.
                      offset_atr > 0 expects a pullback before filling.
        expire_bars:  Watch this many bars (exclusive of trigger bar) for the
                      limit to fill. If unfilled, the signal is dropped.
        atr_period:   ATR period.

    Returns:
        New signals DataFrame:
        - The trigger bar's action is set to 'hold' (we don't fill there).
        - The bar where the limit price is hit gets the enter_* action.
        - SL/TP from the original trigger bar carry forward, unchanged.
        - If never filled, no enter event is emitted.
    """
    if len(sigs) != len(ltf):
        raise ValueError(f"sigs length {len(sigs)} != ltf length {len(ltf)}")
    n = len(ltf)
    a = atr(ltf, atr_period).to_numpy()
    close = ltf["close"].to_numpy()
    high = ltf["high"].to_numpy()
    low = ltf["low"].to_numpy()
    action = sigs["action"].to_numpy()
    sl_arr = sigs["sl"].to_numpy()
    tp_arr = sigs["tp"].to_numpy()

    out = empty_signals(ltf)

    for i in range(n):
        act = action[i]
        if act not in ("enter_long", "enter_short"):
            # Pass through 'exit' actions unchanged; 'hold' is already the default.
            if act == "exit":
                out.at[i, "action"] = "exit"
            continue

        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        if not (np.isfinite(sl_arr[i]) and np.isfinite(tp_arr[i])):
            continue

        side = 1 if act == "enter_long" else -1
        limit_price = close[i] - side * float(offset_atr) * a[i]

        # Watch the next `expire_bars` bars (i+1 .. i+expire_bars).
        end = min(n, i + 1 + int(expire_bars))
        fill_idx = -1
        for j in range(i + 1, end):
            if side == 1 and low[j] <= limit_price:
                fill_idx = j
                break
            if side == -1 and high[j] >= limit_price:
                fill_idx = j
                break

        if fill_idx < 0:
            continue  # expired, drop signal

        # Skip if an entry already exists at fill_idx (collision); first-come wins.
        if out.at[fill_idx, "action"] != "hold":
            continue

        out.at[fill_idx, "action"] = act
        out.at[fill_idx, "sl"] = float(sl_arr[i])
        out.at[fill_idx, "tp"] = float(tp_arr[i])

    return out


def dxy_block_opposite_wrapper(
    ltf: pd.DataFrame,
    sigs: pd.DataFrame,
    dxy_daily: pd.DataFrame,
    *,
    slope_window: int = 5,
    block_threshold_pct: float = 0.5,
) -> pd.DataFrame:
    """Block XAU signals when DXY trend strongly opposes the trade direction.

    XAU and DXY are strongly inverse-correlated (~-0.7 to -0.9). When DXY rallies
    sharply, XAU long signals usually fail; when DXY drops sharply, XAU short
    signals usually fail. This wrapper drops only the STRONGLY opposite signals
    — keep neutral ones (Codex's "block-opposite, not require-align" advice).

    Args:
        ltf:                 XAU M5 OHLCV with timestamp.
        sigs:                Original signal DataFrame.
        dxy_daily:           DXY daily DataFrame from yahoo_loader.
        slope_window:        N days for DXY slope (e.g. 5 = ~1 trading week).
        block_threshold_pct: Drop signal if abs(DXY slope %) > this AND opposite
                             direction (e.g. 0.5% over 5 days = trending DXY).

    Returns:
        New signals DataFrame with strongly-opposite entries set to 'hold'.
    """
    from src.data.yahoo_loader import align_daily_to_ltf

    d = dxy_daily.copy().sort_values("timestamp").reset_index(drop=True)
    # Slope = pct change over N days
    d["__close_n_ago"] = d["close"].shift(slope_window)
    d["__slope_pct"] = (d["close"] / d["__close_n_ago"] - 1) * 100.0
    # Align onto LTF (forward-fill last-known-closed daily slope)
    aligned = align_daily_to_ltf(ltf, d, ["__slope_pct"])
    slope = aligned["__slope_pct"]

    out = sigs.copy()
    action = out["action"].to_numpy()
    # Long blocked when DXY slope > threshold (DXY rallying = USD strong = bad for XAU long)
    # Short blocked when DXY slope < -threshold (DXY dropping = USD weak = bad for XAU short)
    block_long = (action == "enter_long") & (slope > block_threshold_pct).to_numpy()
    block_short = (action == "enter_short") & (slope < -block_threshold_pct).to_numpy()
    blocked = block_long | block_short

    out.loc[blocked, "action"] = "hold"
    out.loc[blocked, "sl"] = float("nan")
    out.loc[blocked, "tp"] = float("nan")
    return out


def partial_tp_wrapper(
    ltf: pd.DataFrame,
    sigs: pd.DataFrame,
    *,
    tp1_frac: float = 0.5,
) -> pd.DataFrame:
    """Approximate partial-TP-at-TP1 by replacing TP with TP1.

    Useful for bounding the value of a "close 50% at TP1, runner to final TP"
    exit without an engine change. This implementation closes 100% at TP1
    (an upper bound on WR, lower bound on per-trade R).

    Args:
        ltf:       OHLCV.
        sigs:      Original signals.
        tp1_frac:  TP1 distance as a fraction of the original (entry → TP) gap.
                   tp1_frac=0.5 means TP1 sits halfway between entry-anchor and
                   the strategy's TP.

    The strategy's reference "entry anchor" is the close of the trigger bar
    (matches what the engine sees for fill anchoring before slippage).
    """
    if len(sigs) != len(ltf):
        raise ValueError("sigs length must equal ltf length")
    close = ltf["close"].to_numpy()
    out = empty_signals(ltf)
    out["action"] = sigs["action"].to_numpy()
    out["sl"] = sigs["sl"].to_numpy()
    out["tp"] = sigs["tp"].to_numpy()

    mask_long = sigs["action"].to_numpy() == "enter_long"
    mask_short = sigs["action"].to_numpy() == "enter_short"
    tp = out["tp"].to_numpy()

    # Long: TP1 = close + tp1_frac * (TP - close), but TP > close already.
    new_tp = tp.copy()
    if mask_long.any():
        new_tp[mask_long] = close[mask_long] + float(tp1_frac) * (tp[mask_long] - close[mask_long])
    if mask_short.any():
        new_tp[mask_short] = close[mask_short] - float(tp1_frac) * (close[mask_short] - tp[mask_short])

    out["tp"] = new_tp
    return out
