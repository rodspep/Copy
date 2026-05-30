"""XAU higher-timeframe trend-following (H4) — ride the trend, exit on reversal.

Rationale: all 18 M5 price-pattern strategies converged on a small, regime-
dependent edge (~34-40% positive walk-forward windows). The ONE thing that
genuinely worked (deep_pullback over the last year) was simply riding gold's
strong uptrend. So test that directly and cleanly on a higher timeframe where
spread/noise/HistData-intrabar issues matter far less.

Method (classic trend-following, stop-and-reverse):
  - Compute trend state on H4: EMA(fast) vs EMA(slow), OR Donchian breakout.
  - Enter in the trend direction. HOLD until the trend flips (no fixed TP — let
    winners run). Exit on reversal, then re-enter the other way.
  - A wide ATR 'disaster' stop bounds risk and is used for position sizing.

Unlike the scalp strategies, this makes few trades and aims to capture large
directional moves — the only thing that reliably pays in a trending market.

No-lookahead: trend state at bar i uses only closed bars ≤ i. Entries/exits are
emitted at the bar where the (confirmed) trend changes; the engine fills at the
next bar's open per the parity ADR.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators import ema, atr
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


class XauTrendFollow(Strategy):
    """H4 EMA/Donchian trend-following, stop-and-reverse, exit-on-reversal."""

    ltf = "H4"
    required_htfs: tuple[str, ...] = ()

    default_params: dict[str, Any] = {
        "entry_mode":   "ema_cross",   # 'ema_cross' | 'donchian'
        "ema_fast":     20,
        "ema_slow":     50,
        "donchian_n":   20,
        "atr_period":   14,
        "sl_atr_mult":  3.0,           # wide disaster stop (sizing anchor)
        "allow_short":  True,
        "min_hold_bars": 0,            # optional: ignore flips within N bars (anti-whipsaw)
    }

    def _trend_state(self, ltf: pd.DataFrame, p: dict) -> np.ndarray:
        """Causal trend state per bar: +1 up, -1 down, 0 undefined."""
        mode = str(p["entry_mode"]).lower()
        n = len(ltf)
        state = np.zeros(n, dtype=np.int8)
        if mode == "ema_cross":
            f = ema(ltf["close"], int(p["ema_fast"])).to_numpy()
            s = ema(ltf["close"], int(p["ema_slow"])).to_numpy()
            up = f > s
            dn = f < s
            valid = ~(np.isnan(f) | np.isnan(s))
            state[up & valid] = 1
            state[dn & valid] = -1
        elif mode == "donchian":
            nN = int(p["donchian_n"])
            high = ltf["high"].to_numpy()
            low = ltf["low"].to_numpy()
            close = ltf["close"].to_numpy()
            # Channel uses bars [i-nN .. i-1] (exclude current → causal breakout).
            cur = 0
            for i in range(n):
                if i < nN:
                    state[i] = 0
                    continue
                hh = high[i - nN:i].max()
                ll = low[i - nN:i].min()
                if close[i] > hh:
                    cur = 1
                elif close[i] < ll:
                    cur = -1
                state[i] = cur
        else:
            raise ValueError(f"unknown entry_mode {mode!r}")
        return state

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        p = self.merged_params(params)
        a = atr(ltf, int(p["atr_period"])).to_numpy()
        close = ltf["close"].to_numpy()
        n = len(ltf)
        state = self._trend_state(ltf, p)

        allow_short = bool(p["allow_short"])
        sl_mult = float(p["sl_atr_mult"])
        min_hold = int(p["min_hold_bars"])

        sigs = empty_signals(ltf)
        action = sigs["action"].to_numpy(dtype=object).copy()
        sl_arr = sigs["sl"].to_numpy().copy()
        tp_arr = sigs["tp"].to_numpy().copy()

        notional = 0          # current notional position direction
        bars_in_pos = 0
        for i in range(n):
            d = int(state[i])
            if not allow_short and d == -1:
                d = 0          # short disabled → treat down-trend as 'flat'
            if np.isnan(a[i]) or a[i] <= 0:
                if notional != 0:
                    bars_in_pos += 1
                continue

            if notional == 0:
                # enter when a trend exists
                if d != 0:
                    action[i] = "enter_long" if d == 1 else "enter_short"
                    sl_dist = sl_mult * a[i]
                    if d == 1:
                        sl_arr[i] = close[i] - sl_dist
                        tp_arr[i] = close[i] + 100.0 * sl_dist   # effectively no TP cap
                    else:
                        sl_arr[i] = close[i] + sl_dist
                        tp_arr[i] = close[i] - 100.0 * sl_dist
                    notional = d
                    bars_in_pos = 0
            else:
                bars_in_pos += 1
                # exit only if trend flipped AND we've held min_hold bars
                if d != notional and bars_in_pos >= min_hold:
                    action[i] = "exit"
                    notional = 0
                    bars_in_pos = 0

        sigs["action"] = action
        sigs["sl"] = sl_arr
        sigs["tp"] = tp_arr
        validate_signals(sigs, len(ltf))
        return StrategyResult(signals=sigs, debug={"trend": pd.Series(state, index=ltf.index)})
