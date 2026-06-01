"""UG-decoded strategy: S/R reversion with rejection-candle confirmation.

Reverse-engineered from the UG bot's signals + a real winning trade (see
docs/decisions/ug_logic_decode.md). The reproducible edge is NOT the multi-TF MA
narrative (post-hoc) but a simple price-action core:

  - Price pulls back to a recent swing level (support for longs, resistance for
    shorts) — UG posts an entry ZONE there.
  - Enter only on a REJECTION candle: the bar wicks into the level but closes back
    across it (close above support / below resistance). This confirmation is what
    flips the raw fade from breakeven to +EV (in-sample on MT5 M5).
  - Fixed SL a set distance beyond the level (UG: 10 price = 100 pip); TP a fixed
    distance from entry (UG scalp 50 pip / PRI-GOLD 150 pip).

Long AND short symmetric. M5, no HTF (an H1-trend filter added nothing in tests).
The engine enters at the next bar's OPEN (market) with spread/slippage, so SL/TP
are set as absolute prices off the level / confirmation close.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.indicators import swings
from src.strategies.base import Strategy, StrategyResult, empty_signals, validate_signals


class XauUgSrReversion(Strategy):
    ltf = "M5"
    required_htfs: tuple[str, ...] = ()

    default_params: dict[str, Any] = {
        "swing_left": 5,
        "swing_right": 5,
        "sl_price": 10.0,        # fixed SL distance beyond the level (UG: 100 pip)
        "tp_price": 15.0,        # TP distance from entry (PRI-GOLD 150 pip; scalp=5)
        "require_confirm": True,  # rejection close at the level (the edge)
    }

    def generate_signals(self, ltf, htfs, params=None) -> StrategyResult:
        p = self.merged_params(params)
        sw = swings(ltf, left=int(p["swing_left"]), right=int(p["swing_right"]))
        R = sw["swing_high_price"]                 # carried last confirmed swing high (resistance)
        S = sw["swing_low_price"]                  # carried last confirmed swing low (support)
        close, high, low = ltf["close"], ltf["high"], ltf["low"]
        prev_close = close.shift(1)

        # Tag: price came from the near side and reached the level this bar.
        long_tag = S.notna() & (prev_close > S) & (low <= S)
        short_tag = R.notna() & (prev_close < R) & (high >= R)
        if bool(p["require_confirm"]):             # rejection close back across the level
            long_mask = long_tag & (close > S)
            short_mask = short_tag & (close < R)
        else:
            long_mask, short_mask = long_tag, short_tag

        both = long_mask & short_mask              # ambiguous bar → take neither
        long_mask &= ~both
        short_mask &= ~both

        slp, tpp = float(p["sl_price"]), float(p["tp_price"])
        sigs = empty_signals(ltf)
        sigs.loc[long_mask, "action"] = "enter_long"
        sigs.loc[long_mask, "sl"] = (S - slp)[long_mask].to_numpy()
        sigs.loc[long_mask, "tp"] = (close + tpp)[long_mask].to_numpy()
        sigs.loc[short_mask, "action"] = "enter_short"
        sigs.loc[short_mask, "sl"] = (R + slp)[short_mask].to_numpy()
        sigs.loc[short_mask, "tp"] = (close - tpp)[short_mask].to_numpy()

        validate_signals(sigs, len(ltf))
        return StrategyResult(signals=sigs)
