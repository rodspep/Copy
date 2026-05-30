"""Strategy base class.

A Strategy is a pure function from (LTF OHLCV, HTF context, parameters) to a
signals DataFrame consumable by `src.backtest.run_backtest`. The schema and
semantic constraints below come from parity ADR §6.

Contract every concrete strategy MUST satisfy:

1. **Determinism**: Given the same inputs, `generate_signals(...)` must return the
   same DataFrame every call. No random state, no wall-clock, no filesystem.
2. **No lookahead**: The signal emitted for LTF bar `i` may consult only
   `ltf[0..i]` and HTF context aligned via `align_htf_to_ltf` (which itself
   enforces availability-time alignment — see ADR §5).
3. **Output schema**: A DataFrame with the same length as `ltf`, columns:
       - 'action' : 'enter_long' | 'enter_short' | 'exit' | 'hold'
       - 'sl'     : float (required for enter_*; NaN otherwise)
       - 'tp'     : float (required for enter_*; NaN otherwise)
4. **Long AND short symmetry**: Strategies must consider symmetric signals on
   both sides; long-only strategies require justification in their docstring
   (per [[feedback-backtest-scope]]).
5. **Per-symbol tuning**: Parameter spaces are tuned per (strategy × symbol) by
   the optimizer; the strategy class itself is symbol-agnostic.

Live-adapter authors implementing the same strategy in streaming mode MUST
produce bit-identical signals at every closed bar. Reuse this class's method
directly in live; do NOT reimplement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


VALID_ACTIONS = ("enter_long", "enter_short", "exit", "hold")


@dataclass
class StrategyResult:
    """Container for `generate_signals` output + any optional debug diagnostics.

    Engines only consume `signals`; `debug` is for offline notebooks / charting.
    Strategies need not populate `debug`.
    """

    signals: pd.DataFrame
    debug: dict[str, pd.Series] | None = None


class Strategy(ABC):
    """Abstract strategy base.

    Subclasses implement `generate_signals(ltf, htfs, params) -> StrategyResult`.
    `htfs` is a dict {tf_name: DataFrame} so strategies can request multiple HTFs
    (e.g. H1 for trend, H4 for higher-level structure).
    """

    #: Default parameters; per-symbol overrides come from configs/strategies/<name>_<symbol>.yaml.
    default_params: dict[str, Any] = {}

    #: Required HTFs (timeframe names) that the engine harness must provide.
    required_htfs: tuple[str, ...] = ()

    #: LTF timeframe this strategy was designed for (e.g. "M5").
    ltf: str = "M5"

    @abstractmethod
    def generate_signals(
        self,
        ltf: pd.DataFrame,
        htfs: dict[str, pd.DataFrame],
        params: dict[str, Any] | None = None,
    ) -> StrategyResult:
        """Produce a signals DataFrame conforming to the contract above.

        Args:
            ltf      — OHLCV at the strategy's LTF.
            htfs     — dict of HTF OHLCVs, e.g. {'H1': df_h1, 'H4': df_h4}.
                       Every timeframe in `required_htfs` must be present.
            params   — overrides default_params. May be None.

        Returns StrategyResult.
        """
        raise NotImplementedError

    def merged_params(self, params: dict[str, Any] | None) -> dict[str, Any]:
        """Return defaults overridden by `params` (shallow merge, no mutation)."""
        if not params:
            return dict(self.default_params)
        return {**self.default_params, **params}


def empty_signals(ltf: pd.DataFrame) -> pd.DataFrame:
    """Return a signals DataFrame of length len(ltf) filled with 'hold'.

    Useful as the starting point for vectorized strategy construction:
        sigs = empty_signals(ltf)
        sigs.loc[long_mask, 'action'] = 'enter_long'
        sigs.loc[long_mask, 'sl'] = ...
    """
    n = len(ltf)
    return pd.DataFrame({
        "action": np.array(["hold"] * n, dtype=object),
        "sl": np.full(n, np.nan, dtype="float64"),
        "tp": np.full(n, np.nan, dtype="float64"),
    })


def validate_signals(signals: pd.DataFrame, ltf_len: int) -> None:
    """Sanity-check a signals DataFrame against the contract.

    Raises ValueError on schema violations. Engines validate too, but strategy
    authors get faster feedback if they call this from within `generate_signals`.
    """
    if len(signals) != ltf_len:
        raise ValueError(f"signals length {len(signals)} != ltf length {ltf_len}")
    required = {"action", "sl", "tp"}
    if not required.issubset(signals.columns):
        raise ValueError(f"signals missing columns: {required - set(signals.columns)}")
    bad = set(signals["action"].unique()) - set(VALID_ACTIONS)
    if bad:
        raise ValueError(f"signals contains invalid actions: {bad}")
    # For enter_* rows, sl and tp must be finite.
    enter_mask = signals["action"].isin(["enter_long", "enter_short"])
    if enter_mask.any():
        if not np.isfinite(signals.loc[enter_mask, "sl"]).all():
            raise ValueError("non-finite SL on enter_* row(s)")
        if not np.isfinite(signals.loc[enter_mask, "tp"]).all():
            raise ValueError("non-finite TP on enter_* row(s)")
