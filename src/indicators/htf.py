"""Higher-timeframe alignment helper.

Per parity ADR §5, LTF↔HTF joins MUST use the HTF bar's `available_at` timestamp
(= bar open + bar duration), not the bar's open timestamp. Merging on bar open
leaks the still-forming HTF bar into LTF signals — the classic MTF lookahead bug.

This module is the ONLY place in the codebase that should do LTF↔HTF joining.
Strategies must call `align_htf_to_ltf()` rather than rolling their own merge_asof.
"""
from __future__ import annotations

import pandas as pd

# Bar-duration map. Keep in sync with src/config.py::TIMEFRAMES.
TF_DURATION = {
    "M1": pd.Timedelta(minutes=1),
    "M5": pd.Timedelta(minutes=5),
    "M15": pd.Timedelta(minutes=15),
    "M30": pd.Timedelta(minutes=30),
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
}


def _duration(tf: str) -> pd.Timedelta:
    if tf not in TF_DURATION:
        raise ValueError(f"Unknown timeframe {tf}. Known: {list(TF_DURATION)}")
    return TF_DURATION[tf]


def align_htf_to_ltf(
    ltf: pd.DataFrame,
    htf: pd.DataFrame,
    ltf_tf: str,
    htf_tf: str,
    htf_cols: list[str] | None = None,
    suffix: str | None = None,
) -> pd.DataFrame:
    """Attach HTF columns onto the LTF frame using availability-time alignment.

    Inputs:
      ltf        — LTF DataFrame with a tz-aware 'timestamp' column. Must be sorted.
      htf        — HTF DataFrame with a tz-aware 'timestamp' column. Must be sorted.
      ltf_tf     — e.g. 'M5' (the LTF that signals will be generated on).
      htf_tf     — e.g. 'H1' (the HTF that provides trend/context). htf must be a
                   strictly higher timeframe than ltf.
      htf_cols   — list of HTF columns to attach. Default: all non-timestamp columns.
      suffix     — string appended to attached column names. Default: f'_{htf_tf}'.

    Output:
      LTF DataFrame with HTF columns added, suffixed. For each LTF row at time T_l,
      the attached HTF value is the **most recently AVAILABLE** HTF bar, i.e. the
      HTF bar with `available_at <= signal_available_at = T_l + Δ_l`.

    The result aligns with the strategy contract (parity ADR §2, §5): a signal
    evaluated at LTF bar `i` close (= T_l + Δ_l) sees HTF context as of the most
    recently fully closed HTF bar at that moment, and never the in-progress one.
    """
    if "timestamp" not in ltf.columns or "timestamp" not in htf.columns:
        raise ValueError("Both ltf and htf must have a 'timestamp' column")
    if ltf["timestamp"].dt.tz is None or htf["timestamp"].dt.tz is None:
        raise ValueError("timestamps must be tz-aware")
    if not ltf["timestamp"].is_monotonic_increasing:
        raise ValueError("ltf timestamps must be sorted ascending")
    if not htf["timestamp"].is_monotonic_increasing:
        raise ValueError("htf timestamps must be sorted ascending")
    if _duration(htf_tf) <= _duration(ltf_tf):
        raise ValueError(f"htf_tf ({htf_tf}) must be strictly greater than ltf_tf ({ltf_tf})")

    if htf_cols is None:
        htf_cols = [c for c in htf.columns if c != "timestamp"]
    if suffix is None:
        suffix = f"_{htf_tf}"

    ltf_dur = _duration(ltf_tf)
    htf_dur = _duration(htf_tf)

    # Use POSITIONAL row tracking so we don't rely on the original LTF index being
    # 0..N-1 or even monotonic.
    left = ltf[["timestamp"]].copy()
    left["_row_pos"] = range(len(left))
    left["signal_available_at"] = left["timestamp"] + ltf_dur
    left = left.sort_values("signal_available_at").reset_index(drop=True)

    # available_at: when the HTF bar's close becomes observable.
    right = htf[["timestamp", *htf_cols]].copy()
    right["available_at"] = right["timestamp"] + htf_dur
    right = right.sort_values("available_at").reset_index(drop=True)
    # Rename HTF columns with suffix BEFORE merge to avoid collisions and to
    # produce a self-documenting output.
    rename_map = {c: f"{c}{suffix}" for c in htf_cols}
    right = right.rename(columns=rename_map)

    merged = pd.merge_asof(
        left,
        right.drop(columns=["timestamp"]),
        left_on="signal_available_at",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )

    # Restore original LTF positional order.
    merged = merged.sort_values("_row_pos")

    out = ltf.copy()
    for c in rename_map.values():
        out[c] = merged[c].to_numpy()
    return out
