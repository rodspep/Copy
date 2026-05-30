"""Volume Profile — daily POC / Value Area (causal, no-lookahead).

For each completed UTC day, build a volume-by-price histogram and derive:
  - POC  (Point of Control): the price bin with the most traded volume.
  - VAH / VAL (Value Area High/Low): the contiguous price range around POC that
    contains `va_pct` (default 70%) of the day's volume.

Each bar is attached the PRIOR day's POC/VAH/VAL (the just-completed profile),
so a strategy trading day D uses only day D-1's finished profile → no lookahead.

Requires a real `volume` column (Binance BTC/PAXG have it; HistData XAU does not).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_value_area(df: pd.DataFrame, n_bins: int = 50, va_pct: float = 0.70
                     ) -> pd.DataFrame:
    """Attach prior-day POC / VAH / VAL to every bar.

    Returns DataFrame indexed like df with columns ['poc','vah','val'] (NaN on
    the first day, before any completed profile exists).
    """
    if "volume" not in df.columns:
        raise ValueError("daily_value_area requires a 'volume' column")
    d = df.copy()
    d["__date"] = d["timestamp"].dt.normalize()

    profiles: dict[pd.Timestamp, tuple[float, float, float]] = {}
    for date, g in d.groupby("__date", sort=True):
        lo = float(g["low"].min()); hi = float(g["high"].max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        tp = ((g["high"] + g["low"] + g["close"]) / 3.0).to_numpy()
        idx = np.clip(np.digitize(tp, edges) - 1, 0, n_bins - 1)
        vol = np.zeros(n_bins)
        np.add.at(vol, idx, g["volume"].to_numpy(dtype="float64"))
        total = vol.sum()
        if total <= 0:
            continue
        poc_bin = int(vol.argmax())
        poc = float(centers[poc_bin])
        # expand value area contiguously from POC until va_pct of volume covered
        target = va_pct * total
        lo_b = hi_b = poc_bin
        acc = vol[poc_bin]
        while acc < target and (lo_b > 0 or hi_b < n_bins - 1):
            left = vol[lo_b - 1] if lo_b > 0 else -1.0
            right = vol[hi_b + 1] if hi_b < n_bins - 1 else -1.0
            if right >= left:
                hi_b += 1; acc += vol[hi_b]
            else:
                lo_b -= 1; acc += vol[lo_b]
        vah = float(edges[hi_b + 1])
        val = float(edges[lo_b])
        profiles[date] = (poc, vah, val)

    dates = sorted(profiles.keys())
    prior: dict[pd.Timestamp, tuple[float, float, float]] = {}
    for k in range(1, len(dates)):
        prior[dates[k]] = profiles[dates[k - 1]]

    nan3 = (np.nan, np.nan, np.nan)
    mapped = d["__date"].map(lambda x: prior.get(x, nan3))
    poc = mapped.map(lambda t: t[0]).to_numpy(dtype="float64")
    vah = mapped.map(lambda t: t[1]).to_numpy(dtype="float64")
    val = mapped.map(lambda t: t[2]).to_numpy(dtype="float64")
    return pd.DataFrame({"poc": poc, "vah": vah, "val": val}, index=df.index)
