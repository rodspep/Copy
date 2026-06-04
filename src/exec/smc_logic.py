"""Pure SMC decision core — the SINGLE SOURCE OF TRUTH shared by the backtest and the
live bot (backtest<->live parity rule: docs/decisions/smc_backtest_live_parity.md).

This mirrors the WALK-FORWARD-SELECTED config used in every sizing/edge analysis
(scripts/smc_sizing.py, optimize.smc_trades): W=2 fractal swings, sweep_win=8,
sl_buf=2.0, H1-EMA50 trend filter, OB-retest entry at 50% of the sweep range, with
the LIVE 2-leg exit (0.01 booked @ +4R then SL->BE; 0.01 runner @ +10R).

NOT the older scripts/smc_replica.py defaults (3/5/8R, buf 0.5, win 6) — those were
the first draft; the optimized path above is what produced +$5562 / maxDD -$644 /
WR 32% over 18 months at 0.02 lot. Parity is pinned by tests/test_smc_logic.py.

Anti-lookahead: swings are confirmed W bars late; htf uses the last CLOSED H1; the
live bot only ever calls decide() on CLOSED M15 bars.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# --- canonical params (walk-forward selected; DO NOT touch without a re-backtest) ---
W = 2                 # fractal swing half-window (a pivot at bar i is confirmed at i+W)
SWEEP_WIN = 8         # lookback window for the liquidity sweep + order block
SL_BUF = 2.0          # price beyond the OB edge for the stop
RETEST_BARS = 24      # bars to wait for an OB retest (pending-order lifetime)
HORIZON = 96          # bars to resolve a filled trade (~24h on M15) then market-close
TP_NEAR_R = 4.0       # leg A target (R multiple); books then drags SL -> BE
TP_RUN_R = 10.0       # leg B target (runner)
R_MAX = 25.0          # sane stop-distance cap (price points)
LOT_PER_LEG = 0.01    # broker minimum; 2 legs => 0.02 / signal
BE_AFTER = 1          # SL -> entry once this many legs have booked a TP


@dataclass(frozen=True)
class Leg:
    role: str          # "near" | "runner"
    tp_r: float        # R multiple of the TP
    tp_price: float    # absolute TP price
    lot: float


@dataclass(frozen=True)
class Setup:
    direction: str            # "long" | "short"
    entry: float              # OB-retest limit price (anchor)
    sl: float
    R: float                  # |entry - sl| in price points
    legs: tuple               # (Leg near, Leg runner)
    bar_time: pd.Timestamp    # time of the CLOSED bar that triggered the setup
    be_after: int = BE_AFTER


def swings(d):
    """Causal fractal swings (W=2): bar i is a swing high if its high is the max of
    [i-W, i+W]; usable only at i+W (anti-lookahead). Returns (last_sh, last_sl) — the
    most recent confirmed swing hi/lo price 'as known at each bar', forward-filled.

    Verbatim-equivalent to scripts.smc_replica.swings (pinned by parity test)."""
    h, l, n = d["high"].values, d["low"].values, len(d)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for i in range(W, n - W):
        win_h, win_l = h[i - W:i + W + 1], l[i - W:i + W + 1]
        if h[i] == win_h.max():
            sh[i + W] = h[i]
        if l[i] == win_l.min():
            sl[i + W] = l[i]
    last_sh = pd.Series(sh).ffill().values
    last_sl = pd.Series(sl).ffill().values
    return last_sh, last_sl


def htf_trend(d):
    """H1 EMA50 trend (+1/-1/0) aligned to each M15 bar, anti-lookahead (last closed H1).
    Verbatim-equivalent to optimize.htf_trend (pinned by parity test)."""
    r = (d.set_index("time")["close"]
         .resample("60min", label="right", closed="right").last().dropna())
    e = r.ewm(span=50, adjust=False).mean()
    tf = pd.DataFrame({"end": r.index, "htf": np.sign(r.values - e.values)})
    m = pd.merge_asof(d[["time"]], tf, left_on="time", right_on="end", direction="backward")
    return m["htf"].fillna(0).values


def detect(o, h, l, c, last_sh, last_sl, htf, i,
           sweep_win=SWEEP_WIN, sl_buf=SL_BUF, htf_align=1):
    """The setup test AT bar i. Returns (sign, entry, sl, R) where sign=+1 long / -1 short,
    or None. Pure; mirrors optimize.smc_trades / smc_sizing.smc_legged exactly.

    Bullish: recent window swept the prior swing LOW (grabbed sell-side liquidity) AND the
    bar closes back above the prior swing HIGH (CHOCH up) AND closes green; bearish mirrors.
    Entry = 50% retest into the order block; SL = just beyond the OB edge."""
    if np.isnan(last_sh[i]) or np.isnan(last_sl[i]):
        return None
    swh, swl = last_sh[i], last_sl[i]
    seg = slice(max(0, i - sweep_win), i + 1)
    sign = None
    if l[seg].min() < swl and c[i] > swh and c[i] > o[i]:
        sign = 1
    elif h[seg].max() > swh and c[i] < swl and c[i] < o[i]:
        sign = -1
    if sign is None:
        return None
    if htf_align and htf is not None and htf[i] != sign:
        return None
    if sign > 0:
        ob = l[seg].min(); entry = ob + (h[seg].max() - ob) * 0.5; sl = ob - sl_buf
    else:
        ob = h[seg].max(); entry = ob - (ob - l[seg].min()) * 0.5; sl = ob + sl_buf
    R = abs(entry - sl)
    if R <= 0 or R > R_MAX:
        return None
    return sign, entry, sl, R


def build_setup(sign, entry, sl, R, bar_time,
                lot_per_leg=LOT_PER_LEG, near_r=TP_NEAR_R, run_r=TP_RUN_R):
    """Turn a detected (sign, entry, sl, R) into the 2-leg bracket to place."""
    near = entry + sign * near_r * R
    run = entry + sign * run_r * R
    return Setup(
        direction="long" if sign > 0 else "short",
        entry=round(entry, 3), sl=round(sl, 3), R=R,
        legs=(Leg("near", near_r, round(near, 3), lot_per_leg),
              Leg("runner", run_r, round(run, 3), lot_per_leg)),
        bar_time=bar_time,
    )


def decide(window_df, htf_align=1):
    """LIVE entrypoint. Given a window of CLOSED M15 bars (cols: time, open, high, low,
    close), return a Setup if the LATEST bar triggers a fresh SMC setup, else None.

    The window must be long enough for swings/HTF to be stable as of the last bar — the
    live bot passes >= a few hundred bars (see the contract doc). Computing swings/htf on
    the window and reading the last index is exactly what the backtest does at that bar."""
    if window_df is None or len(window_df) < 2 * W + 2:
        return None
    d = window_df.reset_index(drop=True)
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    last_sh, last_sl = swings(d)
    htf = htf_trend(d)
    i = len(d) - 1
    res = detect(o, h, l, c, last_sh, last_sl, htf, i, htf_align=htf_align)
    if res is None:
        return None
    sign, entry, sl, R = res
    return build_setup(sign, entry, sl, R, d["time"].iloc[i])
