"""Sanity tests for the edge-simulation core (scripts/tcu_edge.our_exit) — a buggy
replay would give wrong conclusions, so pin the three canonical paths with hand-built M1.
PIP=0.1 → TP1 +50pip = +5 price, TP3 +150pip = +15 price; 0.01 lot → $1/price; COST ~$0.30."""
from __future__ import annotations

import pandas as pd

from scripts.tcu_edge import our_exit, COST


def _m(highs, lows):
    closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
    return df, df["high"].values, df["low"].values, df["close"].values


def test_full_win_tp1_then_tp3_long():
    # entry 100, SL 90; bar1 hits TP1 (105), bar2 hits TP3 (115)
    df, hi, lo, cl = _m([100, 105, 115], [100, 104, 106])
    usd, r50, r150, ssl = our_exit(100.0, 1, 0, 90.0, df, hi, lo, cl)
    # move = (105-100)*0.5 + (115-100)*0.5 = 2.5 + 7.5 = 10 price
    assert abs(usd - (10.0 - COST)) < 1e-9
    assert r50 and r150 and not ssl


def test_straight_sl_long():
    # entry 100, SL 90; bar1 drops straight to 90 → both legs lose at SL, never +50
    df, hi, lo, cl = _m([100, 101], [100, 90])
    usd, r50, r150, ssl = our_exit(100.0, 1, 0, 90.0, df, hi, lo, cl)
    assert abs(usd - (-10.0 - COST)) < 1e-9       # (90-100)*1.0 = -10
    assert (not r50) and (not r150) and ssl


def test_tp1_then_runner_back_to_be_long():
    # entry 100, SL 90; bar1 hits TP1 (105) → book +5*0.5, SL→BE(100); bar2 falls to 100 → runner BE
    df, hi, lo, cl = _m([100, 105, 101], [100, 104, 100])
    usd, r50, r150, ssl = our_exit(100.0, 1, 0, 90.0, df, hi, lo, cl)
    # move = (105-100)*0.5 + (100-100)*0.5 = 2.5
    assert abs(usd - (2.5 - COST)) < 1e-9
    assert r50 and (not r150) and (not ssl)


def test_short_full_win():
    # short entry 100, SL 110; bar1 hits TP1 (95), bar2 hits TP3 (85)
    df, hi, lo, cl = _m([100, 96, 94], [100, 95, 85])
    usd, r50, r150, ssl = our_exit(100.0, -1, 0, 110.0, df, hi, lo, cl)
    assert abs(usd - (10.0 - COST)) < 1e-9
    assert r50 and r150 and not ssl
