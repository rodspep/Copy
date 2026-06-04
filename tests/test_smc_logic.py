"""Tests for the standalone SMC bot's pure core (src/exec/smc_logic.py).

Two jobs:
  1. PARITY — the live core must be byte-identical in behaviour to the backtest that
     produced the +$5562 / maxDD -$644 / WR 32% numbers. We assert (a) swings/htf match
     the research modules exactly, and (b) a full backtest driven by smc_logic.detect
     reproduces the known-good trade count / net / maxDD.
  2. UNIT — detect() fires on a hand-built sweep+CHOCH and stays silent when any
     precondition (sweep, CHOCH, HTF alignment, sane R, known swings) is missing.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.exec import smc_logic as S


# --------------------------------------------------------------------------- parity
@pytest.fixture(scope="module")
def data():
    from scripts.smc_replica import m15
    return m15()


def test_swings_parity(data):
    from scripts.smc_replica import swings as ref_swings
    a_sh, a_sl = S.swings(data)
    b_sh, b_sl = ref_swings(data)
    assert np.array_equal(a_sh, b_sh, equal_nan=True)
    assert np.array_equal(a_sl, b_sl, equal_nan=True)


def test_htf_parity(data):
    from scripts.optimize import htf_trend as ref_htf
    assert np.array_equal(S.htf_trend(data), ref_htf(data))


def test_backtest_parity(data):
    """A full 2-leg (4R+10R) backtest driven by smc_logic.detect must reproduce the
    canonical numbers. smc_sizing.smc_legged now calls smc_logic.detect internally, so
    this pins the end-to-end behaviour the live bot inherits."""
    from scripts.optimize import swings, htf_trend
    from scripts.smc_sizing import smc_legged, metrics
    sh, sl = swings(data); htf = htf_trend(data)
    m = metrics(smc_legged(data, sh, sl, [(4, 1), (10, 1)], htf=htf, be_after=1))
    assert m["n"] == 519
    assert 5560 <= m["net"] <= 5565          # +$5562
    assert -646 <= m["maxdd"] <= -642         # -$644
    assert 31 <= m["wr"] <= 33                # 32%


def test_live_sim_realizable_parity(data):
    """Pin the REALIZABLE live-state-machine numbers (no-gate, cap 4 concurrent) — what the
    deployed bot must reproduce. The idealized smc_legged (+$5562) used look-ahead fill and
    is NOT the deployable target; the realizable edge is ~+$5019 / maxDD -$915 / WR 28%."""
    from scripts.optimize import swings, htf_trend
    from scripts.smc_live_sim import live_sim, metrics
    sh, sl = swings(data); htf = htf_trend(data)
    tr, mc = live_sim(data, sh, sl, htf, [4, 10], [1, 1], max_setups=4, gate=False)
    m = metrics(tr)
    assert 4850 <= m["net"] <= 5200           # +$5019
    assert -970 <= m["maxdd"] <= -860         # -$915
    assert mc <= 4 and 26 <= m["wr"] <= 30    # cap respected; WR ~28%


def test_decide_matches_detect_on_window(data):
    """The live entrypoint decide(window) must equal detect() at the window's last bar.
    Find a real setup bar, then assert decide() on the truncated window reproduces it."""
    o, h, l, c = (data[k].values for k in ("open", "high", "low", "close"))
    sh, sl = S.swings(data); htf = S.htf_trend(data)
    # first setup bar comfortably past warmup
    hit = next(i for i in range(300, len(data) - 1)
               if S.detect(o, h, l, c, sh, sl, htf, i) is not None)
    sign, entry, slp, R = S.detect(o, h, l, c, sh, sl, htf, hit)
    setup = S.decide(data.iloc[:hit + 1])
    assert setup is not None
    assert setup.direction == ("long" if sign > 0 else "short")
    assert setup.entry == round(entry, 3) and setup.sl == round(slp, 3)
    # 2 legs, correct multiples, both 0.01, runner strictly further than near
    near, run = setup.legs
    assert near.role == "near" and run.role == "runner"
    assert near.lot == run.lot == 0.01
    assert near.tp_r == 4.0 and run.tp_r == 10.0
    if sign > 0:
        assert run.tp_price > near.tp_price > setup.entry
    else:
        assert run.tp_price < near.tp_price < setup.entry


# ----------------------------------------------------------------------------- unit
def _long_scene():
    """10 bars; bar 9 sweeps below swing-low 100 (l[3]=99) then closes 111 (> swing-high
    110), green. seg=[1:10]. OB low=99, top=111 -> entry 105, sl 97, R 8."""
    o = np.full(10, 100.0); h = np.full(10, 101.0)
    l = np.full(10, 100.0); c = np.full(10, 100.0)
    l[3] = 99.0                      # liquidity sweep below swl
    o[9], c[9], h[9] = 109.0, 111.0, 111.0   # CHOCH close above swh, green
    last_sh = np.full(10, 110.0); last_sl = np.full(10, 100.0)
    htf = np.ones(10)                # H1 trend up -> aligns long
    return o, h, l, c, last_sh, last_sl, htf


def test_detect_long_fires():
    o, h, l, c, sh, sl, htf = _long_scene()
    res = S.detect(o, h, l, c, sh, sl, htf, 9)
    assert res is not None
    sign, entry, slp, R = res
    assert sign == 1
    assert entry == pytest.approx(105.0) and slp == pytest.approx(97.0)
    assert R == pytest.approx(8.0)


def test_detect_short_fires():
    o = np.full(10, 100.0); h = np.full(10, 100.0)
    l = np.full(10, 99.0); c = np.full(10, 100.0)
    h[3] = 101.0                     # sweep above swh
    o[9], c[9], l[9] = 91.0, 89.0, 89.0      # close below swl, red
    last_sh = np.full(10, 100.0); last_sl = np.full(10, 90.0)
    htf = -np.ones(10)
    res = S.detect(o, h, l, c, last_sh, last_sl, htf, 9)
    assert res is not None and res[0] == -1
    # OB top = h[3]=101 (max in seg); entry = 101-(101-89)*0.5 = 95; sl = 101+2 = 103
    assert res[1] == pytest.approx(95.0) and res[2] == pytest.approx(103.0)


def test_detect_blocks_when_htf_opposes():
    o, h, l, c, sh, sl, htf = _long_scene()
    htf = -np.ones(10)               # trend down vs a long setup
    assert S.detect(o, h, l, c, sh, sl, htf, 9) is None
    # but disabling the filter lets it through
    assert S.detect(o, h, l, c, sh, sl, htf, 9, htf_align=0) is not None


def test_detect_needs_sweep_and_choch():
    o, h, l, c, sh, sl, htf = _long_scene()
    l[3] = 100.0                     # no sweep below the swing low
    assert S.detect(o, h, l, c, sh, sl, htf, 9) is None
    o, h, l, c, sh, sl, htf = _long_scene()
    c[9] = 109.0                     # no CHOCH (close not above swh 110)
    assert S.detect(o, h, l, c, sh, sl, htf, 9) is None


def test_detect_rejects_oversized_R():
    o, h, l, c, sh, sl, htf = _long_scene()
    h[5] = 160.0                     # huge sweep range -> R > 25
    assert S.detect(o, h, l, c, sh, sl, htf, 9) is None


def test_detect_needs_known_swings():
    o, h, l, c, sh, sl, htf = _long_scene()
    sh = sh.copy(); sh[9] = np.nan
    assert S.detect(o, h, l, c, sh, sl, htf, 9) is None
