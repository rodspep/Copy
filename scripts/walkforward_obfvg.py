"""Scoped walk-forward (out-of-sample) for ob_fvg_trend on fresh H1 Exness data.

Step 1 of the pivot. Rolling train/test: on each TRAIN window grid-search a small
param set, pick the best by train expectancy (min-trades gated), then apply those
params UNSEEN to the next TEST window. Accumulated TEST results = honest OOS.

Lightweight (no Optuna / no parquet); H1 only (620 days ≈ enough folds). Real
spread/slippage via the parity engine. Long-only (the deploy config).

Run: python -X utf8 -m scripts.walkforward_obfvg
"""
from __future__ import annotations

import itertools

import pandas as pd

from src.data.tv_loader import load_tv_csv
from src.backtest.engine import run_backtest
from src.strategies.xau.ob_fvg_trend import XauObFvgTrend

BT = {"initial_equity": 10_000.0, "risk_pct": 0.005, "compounding": False}  # fixed risk for fair fold compare
TRAIN_DAYS, TEST_DAYS, STEP_DAYS = 180, 90, 90
MIN_TRADES = 8

GRID = {
    "ema_fast": [34, 50],
    "ema_slow": [89, 100, 144],
    "tp_rr": [2.0, 2.5, 3.0, 3.5],
    "tol_atr": [0.3],
    "sl_buf_atr": [1.0],
    "swing_left": [3], "swing_right": [3], "atr_period": [14],
    "allow_short": [False],
}


def _combos():
    keys = list(GRID)
    for vals in itertools.product(*(GRID[k] for k in keys)):
        yield dict(zip(keys, vals))


def _expR(df, sigs, tf):
    res = run_backtest(df, sigs, "XAUUSD", tf, BT)
    tr = res["trades"]
    if tr.empty:
        return 0, float("-inf"), 0.0
    r = tr["R_realized"].dropna()
    return len(tr), r.mean(), r.sum()


def main() -> int:
    df = load_tv_csv("data/xau/XAUUSD_H1.csv").sort_values("timestamp").reset_index(drop=True)
    t0, t1 = df["timestamp"].min(), df["timestamp"].max()
    print(f"H1 {len(df)} bars · {t0.date()} → {t1.date()} · "
          f"train {TRAIN_DAYS}d / test {TEST_DAYS}d / step {STEP_DAYS}d\n")
    strat = XauObFvgTrend()

    folds, oos_sumR, oos_n = [], 0.0, 0
    start = t0
    while True:
        tr_lo = start
        tr_hi = tr_lo + pd.Timedelta(days=TRAIN_DAYS)
        te_hi = tr_hi + pd.Timedelta(days=TEST_DAYS)
        if te_hi > t1:
            break
        train = df[(df.timestamp >= tr_lo) & (df.timestamp < tr_hi)].reset_index(drop=True)
        test = df[(df.timestamp >= tr_hi) & (df.timestamp < te_hi)].reset_index(drop=True)
        # grid search on train
        best, best_expR = None, float("-inf")
        for p in _combos():
            sigs = strat.generate_signals(train, {}, params=p).signals
            n, e, _ = _expR(train, sigs, "H1")
            if n >= MIN_TRADES and e > best_expR:
                best_expR, best = e, p
        if best is None:
            start = start + pd.Timedelta(days=STEP_DAYS); continue
        # apply UNSEEN to test
        sigs = strat.generate_signals(test, {}, params=best).signals
        n, e, s = _expR(test, sigs, "H1")
        folds.append((tr_hi.date(), te_hi.date(), best["ema_fast"], best["ema_slow"],
                      best["tp_rr"], best_expR, n, e, s))
        oos_sumR += s; oos_n += n
        start = start + pd.Timedelta(days=STEP_DAYS)

    print(f"{'test_start':>11} {'test_end':>10} {'ema':>7} {'tp_rr':>5} "
          f"{'trainE':>7} {'n':>4} {'oosE':>7} {'oosSumR':>8}")
    for ts, te, ef, es, rr, tre, n, e, s in folds:
        print(f"{str(ts):>11} {str(te):>10} {f'{ef}/{es}':>7} {rr:>5.1f} "
              f"{tre:>+7.3f} {n:>4} {e:>+7.3f} {s:>+8.1f}")
    avg = oos_sumR / oos_n if oos_n else 0
    print(f"\nOOS total: {oos_n} trades · sumR {oos_sumR:+.1f} · meanR {avg:+.3f}")
    print("Each fold's params chosen on TRAIN only, measured on the next unseen TEST.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
