"""Portfolio bundle for ob_fvg_trend — continuous OOS equity, gold + BTC.

Builds a TRUE out-of-sample, continuously-compounding equity curve for each
symbol from its walk-forward windows.csv (each test window uses the params
Optuna picked on that window's train slice — no in-sample leakage), then
combines the two into a portfolio and reports Sharpe / MaxDD / CAGR.

How the continuous OOS curve is built (no per-window equity reset):
  1. For each window, regenerate signals on the FULL df with that window's
     best params (so EMA/swing warmup is intact).
  2. Stitch: each bar in [test_start, test_end) takes its covering window's
     signal. Windows are non-overlapping in test time → unambiguous.
  3. Run ONE backtest over the whole stitched OOS span → equity compounds
     across windows exactly as a live deployment would.

Portfolio combine: resample each symbol's equity to daily, take daily returns,
blend at fixed capital weights (default 50/50). Each symbol trades at its own
`risk_pct`; the blend just allocates capital. Reports XAU-only, BTC-only, and
the blend so the diversification benefit (or lack of it) is explicit.

Usage:
  python -X utf8 -m scripts.portfolio_ob_fvg
  python -X utf8 -m scripts.portfolio_ob_fvg --xau-only   # gold engine alone
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from src.data import histdata_loader, binance_loader
from src.strategies.xau.ob_fvg_trend import XauObFvgTrend
from src.backtest import run_backtest


WINDOWS = {
    "XAUUSD": "results/optimize/XauObFvgTrend_XAUUSD_20260530T141544/windows.csv",
    "BTCUSDT": "results/optimize/XauObFvgTrend_BTCUSDT_20260530T141625/windows.csv",
}
PARAM_COLS = ["swing_left", "swing_right", "ema_fast", "ema_slow",
              "tol_atr", "sl_buf_atr", "tp_rr"]
INT_PARAMS = {"swing_left", "swing_right", "ema_fast", "ema_slow"}
RISK_PCT = 0.01
PER_BAR_TF = {"XAUUSD": "H1", "BTCUSDT": "H1"}


def _load(symbol: str) -> pd.DataFrame:
    if symbol == "XAUUSD":
        return histdata_loader.load("XAUUSD", "H1")
    return binance_loader.load("BTCUSDT", "1h")


def _params_from_row(row: pd.Series) -> dict:
    p = {}
    for c in PARAM_COLS:
        v = row[f"best_{c}"]
        p[c] = int(v) if c in INT_PARAMS else float(v)
    return p


def continuous_oos(symbol: str, allow_short: bool = True) -> pd.DataFrame:
    """Return a continuous OOS equity curve (indexed by timestamp)."""
    df = _load(symbol)
    w = pd.read_csv(WINDOWS[symbol])
    w["test_start"] = pd.to_datetime(w["test_start"], utc=True)
    w["test_end"] = pd.to_datetime(w["test_end"], utc=True)
    w = w.sort_values("test_start").reset_index(drop=True)

    ts = df["timestamp"]
    # Build a stitched action/sl/tp by overlaying each window's masked signals.
    action = np.array(["hold"] * len(df), dtype=object)
    sl = np.full(len(df), np.nan)
    tp = np.full(len(df), np.nan)

    for _, row in w.iterrows():
        p = _params_from_row(row)
        p["allow_short"] = allow_short
        sig = XauObFvgTrend().generate_signals(df, {}, params=p).signals
        mask = (ts >= row["test_start"]) & (ts < row["test_end"])
        m = mask.to_numpy()
        a = sig["action"].to_numpy(dtype=object)
        action[m] = a[m]
        sl[m] = sig["sl"].to_numpy()[m]
        tp[m] = sig["tp"].to_numpy()[m]

    stitched = pd.DataFrame({"action": action, "sl": sl, "tp": tp}, index=df.index)
    # Restrict to the OOS span (first test_start .. last test_end).
    span = (ts >= w["test_start"].iloc[0]) & (ts < w["test_end"].iloc[-1])
    sub = df[span].reset_index(drop=True)
    ssig = stitched[span.to_numpy()].reset_index(drop=True)

    r = run_backtest(sub, ssig, symbol=symbol, ltf_tf="H1",
                     params={"initial_equity": 10000.0, "risk_pct": RISK_PCT,
                             "compounding": True})
    eq = r["equity_curve"].copy()
    # equity_curve columns: find the equity + time columns robustly
    tcol = "timestamp" if "timestamp" in eq.columns else eq.columns[0]
    ecol = "equity" if "equity" in eq.columns else [c for c in eq.columns if "equity" in c][0]
    out = eq[[tcol, ecol]].rename(columns={tcol: "timestamp", ecol: "equity"})
    out = out.set_index("timestamp")
    return out, r["trades"]


def stats(daily_ret: pd.Series, equity: pd.Series, label: str) -> dict:
    ann = 252.0
    mu = daily_ret.mean() * ann
    sd = daily_ret.std() * np.sqrt(ann)
    sharpe = mu / sd if sd > 0 else 0.0
    roll_max = equity.cummax()
    dd = (equity / roll_max - 1.0)
    maxdd = dd.min()
    yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    total = equity.iloc[-1] / equity.iloc[0] - 1
    return {"label": label, "total_return": total, "cagr": cagr,
            "sharpe": sharpe, "maxdd": maxdd}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xau-only", action="store_true")
    ap.add_argument("--xau-allow-short", action="store_true",
                    help="allow shorts on XAU (default off — gold is one-way bull)")
    args = ap.parse_args()

    curves = {}
    xeq, xtr = continuous_oos("XAUUSD", allow_short=args.xau_allow_short)
    curves["XAUUSD"] = xeq
    print(f"XAU OOS trades={len(xtr)} final_equity={xeq['equity'].iloc[-1]:.0f} "
          f"(short={'on' if args.xau_allow_short else 'OFF'})")
    if not args.xau_only:
        beq, btr = continuous_oos("BTCUSDT", allow_short=True)
        curves["BTCUSDT"] = beq
        print(f"BTC OOS trades={len(btr)} final_equity={beq['equity'].iloc[-1]:.0f}")

    # Per-symbol daily on each symbol's OWN index (no cross-contamination).
    daily_raw = {s: eq["equity"].resample("1D").last().ffill().dropna()
                 for s, eq in curves.items()}
    rets_raw = {s: d.pct_change().fillna(0.0) for s, d in daily_raw.items()}

    print("\n=== Per-symbol OOS (continuous compounding, risk 1%) ===")
    rows = []
    for sym in curves:
        rows.append(stats(rets_raw[sym], daily_raw[sym], sym))
    if len(curves) == 2:
        # Blend on the union index (only here do we align the two books).
        idx = daily_raw["XAUUSD"].index.union(daily_raw["BTCUSDT"].index)
        rx = daily_raw["XAUUSD"].reindex(idx).ffill().pct_change().fillna(0.0)
        rb = daily_raw["BTCUSDT"].reindex(idx).ffill().pct_change().fillna(0.0)
        corr = rx.corr(rb)
        blend = 0.5 * rx + 0.5 * rb
        beq = (1 + blend).cumprod() * 10000
        rows.append(stats(blend, beq, "PORTFOLIO 50/50"))
        print(f"daily-return corr(XAU,BTC) = {corr:+.3f}")
    hdr = f"{'book':18s} {'total':>9s} {'CAGR':>8s} {'Sharpe':>8s} {'MaxDD':>8s}"
    print(hdr)
    for r in rows:
        print(f"{r['label']:18s} {r['total_return']:>+8.0%} {r['cagr']:>+7.0%} "
              f"{r['sharpe']:>8.2f} {r['maxdd']:>7.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
