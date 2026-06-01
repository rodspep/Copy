"""Combined H1+M30 ob_fvg_trend on ONE $1000 account (like the live bot running
both timeframes), over the full window and YTD-2026.

Each strategy's per-trade R_realized is computed by the parity engine (real costs,
fixed risk so R is path-independent). We then merge both timeframes' trades by
entry time and replay them on a single compounding account — concurrent positions
allowed (the live bot does run both at once).

Run: python -X utf8 -m scripts.combined_account
"""
from __future__ import annotations

import pandas as pd

from src.data.tv_loader import load_tv_csv
from src.backtest.engine import run_backtest
from src.strategies.xau.ob_fvg_trend import XauObFvgTrend

DEPLOY = {"swing_left": 3, "swing_right": 3, "ema_fast": 50, "ema_slow": 100,
          "atr_period": 14, "tol_atr": 0.3, "sl_buf_atr": 1.0, "tp_rr": 3.0,
          "allow_short": False}


def trades_for(tf: str, start=None) -> pd.DataFrame:
    df = load_tv_csv(f"data/xau/XAUUSD_{tf}.csv")
    if start:
        df = df[df["timestamp"] >= start]
    df = df.reset_index(drop=True)
    sigs = XauObFvgTrend().generate_signals(df, {}, params=DEPLOY).signals
    # fixed risk + no compounding → R_realized is a clean path-independent R multiple
    res = run_backtest(df, sigs, "XAUUSD", tf,
                       {"initial_equity": 10_000.0, "risk_pct": 0.01, "compounding": False})
    tr = res["trades"][["entry_time", "R_realized"]].dropna()
    tr["tf"] = tf
    return tr


def replay(trades: pd.DataFrame, start_eq: float, risk: float) -> tuple[float, float]:
    eq = start_eq
    peak = eq
    max_dd = 0.0
    for _, t in trades.sort_values("entry_time").iterrows():
        eq += t["R_realized"] * (eq * risk)        # risk % of current equity per trade
        peak = max(peak, eq)
        max_dd = min(max_dd, eq / peak - 1)
    return eq, max_dd


def main() -> int:
    windows = {
        "FULL overlap (2025-02 → 2026-06)": "2025-02-21",
        "YTD 2026 (2026-01 → 2026-06)": "2026-01-01",
    }
    for label, start in windows.items():
        h1 = trades_for("H1", start)
        m30 = trades_for("M30", start)
        both = pd.concat([h1, m30], ignore_index=True)
        days = (pd.Timestamp("2026-06-01", tz="UTC") - pd.Timestamp(start, tz="UTC")).days
        print(f"\n=== {label} · {days}d · H1 {len(h1)} + M30 {len(m30)} = {len(both)} trades ===")
        print(f"{'risk':>6} {'final$':>9} {'return':>8} {'CAGR':>8} {'maxDD':>7}")
        for risk in (0.005, 0.01, 0.02):
            final, dd = replay(both, 1000.0, risk)
            ret = final / 1000 - 1
            cagr = (final / 1000) ** (365.0 / days) - 1 if days > 0 else 0
            print(f"{risk*100:>5.1f}% {final:>9.0f} {ret:>+7.0%} {cagr:>+7.0%} {dd:>6.0%}")
    print("\nStart $1000, compounding, real spread/slippage. Concurrent H1+M30 positions "
          "allowed (live bot runs both). CAGR = annualized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
