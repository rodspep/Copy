"""Realistic-cost backtest of the UG-decoded S/R-reversion strategy.

Runs the repo's parity engine (next-bar-open fills + spread + slippage) on the
MT5 M5 export — the true test of whether the decoded edge survives costs.

Run: python -X utf8 -m scripts.backtest_ug_sr
"""
from __future__ import annotations

import numpy as np

from src.data.tv_loader import load_tv_csv
from src.backtest.engine import run_backtest
from src.strategies.xau.ug_sr_reversion import XauUgSrReversion

PARAMS = {"initial_equity": 10_000.0, "risk_pct": 0.005, "compounding": True}


def stats(trades) -> dict:
    if trades.empty:
        return {"n": 0}
    r = trades["R_realized"].dropna()
    wins = (r > 0).sum()
    reasons = trades["exit_reason"].value_counts().to_dict()
    return {
        "n": len(trades),
        "win_rate": wins / len(r) if len(r) else float("nan"),
        "sum_R": r.sum(),
        "mean_R": r.mean(),
        "net_pnl": trades["pnl"].sum(),
        "reasons": reasons,
    }


def main() -> int:
    df = load_tv_csv("data/xau/XAUUSD_M5.csv")
    span = f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}"
    print(f"M5: {len(df)} bars · {span} · XAUUSD spread=2pip slip=1pip\n")
    print(f"{'variant':>22} {'confirm':>8} {'trades':>7} {'WR':>5} {'meanR':>7} "
          f"{'sumR':>8} {'net$':>9}")
    for label, tp in [("scalp tp=5", 5.0), ("PRI-GOLD tp=15", 15.0)]:
        for confirm in (False, True):
            sigs = XauUgSrReversion().generate_signals(
                df, {}, params={"tp_price": tp, "require_confirm": confirm}).signals
            res = run_backtest(df, sigs, "XAUUSD", "M5", PARAMS)
            s = stats(res["trades"])
            if s["n"] == 0:
                print(f"{label:>22} {str(confirm):>8} {0:>7}")
                continue
            print(f"{label:>22} {str(confirm):>8} {s['n']:>7} {s['win_rate']:>4.0%} "
                  f"{s['mean_R']:>+7.3f} {s['sum_R']:>+8.1f} {s['net_pnl']:>+9.0f}")
    print("\nmeanR = expectancy in R per trade AFTER spread+slippage. "
          "Engine fills next-bar-open (market). risk 0.5%/trade, compounding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
