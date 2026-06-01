"""Validate the repo's existing XAU edges on the FRESH Exness MT5 data (with costs).

Pivot from copying UG: test strategies that are already in the registry against
the newly-fetched MT5 CSVs through the parity engine (real spread/slippage), to
see which hold up on the live broker feed. Starts with ob_fvg_trend (the repo's
"best edge", marked positive 3/4 yrs) at its deploy params.

Run: python -X utf8 -m scripts.validate_edges
"""
from __future__ import annotations

from src.data.tv_loader import load_tv_csv
from src.backtest.engine import run_backtest
from src.strategies.xau.ob_fvg_trend import XauObFvgTrend

BT = {"initial_equity": 10_000.0, "risk_pct": 0.005, "compounding": True}

# Live-validated deploy params (from scripts/live_signal_bot DEPLOY_PARAMS).
DEPLOY = {"swing_left": 3, "swing_right": 3, "ema_fast": 50, "ema_slow": 100,
          "atr_period": 14, "tol_atr": 0.3, "sl_buf_atr": 1.0, "tp_rr": 3.0}


def stat(trades):
    if trades.empty:
        return None
    r = trades["R_realized"].dropna()
    return dict(n=len(trades), wr=(r > 0).mean(), sumR=r.sum(), meanR=r.mean(),
                net=trades["pnl"].sum())


def main() -> int:
    print(f"{'strategy':>14} {'tf':>4} {'short?':>6} {'bars':>6} {'trades':>7} "
          f"{'WR':>5} {'meanR':>7} {'sumR':>8} {'net$':>9}")
    for tf in ("H1", "M30", "M5"):
        df = load_tv_csv(f"data/xau/XAUUSD_{tf}.csv")
        for allow_short in (False, True):
            params = {**DEPLOY, "allow_short": allow_short}
            sigs = XauObFvgTrend().generate_signals(df, {}, params=params).signals
            res = run_backtest(df, sigs, "XAUUSD", tf, BT)
            s = stat(res["trades"])
            if not s:
                print(f"{'ob_fvg_trend':>14} {tf:>4} {str(allow_short):>6} "
                      f"{len(df):>6} {0:>7}")
                continue
            print(f"{'ob_fvg_trend':>14} {tf:>4} {str(allow_short):>6} {len(df):>6} "
                  f"{s['n']:>7} {s['wr']:>4.0%} {s['meanR']:>+7.3f} {s['sumR']:>+8.1f} "
                  f"{s['net']:>+9.0f}")
    print("\nDeploy params, real spread+slippage, risk 0.5%/trade. "
          "meanR = expectancy R/trade. (in-sample on the fetched window)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
