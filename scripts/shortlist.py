"""Read results/optimize/summary.csv and emit a per-symbol top-K shortlist.

Per `[[feedback-per-symbol-strategies]]`: XAU and BTC are ranked INDEPENDENTLY.
Ranking key: composite of OOS winrate, profit factor, MaxDD-penalty, expectancy.

Usage:
  python -m scripts.shortlist                # top 3 per symbol
  python -m scripts.shortlist --top 5
  python -m scripts.shortlist --min-trades 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR


def _score_row(row: pd.Series, wr_w: float = 1.0, pf_w: float = 0.5,
               dd_pen: float = 1.5, sharpe_w: float = 0.1) -> float:
    """Composite score — matches src.reports.composite_objective."""
    if row.get("oos_n_trades", 0) < 5 or row.get("oos_expectancy_R", 0.0) <= 0:
        return float("-inf")
    return (
        wr_w * row.get("oos_winrate", 0.0)
        + pf_w * np.tanh(row.get("oos_pf", 0.0) - 1.0)
        - dd_pen * row.get("oos_maxdd", 0.0)
        + sharpe_w * np.tanh(row.get("oos_sharpe", 0.0))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-symbol strategy shortlist from optimize results.")
    parser.add_argument("--summary", default=str(RESULTS_DIR / "optimize" / "summary.csv"),
                        help="Path to summary CSV from optimize_all")
    parser.add_argument("--top", type=int, default=3, help="Top K strategies per symbol")
    parser.add_argument("--min-trades", type=int, default=20,
                        help="Drop rows with fewer OOS trades than this")
    args = parser.parse_args()

    path = Path(args.summary)
    if not path.exists():
        print(f"No summary file at {path}. Run scripts/optimize_all.py first.", file=sys.stderr)
        return 2

    df = pd.read_csv(path)
    df = df[df["oos_n_trades"] >= args.min_trades].copy()
    if df.empty:
        print(f"No strategies meet min_trades={args.min_trades}", file=sys.stderr)
        return 2

    df["score"] = df.apply(_score_row, axis=1)
    df = df.sort_values(["symbol", "score"], ascending=[True, False])

    print(f"\n=== Top {args.top} per symbol (min_trades={args.min_trades}) ===\n")
    for symbol in df["symbol"].unique():
        sub = df[df["symbol"] == symbol].head(args.top)
        print(f"--- {symbol} ---")
        cols = ["strategy", "oos_n_trades", "oos_winrate", "oos_pf",
                "oos_expectancy_R", "oos_maxdd", "oos_sharpe", "oos_total_return_pct", "score"]
        cols = [c for c in cols if c in sub.columns]
        print(sub[cols].to_string(index=False))
        print()

    out = RESULTS_DIR / "optimize" / "shortlist.csv"
    df.to_csv(out, index=False)
    print(f"Full ranked table written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
