"""Build the UG feature matrix from captured signals + OHLC, print discriminators.

OHLC source (authoritative = TradingView export per tv_loader):
  --tv data/tv/OANDA_XAUUSD_5m.csv        TradingView CSV export
  --parquet XAUUSD --tf M5                 MT5/parquet via histdata_loader.load

Signals: structured JSONL (see src/analysis/signals.py), produced by the parser
once UG's message format is known.

Usage:
  python -m scripts.analyze_ug --signals data/ug/signals.jsonl --tv data/tv/xau_m5.csv
  python -m scripts.analyze_ug --signals data/ug/signals.jsonl --parquet XAUUSD --tf M5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.features import build_feature_matrix
from src.analysis.signals import load_signals

OUT = Path("data/ug/features.parquet")


def _load_ohlc(args) -> pd.DataFrame:
    if args.tv:
        from src.data.tv_loader import load_tv_csv
        return load_tv_csv(args.tv)
    if args.parquet:
        from src.data import histdata_loader
        return histdata_loader.load(args.parquet, args.tf)
    raise SystemExit("Provide --tv <csv> or --parquet <symbol> --tf <tf>")


def _summary(fm: pd.DataFrame) -> None:
    """Per-direction view: which conditions differ most between long and short."""
    if fm.empty:
        print("No feature rows (no signals landed in the data range).")
        return
    num = fm.select_dtypes("number").drop(columns=["dir_sign"], errors="ignore")
    means = fm.groupby("direction")[num.columns].mean(numeric_only=True).T
    counts = fm["direction"].value_counts().to_dict()
    have_both = {"long", "short"}.issubset(means.columns)
    if have_both:
        means["spread"] = (means["long"] - means["short"]).abs()
        means = means.sort_values("spread", ascending=False)
    print(f"\nSignals: {fm.attrs.get('n_signals')} · used {len(fm)} · "
          f"skipped {fm.attrs.get('skipped')} (before data) · "
          f"bar={fm.attrs.get('bar_seconds')}s")
    print(f"By direction: {counts}\n")
    if not have_both:
        print("⚠️ Only one direction present — long↔short spread N/A. "
              "Showing per-direction means:")
    else:
        print("Top features by long↔short spread (mean per direction):")
    with pd.option_context("display.max_rows", 40, "display.width", 120):
        print(means.round(3).to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True, help="structured signals JSONL")
    ap.add_argument("--tv", help="TradingView CSV export")
    ap.add_argument("--parquet", help="symbol for histdata_loader.load (e.g. XAUUSD)")
    ap.add_argument("--tf", default="M5", help="timeframe for --parquet")
    ap.add_argument("--out", default=str(OUT), help="output parquet path")
    args = ap.parse_args()

    signals = load_signals(args.signals)
    if not signals:
        raise SystemExit(f"No signals in {args.signals}")
    df = _load_ohlc(args)
    if df.empty:
        raise SystemExit("OHLC source is empty.")
    fm = build_feature_matrix(signals, df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fm.to_parquet(out, index=False)
    print(f"Wrote feature matrix → {out}  ({len(fm)} rows × {fm.shape[1]} cols)")
    _summary(fm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
