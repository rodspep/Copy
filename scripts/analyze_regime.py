"""Regime-gate analysis for walk-forward results.

For each walk-forward window of a given strategy, compute pre-window features
derived ONLY from data available before the OOS test period (no lookahead),
join with the OOS performance label (`oos_expectancy_R > 0`), and train a
tiny decision tree (depth ≤ 3) to extract human-readable regime rules.

Per Codex's recommendation: with only 38 labels per strategy, deep ML overfits.
A depth-2 tree gives us 4 leaves max — enough to express "trade only when
HTF slope X and ATR percentile Y" — and the rules are inspectable.

Usage:
  python -m scripts.analyze_regime \
      --windows-csv results/optimize/XauEmaPullback_XAUUSD_20260527T065703/windows.csv \
      --symbol XAUUSD

Outputs:
  - prints feature table + per-window labels
  - prints depth-2 decision tree rules
  - writes <strategy>_regime_features.csv next to windows.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.histdata_loader import load as load_xau
from src.indicators import ema, adx, atr
from src.indicators.smc import swings


# -----------------------------------------------------------------------------
# Features (all computed on data ENDING at window's train_end timestamp, so
# strictly causal w.r.t. the OOS window)
# -----------------------------------------------------------------------------

def _slope(series: pd.Series, n: int) -> float:
    """Slope of last `n` values per bar, normalized by series mean."""
    s = series.dropna().tail(n)
    if len(s) < n // 2 or s.mean() == 0:
        return np.nan
    x = np.arange(len(s))
    slope, _ = np.polyfit(x, s.values, 1)
    return float(slope / abs(s.mean()))


def _percentile_rank(series: pd.Series, value: float) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    return float((s <= value).mean())


def compute_features_at(h1: pd.DataFrame, m5: pd.DataFrame,
                        train_end: pd.Timestamp) -> dict[str, float]:
    """Pre-window features. All inputs filtered to bars closing at <= train_end.

    Features chosen per Codex's guidance:
      - trend strength: H1 EMA slope, ADX
      - volatility: ATR percentile (M5 and H1)
      - market structure: HH/HL frequency (swing efficiency proxy)
      - session range ratio (London vs Asia, last 20 days)
    """
    h1_hist = h1[h1["timestamp"] < train_end].copy()
    m5_hist = m5[m5["timestamp"] < train_end].copy()
    if len(h1_hist) < 250 or len(m5_hist) < 2000:
        return {}

    # Trend strength on H1
    ema50 = ema(h1_hist["close"], 50)
    ema200 = ema(h1_hist["close"], 200)
    adx_h1 = adx(h1_hist, 14)["adx"]
    feat = {
        "h1_ema50_slope": _slope(ema50, 200),               # ~8 days of H1
        "h1_ema_diff_pct": float((ema50.iloc[-1] - ema200.iloc[-1]) / ema200.iloc[-1])
                            if pd.notna(ema50.iloc[-1]) and pd.notna(ema200.iloc[-1]) else np.nan,
        "h1_adx_mean_20d": float(adx_h1.dropna().tail(480).mean()),   # 480 H1 bars ≈ 20d
        "h1_adx_now": float(adx_h1.iloc[-1]) if pd.notna(adx_h1.iloc[-1]) else np.nan,
    }

    # Volatility regime
    atr_m5 = atr(m5_hist, 14)
    atr_h1 = atr(h1_hist, 14)
    if pd.notna(atr_m5.iloc[-1]):
        feat["atr_m5_pctile"] = _percentile_rank(atr_m5.tail(20000), atr_m5.iloc[-1])  # last ~10 weeks
    if pd.notna(atr_h1.iloc[-1]):
        feat["atr_h1_pctile"] = _percentile_rank(atr_h1.tail(2400), atr_h1.iloc[-1])

    # Market structure: swing density on M5 (last 5000 bars ≈ 17 days)
    last_chunk = m5_hist.tail(5000).reset_index(drop=True)
    sw = swings(last_chunk, left=4, right=4)
    n_swings_high = sw["new_swing_high"].sum()
    n_swings_low = sw["new_swing_low"].sum()
    feat["swing_density"] = float((n_swings_high + n_swings_low) / max(len(last_chunk), 1))

    # Swing efficiency: net move / sum of |returns| over last 5000 bars
    closes = last_chunk["close"].values
    if len(closes) > 2:
        net = closes[-1] - closes[0]
        gross = np.abs(np.diff(closes)).sum()
        feat["swing_efficiency"] = float(abs(net) / gross) if gross > 0 else 0.0

    # Session range ratio (London vs Asia) over last 20 days using H1
    last_h1 = h1_hist.tail(480).copy()  # 20 days
    last_h1["hour"] = last_h1["timestamp"].dt.hour
    last_h1["date"] = last_h1["timestamp"].dt.date
    asian = last_h1[(last_h1["hour"] >= 22) | (last_h1["hour"] < 7)]
    london = last_h1[(last_h1["hour"] >= 7) & (last_h1["hour"] < 12)]
    if not asian.empty and not london.empty:
        asian_range = (asian.groupby("date")["high"].max() -
                       asian.groupby("date")["low"].min()).mean()
        london_range = (london.groupby("date")["high"].max() -
                        london.groupby("date")["low"].min()).mean()
        feat["london_asia_range_ratio"] = float(london_range / asian_range) if asian_range > 0 else np.nan
    return feat


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-csv", required=True, type=Path)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()

    if args.symbol != "XAUUSD":
        raise NotImplementedError("Only XAUUSD wired up here; add binance loader for BTC")

    windows = pd.read_csv(args.windows_csv)
    print(f"Loaded {len(windows)} windows from {args.windows_csv}")

    m5 = load_xau("XAUUSD", "M5")
    h1 = load_xau("XAUUSD", "H1")
    print(f"M5: {len(m5)} bars, H1: {len(h1)} bars")

    rows = []
    for _, w in windows.iterrows():
        train_end = pd.Timestamp(w["test_start"], tz="UTC")
        feats = compute_features_at(h1, m5, train_end)
        if not feats:
            continue
        feats["window"] = int(w["window"])
        feats["oos_exp_R"] = float(w["oos_expectancy_R"])
        feats["oos_wr"] = float(w["oos_winrate"])
        feats["label_positive"] = 1 if w["oos_expectancy_R"] > 0 else 0
        rows.append(feats)

    df = pd.DataFrame(rows)
    out_csv = args.windows_csv.with_suffix(".regime_features.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nFeature table written: {out_csv}")

    print("\n=== Feature table ===")
    cols = [c for c in df.columns if c not in {"window", "oos_exp_R", "oos_wr", "label_positive"}]
    print(df[["window", "label_positive", "oos_exp_R", "oos_wr"] + cols].to_string(index=False))

    print(f"\nPositive windows: {df['label_positive'].sum()}/{len(df)}")

    # ---- Per-feature univariate split (manual eyeballing) ----
    print("\n=== Per-feature comparison (mean by label) ===")
    summary = df.groupby("label_positive")[cols].mean().T
    summary.columns = ["mean_neg", "mean_pos"]
    summary["delta_pct"] = (summary["mean_pos"] - summary["mean_neg"]) / summary["mean_neg"].abs().replace(0, np.nan) * 100
    print(summary.to_string())

    # ---- Decision tree ----
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
    except ImportError:
        print("\n[!] sklearn not installed — skipping tree. pip install scikit-learn")
        return 0

    feat_cols = [c for c in cols if df[c].notna().sum() >= len(df) * 0.8]
    X = df[feat_cols].fillna(df[feat_cols].median())
    y = df["label_positive"].values

    tree = DecisionTreeClassifier(max_depth=args.depth, min_samples_leaf=4, random_state=42)
    tree.fit(X, y)
    print(f"\n=== Decision tree depth={args.depth} (n={len(df)} windows) ===")
    print(export_text(tree, feature_names=feat_cols))

    # In-sample fit (just for context — NOT a generalization estimate)
    print(f"In-sample accuracy: {tree.score(X, y):.3f} (baseline {max(y.mean(), 1 - y.mean()):.3f})")

    # Leave-one-out cross-val to get a fairer estimate
    from sklearn.model_selection import LeaveOneOut
    loo_correct = 0
    for tr_idx, te_idx in LeaveOneOut().split(X):
        t = DecisionTreeClassifier(max_depth=args.depth, min_samples_leaf=4, random_state=42)
        t.fit(X.iloc[tr_idx], y[tr_idx])
        loo_correct += int(t.predict(X.iloc[te_idx])[0] == y[te_idx][0])
    print(f"LOO accuracy:       {loo_correct/len(df):.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
