"""Decode UG's DIRECTION decision from features we can recompute live (MT5 data).

The geometry (SL/TP/range) is already solved deterministically; the open question
is what drives DIRECTION (and entry). This script rebuilds the inputs UG appears
to use — multi-TF SMA34/SMA89 stack + price-vs-mean + ATR — FROM MT5 OHLC (not
from UG's text), then measures how much those visible features explain UG's actual
direction. The residual = quantified hidden logic (not hand-waved as 100%).

Inputs UG cites are M5/M15/M30/H1. We have native M5/M30/H1 CSV; M15 is resampled
from M5 so the whole stack is reproducible from one feed.

Run: python -X utf8 -m scripts.ug_model
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score

from src.data.tv_loader import load_tv_csv
from src.indicators import atr, rsi, bollinger
from src.analysis.features import closed_bar_index, infer_bar_seconds

SIG = Path("data/ug/signals.jsonl")
CSV = "data/xau/XAUUSD_{}.csv"


def _strip(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()


def _text_features(raw: str) -> dict:
    """Extract directional cues UG STATES in its Elliott/SMC narrative."""
    t = _strip(raw)
    ell = re.search(r"elliott wave.*?:\s*(.*)", t)
    smc = re.search(r"smc:\s*(.*)", t)
    ell_t = ell.group(1) if ell else ""
    smc_t = smc.group(1) if smc else ""
    # CHOCH direction
    choch = 0
    if "choch bearish" in smc_t:
        choch = -1
    elif "choch bullish" in smc_t:
        choch = 1
    # Elliott net bias: up-words vs down-words
    up = len(re.findall(r"\b(len|tang)\b", ell_t))
    dn = len(re.findall(r"\b(xuong|giam)\b", ell_t))
    ell_net = 1 if up > dn else -1 if dn > up else 0
    # explicit conflict language
    conflict = int(any(w in ell_t + smc_t for w in
                       ("mau thuan", "xung dot", "trai chieu", "khong dong nhat", "chua ro")))
    return {"choch": choch, "ell_net": ell_net, "conflict": conflict}


def _resample_m15(m5: pd.DataFrame) -> pd.DataFrame:
    g = (m5.set_index("timestamp")
         .resample("15min", label="left", closed="left")
         .agg({"open": "first", "high": "max", "low": "min", "close": "last",
               "volume": "sum"}).dropna().reset_index())
    return g


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["sma34"] = df["close"].rolling(34).mean()
    df["sma89"] = df["close"].rolling(89).mean()
    df["atr"] = atr(df, 14)
    df["rsi"] = rsi(df["close"], 14)
    df["bbp"] = bollinger(df["close"])["percent_b"]
    return df


def main() -> int:
    m5 = _prep(load_tv_csv(CSV.format("M5")))
    frames = {"M5": m5, "M15": _prep(_resample_m15(load_tv_csv(CSV.format("M5")))),
              "M30": _prep(load_tv_csv(CSV.format("M30"))),
              "H1": _prep(load_tv_csv(CSV.format("H1")))}
    bs = {tf: infer_bar_seconds(f["timestamp"]) for tf, f in frames.items()}

    sigs = [json.loads(l) for l in SIG.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen, rows = set(), []
    for s in sigs:
        key = (s["direction"], s["entry"], s["sl"])
        if key in seen:
            continue
        seen.add(key)
        ts = pd.Timestamp(s["ts"])
        feat, ok = {}, True
        for tf, f in frames.items():
            i = closed_bar_index(f["timestamp"], ts, bs[tf])
            if i is None or pd.isna(f["sma34"].iloc[i]) or pd.isna(f["sma89"].iloc[i]):
                ok = False
                break
            feat[f"stack_{tf}"] = 1 if f["sma34"].iloc[i] > f["sma89"].iloc[i] else -1
            if tf == "M5":
                feat["px_vs_sma34"] = float(f["close"].iloc[i] - f["sma34"].iloc[i])
                feat["atr_m5"] = float(f["atr"].iloc[i])
                feat["close"] = float(f["close"].iloc[i])
                feat["rsi_m5"] = float(f["rsi"].iloc[i]) if pd.notna(f["rsi"].iloc[i]) else 50.0
                feat["bbp_m5"] = float(f["bbp"].iloc[i]) if pd.notna(f["bbp"].iloc[i]) else 0.5
        if not ok:
            continue
        feat.update(_text_features(s.get("raw", "")))       # UG's stated Elliott/SMC cues
        feat["dir"] = 1 if s["direction"] == "long" else -1
        feat["entry_off"] = s["entry"] - feat["close"]      # limit offset from price
        rows.append(feat)

    d = pd.DataFrame(rows)
    n = len(d)
    print(f"Unique signals with full recomputed stack: {n}  "
          f"(long {int((d['dir']==1).sum())} / short {int((d['dir']==-1).sum())})\n")

    # --- hand rule: fade vs M5 SMA34, flip to trend when stack >=3/4 aligned ---
    def hand(r):
        up = sum(r[f"stack_{tf}"] == 1 for tf in frames)
        if up >= 3:
            return 1
        if up <= 1:
            return -1
        return 1 if r["px_vs_sma34"] <= 0 else -1     # fade in mixed regime
    d["pred_hand"] = d.apply(hand, axis=1)
    acc_hand = (d["pred_hand"] == d["dir"]).mean()
    print(f"Hand rule (trend if stack>=3/4 aligned else fade vs M5 SMA34): "
          f"{acc_hand:.0%} ({int((d['pred_hand']==d['dir']).sum())}/{n})")

    # --- how much do visible features explain DIRECTION? compare feature sets ---
    y = d["dir"].to_numpy()
    k = min(5, n // 6 or 2)
    sma_feats = [f"stack_{tf}" for tf in frames] + ["px_vs_sma34"]
    text_feats = ["choch", "ell_net", "conflict"]
    pa_feats = ["rsi_m5", "bbp_m5"]
    for name, cols in [("SMA stack only", sma_feats),
                       ("SMA + stated Elliott/SMC", sma_feats + text_feats),
                       ("SMA + price-action(rsi,bb)", sma_feats + pa_feats),
                       ("ALL visible+computable", sma_feats + text_feats + pa_feats)]:
        X = d[cols].to_numpy()
        tr = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0).fit(X, y)
        cv = cross_val_score(tr, X, y, cv=k)
        print(f"  {name:28} train {tr.score(X,y):.0%} · {k}-fold CV {cv.mean():.0%}±{cv.std():.0%}")
    base = max((y == 1).mean(), (y == -1).mean())
    print(f"  (baseline = always-majority: {base:.0%})")
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0).fit(
        d[sma_feats].to_numpy(), y)
    print("\nLearned rule (SMA stack):")
    print(export_text(tree, feature_names=sma_feats))

    # --- entry offset: ATR-scaled? ---
    off = d["entry_off"].abs()
    ratio = (off / d["atr_m5"]).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Entry limit offset |entry-close|: median {off.median():.2f} price · "
          f"as ×ATR(M5): median {ratio.median():.2f} (IQR {ratio.quantile(.25):.2f}-{ratio.quantile(.75):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
