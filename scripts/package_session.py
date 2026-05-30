"""Package this session's optimization run into a portable bundle for handoff.

Creates `results/session_<timestamp>/` containing:
  - summary.csv             — copy of results/optimize/summary.csv
  - shortlist.csv           — copy of results/optimize/shortlist.csv
  - top_per_symbol.md       — human-readable per-symbol top-3 with metrics
  - manifest.json           — code SHA-256s, parity-doc hash, library versions, run params
  - reports/                — per-strategy manifest.json + windows.csv (small enough to commit)
  - README.md               — handoff note for the next session

The full per-trial CSVs and OOS parquets stay in their original results/optimize/<run_id>/
folders (too big to bundle).

Usage:
  python -m scripts.package_session
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import ROOT, RESULTS_DIR


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _pip_freeze() -> list[str]:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], timeout=30, text=True)
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception as e:
        return [f"<pip freeze failed: {e}>"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Package session results for handoff.")
    parser.add_argument("--out", default=None, help="Output directory (default: results/session_<ts>)")
    parser.add_argument("--top", type=int, default=3, help="Top K per symbol for shortlist")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.out) if args.out else RESULTS_DIR / f"session_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(exist_ok=True)

    summary_csv = RESULTS_DIR / "optimize" / "summary.csv"
    shortlist_csv = RESULTS_DIR / "optimize" / "shortlist.csv"

    if summary_csv.exists():
        shutil.copy2(summary_csv, out_dir / "summary.csv")
    if shortlist_csv.exists():
        shutil.copy2(shortlist_csv, out_dir / "shortlist.csv")

    # Per-strategy small artifacts (skip the big parquets)
    if (RESULTS_DIR / "optimize").exists():
        for run_dir in (RESULTS_DIR / "optimize").iterdir():
            if not run_dir.is_dir():
                continue
            target = out_dir / "reports" / run_dir.name
            target.mkdir(exist_ok=True)
            for f in ["manifest.json", "windows.csv", "trials.csv"]:
                src = run_dir / f
                if src.exists():
                    shutil.copy2(src, target / f)

    # Build the human-readable top-K-per-symbol report.
    top_md_lines: list[str] = [
        f"# Top-{args.top} per symbol\n",
        f"_Generated: {ts} UTC_\n",
    ]
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        if not df.empty:
            for symbol in sorted(df["symbol"].unique()):
                sub = df[df["symbol"] == symbol].copy()
                # Same scoring as scripts/shortlist
                import numpy as np
                def score_row(r):
                    if r.get("oos_n_trades", 0) < 5 or r.get("oos_expectancy_R", 0.0) <= 0:
                        return float("-inf")
                    return (r.get("oos_winrate", 0) +
                            0.5 * np.tanh(r.get("oos_pf", 0) - 1.0) -
                            1.5 * r.get("oos_maxdd", 0) +
                            0.1 * np.tanh(r.get("oos_sharpe", 0)))
                sub["score"] = sub.apply(score_row, axis=1)
                sub = sub.sort_values("score", ascending=False).head(args.top)
                top_md_lines.append(f"\n## {symbol}\n")
                cols = ["strategy", "oos_n_trades", "oos_winrate", "oos_pf",
                        "oos_expectancy_R", "oos_maxdd", "oos_sharpe",
                        "oos_total_return_pct", "score"]
                cols = [c for c in cols if c in sub.columns]
                top_md_lines.append("| " + " | ".join(cols) + " |")
                top_md_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
                for _, r in sub.iterrows():
                    row_vals = []
                    for c in cols:
                        v = r[c]
                        if isinstance(v, float):
                            if v == float("-inf"):
                                row_vals.append("-inf")
                            elif abs(v) < 1.0:
                                row_vals.append(f"{v:.4f}")
                            else:
                                row_vals.append(f"{v:.3f}")
                        else:
                            row_vals.append(str(v))
                    top_md_lines.append("| " + " | ".join(row_vals) + " |")
        else:
            top_md_lines.append("\n_summary.csv is empty._\n")
    else:
        top_md_lines.append("\n_No summary.csv found._\n")

    (out_dir / "top_per_symbol.md").write_text("\n".join(top_md_lines), encoding="utf-8")

    # Manifest: code SHA-256 sums, parity hash, run env
    parity_doc = ROOT / "docs" / "decisions" / "backtest_live_parity.md"
    code_files = []
    for sub in ["src/backtest/engine.py", "src/backtest/fills.py", "src/backtest/sizing.py",
                "src/indicators/__init__.py", "src/indicators/trend.py",
                "src/indicators/volatility.py", "src/indicators/volume.py",
                "src/indicators/htf.py", "src/indicators/smc.py",
                "src/strategies/base.py", "src/strategies/registry.py",
                "src/optimize/walkforward.py", "src/reports/metrics.py",
                "src/config.py"]:
        p = ROOT / sub
        if p.exists():
            code_files.append({"path": sub, "sha256": _file_sha256(p)})

    manifest = {
        "session_ts": ts,
        "parity_doc_sha256": _file_sha256(parity_doc) if parity_doc.exists() else None,
        "code_files": code_files,
        "pip_freeze": _pip_freeze(),
        "python_executable": sys.executable,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Handoff README
    readme = f"""# Session bundle — {ts}

This directory packages the results of a walk-forward optimization run so the next
session can pick up where this one left off.

## What's in this bundle

- `summary.csv`         — One row per (symbol, strategy) with aggregate OOS stats.
- `shortlist.csv`       — Full ranked table (output of `scripts/shortlist.py`).
- `top_per_symbol.md`   — Human-readable top-K per symbol (this is the headline).
- `reports/<run_id>/`   — Per-strategy `manifest.json`, `windows.csv`, `trials.csv`.
- `manifest.json`       — Code SHA-256s + parity-doc hash + `pip freeze`.

The large per-window OOS trade/equity parquets stay in `results/optimize/<run_id>/`.

## For the next session

1. Read `top_per_symbol.md` first — that's the actionable result.
2. If you want to extend a winning strategy: its hyper-params are in
   `reports/<run_id>/windows.csv` (one row per walk-forward window, columns
   `best_*` give the IS-optimal params per window).
3. If you want to re-run with the same engine: confirm the parity hash in
   `manifest.json` matches the live `docs/decisions/backtest_live_parity.md`.
4. To run again with full scope:
   ```
   python -m scripts.download_all --start 2023-01-01
   python -m scripts.optimize_all --trials 50
   python -m scripts.shortlist --top 3
   python -m scripts.package_session
   ```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Packaged session to: {out_dir}")
    print(f"Headline result: {out_dir / 'top_per_symbol.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
