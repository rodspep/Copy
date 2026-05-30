# TT — Scalping Bot Research Framework

Backtest + optimization framework for high-winrate scalp strategies on XAU/USD and BTC/USDT.

## Folder layout

```
TT/
├── src/                # production code (importable as `from src.* import ...`)
│   ├── data/           # market-data loaders (Binance, Dukascopy)
│   ├── indicators/     # EMA, MACD, RSI, ATR, BB, VWAP, SMC (OB/FVG/BOS)
│   ├── strategies/     # one file per strategy variant
│   ├── backtest/       # vectorized engine + metrics
│   ├── optimize/       # walk-forward + Optuna
│   ├── reports/        # HTML / plotly outputs
│   └── config.py       # paths, symbol specs, fees
├── tests/              # ALL test code (mirror src/ layout). No test code in __main__.
├── configs/            # YAML/JSON configs
│   ├── strategies/     # per-strategy parameter files
│   └── backtest/       # backtest run configs
├── data/               # downloaded market data
│   ├── btc/<sym>/<tf>/<YYYY-MM>.parquet
│   ├── xau/<sym>/<tf>/<YYYY-MM>.parquet
│   └── _cache/         # raw .bi5 / .zip cache
├── results/            # backtest outputs (timestamped)
├── docs/               # design notes, decisions, research
│   ├── decisions/      # ADRs
│   └── research/       # exploratory notes
├── scripts/            # CLI entry points
├── notebooks/          # exploratory Jupyter
└── requirements.txt
```

## Workflow rules

1. **Never put test code in `__main__` blocks under `src/`.** Move it to `tests/`.
2. **Never write loose `.py` files at the repo root.** Use `scripts/` for CLI tools.
3. **Every non-trivial new module gets 2-3 reviewer rounds** (via subagent) before being marked done. Apply fixes between rounds.
4. **Strategy configs** live in `configs/strategies/<name>.yaml`. Backtest configs in `configs/backtest/`.
5. **Design notes / decision rationale** → `docs/`.

## Quick start

```bash
pip install -r requirements.txt

# Download last 35 days of BTC 5m (smoke test):
python -m tests.data.test_binance_loader
```
