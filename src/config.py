"""Project-wide config: paths, symbol specs, fees, sizing.

All values here are the **single source of truth** referenced by both the backtest
engine (`src/backtest/`) and any future live adapter (`src/live/...`). The execution
parity contract in `docs/decisions/backtest_live_parity.md` mandates that both
implementations read the same `SYMBOLS` dict — never hardcode fees, spreads, or
size steps anywhere else.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CONFIGS_DIR = ROOT / "configs"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
CONFIGS_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Symbol specifications
#
# Fields (all required, even if zero):
#   pip                  — price units per "1 pip" (XAU 0.1, BTC 1.0)
#   min_tick             — smallest price increment the broker accepts
#   qty_step             — smallest quantity increment for orders (lot step / qty step)
#   min_qty              — broker minimum order quantity; sub-min sizes are SKIPPED, not enlarged
#   contract_multiplier  — PnL in account currency per 1.0 price move per 1.0 qty
#                          (XAU on MT5 std lots: 100 oz/lot; here we model qty as ounces -> 1.0)
#                          (BTC spot: qty in BTC, PnL in USD per 1.0 USD price move per 1.0 BTC -> 1.0)
#   spread_pips          — synthetic spread cost folded into entry price (ADR §3.3)
#   slippage_pips        — entry slippage AND SL slippage (ADR §3.1, §3.4)
#   commission_pct       — percentage fee on notional, applied entry AND exit (BTC taker default)
#   commission_usd       — fixed fee per executed order fill in account currency,
#                          independent of qty; applied entry AND exit. (For per-lot
#                          ECN commissions, add a `commission_usd_per_qty` field
#                          later; v1 does not need it.)
#   category             — informational label
# -----------------------------------------------------------------------------
SYMBOLS = {
    "XAUUSD": {
        "pip": 0.1,
        "min_tick": 0.01,
        "qty_step": 0.01,          # 0.01 oz minimum increment (Exness Pro/Raw); update per broker
        "min_qty": 0.01,
        "contract_multiplier": 1.0,  # qty in ounces -> $1 PnL per $1 move per oz
        "spread_pips": 2.0,        # typical Exness Standard spread (in pips of 0.1)
        "slippage_pips": 1.0,      # entry + SL slippage allowance
        "commission_pct": 0.0,
        "commission_usd": 0.0,
        "category": "fx_metal",
    },
    "BTCUSDT": {
        "pip": 1.0,
        "min_tick": 0.01,
        "qty_step": 0.00001,       # Binance BTCUSDT spot LOT_SIZE.stepSize as of 2026
        "min_qty": 0.00001,
        "contract_multiplier": 1.0,  # qty in BTC -> $1 PnL per $1 move per BTC
        "spread_pips": 1.0,        # Binance spot ~$1 on liquid hours
        "slippage_pips": 1.0,
        "commission_pct": 0.0004,  # 0.04% taker, applied on BOTH entry and exit (ADR §3.5)
        "commission_usd": 0.0,
        "category": "crypto_spot",
    },
}

TIMEFRAMES = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}

# Bar duration in pandas Timedelta strings — used for HTF availability-time alignment (ADR §5).
TIMEFRAME_DELTAS = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}
