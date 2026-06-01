"""Quick screen: backtest every registered XAU strategy on the fresh MT5 data.

Step 2 of the pivot — a fast lay-of-the-land using each strategy's DEFAULT params
(NOT tuned; walk-forward tunes later). HTFs are loaded from the fetched CSVs;
M15/H4 are resampled from M5/H1. Real spread/slippage via the parity engine.

Run: python -X utf8 -m scripts.screen_edges
"""
from __future__ import annotations

import pandas as pd

from src.data.tv_loader import load_tv_csv
from src.backtest.engine import run_backtest
from src.strategies.registry import get_strategies_for_symbol

BT = {"initial_equity": 10_000.0, "risk_pct": 0.005, "compounding": True}
_RULE = {"M15": "15min", "H4": "4h"}
_cache: dict = {}


def _resample(src: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = (src.set_index("timestamp").resample(rule, label="left", closed="left")
         .agg({"open": "first", "high": "max", "low": "min", "close": "last",
               "volume": "sum"}).dropna().reset_index())
    return g


def frame(tf: str) -> pd.DataFrame:
    if tf in _cache:
        return _cache[tf]
    if tf in ("M1", "M5", "M30", "H1"):
        df = load_tv_csv(f"data/xau/XAUUSD_{tf}.csv").sort_values("timestamp").reset_index(drop=True)
    elif tf == "M15":
        df = _resample(frame("M5"), "15min")
    elif tf == "H4":
        df = _resample(frame("H1"), "4h")
    else:
        raise ValueError(f"no data for {tf}")
    _cache[tf] = df
    return df


def main() -> int:
    rows = []
    for name, e in get_strategies_for_symbol("XAUUSD").items():
        inst = e["strategy_cls"]()
        try:
            ltf = frame(inst.ltf)
            htfs = {tf: frame(tf) for tf in inst.required_htfs}
            sigs = inst.generate_signals(ltf, htfs).signals
            res = run_backtest(ltf, sigs, "XAUUSD", inst.ltf, BT)
            tr = res["trades"]
            if tr.empty:
                rows.append((name, inst.ltf, 0, float("nan"), float("nan"), 0.0)); continue
            r = tr["R_realized"].dropna()
            rows.append((name, inst.ltf, len(tr), (r > 0).mean(), r.mean(), tr["pnl"].sum()))
        except Exception as ex:
            rows.append((name, inst.ltf, -1, float("nan"), float("nan"), 0.0))
            print(f"  [skip] {name}: {type(ex).__name__}: {str(ex)[:80]}")
    rows.sort(key=lambda x: (x[4] if x[4] == x[4] else -9))   # by meanR desc (NaN last)
    print(f"\n{'strategy':>24} {'tf':>4} {'trades':>7} {'WR':>5} {'meanR':>7} {'net$':>9}")
    for name, tf, n, wr, mr, net in reversed(rows):
        if n <= 0:
            print(f"{name:>24} {tf:>4} {'n/a':>7}")
            continue
        print(f"{name:>24} {tf:>4} {n:>7} {wr:>4.0%} {mr:>+7.3f} {net:>+9.0f}")
    print("\nDEFAULT params (untuned) + real costs, in-sample on fetched window. "
          "Positive meanR here = worth walk-forward tuning next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
