# UG bot logic — decode progress

Source: Telegram Desktop export of "XAUUSD AI Sclaping UG" (Trade Coin Underground
AI), 25 May–1 Jun 2026. Parsed by `scripts/parse_ug_export.py` →
`data/ug/signals.jsonl` (gitignored). 93 signals raw, **56 unique** after
de-duplicating reposts (UG often posts the same signal ~1 min apart).

UG **self-documents** its analysis in every message, so the decode is grounded in
UG's own stated inputs (not inferred from price alone). External OHLC is only
needed to VERIFY those inputs are reproducible.

## What each signal states
- Direction: `BUY`/`SELL XAUUSD`, entry **range**, `SL: <price> (10.0 gia)`,
  `TP1..TP4` in **pip**.
- `MA34/MA89` per timeframe (M5/M15/M30/H1) with values + a ⬆/⬇ arrow.
- Elliott Wave (H1/H4/D1 wave bias), SMC (liquidity sweep / CHOCH / OB), a risk
  score `/10`, and a recommendation (CAUTION/FOLLOW).

## Decoded so far (confidence)
1. **⬆/⬇ arrow per TF == (MA34 > MA89)** — 372/372 = **100%**. The arrow is just
   the MA34/89 cross on that timeframe.
2. **SL is FIXED = 10.0 price (100 pip)** from entry — stated as "(10.0 gia)" on
   every signal.
3. **TP templates** (pip): `{50,100,150,200}` (dominant, 71×), `{150,200,300,400}`
   (wider, 16×), `{100,200,300}` (6×). With SL=10.0 → TP1 R:R ∈ {0.5, 1.0, 1.5}.
   Wider templates cluster in **trend-aligned** regimes; the tight `{50,100,150,200}`
   dominates **mixed** regimes (trend → let it run; chop → tight scalp).
4. **Entry side = mean-reversion vs M5 MA34 (fade), by default:** long when entry
   ≤ M5 MA34, short when entry ≥ M5 MA34. Holds 75/93 raw.
5. **Hybrid rule** — fade by default, but **trend-follow when the multi-TF stack
   is ≥3/4 aligned** (e.g. DDDD/DDDU → short even below MA34): covers **51/56 =
   91%** of unique signals. The exceptions' stacks are strongly aligned, i.e.
   genuine trend-continuation, not noise.

## Open questions (need next)
- Exact ENTRY construction (the range vs current price / MA34 offset).
- What selects the TP template precisely (ATR/volatility? Elliott? risk score?).
- The remaining ~9%: role of Elliott Wave + SMC + risk score as modifiers.
- **Verify UG's MA34/89 against real chart data** (TradingView OANDA:XAUUSD per
  `tv_loader`, or MT5/Exness): EMA vs SMA? which feed? If we can reproduce UG's
  stated MA values, we can recompute its inputs live and replicate the rule.

## Next steps
- Dedup + rule-fit script (formalize the hybrid rule as a tiny decision model).
- Run `scripts/analyze_ug.py --signals data/ug/signals.jsonl --tv|--parquet ...`
  once OHLC is available → verify MA feed + add the indicator/structure context
  features at each signal bar.

Related: `src/strategies/xau/ug_methods.py` (prior hypotheses: PP2 scalp fade +
deep pullback), `src/strategies/xau/ma34_cascade.py`.
