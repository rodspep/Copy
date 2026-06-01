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

## UG's own description of the two families (from channel admin posts)
- **Phương Pháp 2 / Scalp Signal** (template `{50,100,150,200}`): "đớp 5 giá là
  chuẩn; vào muộn gần cạnh trên thì gồng 10-15 giá tới TP1." Claimed hit-rate:
  **TP1 ~95%**, TP2 60%, TP3 40%.
- **PRI GOLD SLCAP NOMAL** (template `{150,200,300,400}`): "thường ăn 10-20 giá
  (~85%); 30-60 giá (65%); chốt 50% ở target đầu, để 50%."
These are UG's *claimed* win-rates — use them as ground-truth checks when we
backtest the reconstructed rule.

## Signal geometry is DETERMINISTIC (verified 93/93) — only entry varies
Per the channel and confirmed in data: **SL and TP are fixed per method; only the
entry price changes.** Exactly:
- **SL = entry_B ∓ 10.0 price (100 pip)** where entry_B is the 2nd number of the
  "Entry: A - B" range. `sl − entry_B = −10` for 40/40 longs, `+10` for 53/53
  shorts. Zero exceptions.
- **TP = fixed pip template per method** ({50,100,150,200} scalp / {150,200,300,400}
  / {100,200,300}).
- **Entry-range width fixed per method**: 3 price (scalp) or 10 price (wide).
- Range orientation encodes direction: long written high→low, short low→high.
So once (method, direction, entry-anchor) are known, SL/TP/range follow by formula.
**The only free variable to reverse-engineer is the entry-anchor price** (+ method
selection + direction).

## Caveat: hidden features
The displayed analysis (MA34/89, Elliott, SMC, risk) may be NECESSARY-not-sufficient
or partly cosmetic — the true entry/direction/method choice may use hidden inputs.
Plan: with OHLC, MEASURE how much the visible features explain (method, direction,
entry-anchor); treat the residual as quantified hidden logic rather than claiming
100% from text. The geometry above is the part that IS 100% deterministic.

## Open questions (need next)
- Exact ENTRY construction (the range vs current price / MA34 offset).
- What selects the TP template precisely (ATR/volatility? Elliott? risk score?).
- The remaining ~9%: role of Elliott Wave + SMC + risk score as modifiers.
- **Verify UG's MA34/89 against real chart data** (TradingView OANDA:XAUUSD per
  `tv_loader`, or MT5/Exness): EMA vs SMA? which feed? If we can reproduce UG's
  stated MA values, we can recompute its inputs live and replicate the rule.

## Verified against MT5 OHLC (data/xau/XAUUSD_{M5,M30,H1}.csv)
- **UG uses SMA34/SMA89, NOT EMA.** Recomputed-vs-stated median |err|: M5 SMA
  0.25–0.31 (EMA 1.5–1.8), H1 SMA89 0.29 (EMA 8.75). SMA wins on every TF.
- **Feed matches Exness MT5 closely** (M5 |err| ~0.25 price) → we can reproduce
  UG's MA inputs from MT5 data, hence reproduce its decision inputs. (M30/H1 SMA34
  residual ~1–1.6 — likely a one-bar/forming-bar or micro-feed offset; refine.)
- **Entry = LIMIT order placed ~3–7 price from CURRENT price, in the fade
  direction** (long below, short above): long entry−close median −4.8 (IQR
  −6.4..−3.0), short +4.4 (IQR +2.8..+7.0). NOT anchored to SMA34. The ~3–7 offset
  still varies (candidate: ATR/volatility-scaled) — to refine.

## Direction is NOT fully explainable from visible/computable features (n=52)
Decision-tree, 5-fold CV on 52 unique signals (features RECOMPUTED from MT5, i.e.
reproducible live):
- SMA stack only: CV **69%±14%**
- + stated Elliott/SMC cues: **69%** (no lift → the Elliott/SMC narrative is
  post-hoc decoration, not a direction trigger)
- + price-action (RSI, Bollinger): train 88% but CV **69%** (overfit; no real lift)
- baseline always-short = 58%
**Conclusion: ~69% is the out-of-sample ceiling for DIRECTION from everything UG
shows + everything we can compute. ~31% is irreducible-from-visible** — likely an
opaque AI call or finer (tick/M1) price action, OR just sample-size noise (only
~1 week of data). Do NOT claim 100% direction logic.

## What IS reproducible (the decoded, deterministic/mechanical core)
- **Indicators:** SMA34/SMA89 on M5/M15/M30/H1 (feed matches Exness MT5 ~0.25px).
- **Entry:** LIMIT ~**1×ATR(M5)** from current price, in the fade direction
  (median 0.90 ATR).
- **Geometry:** SL = entry_B ∓10; TP = fixed pip template per method; range width
  fixed per method. (100%.)
- **Direction:** ~69% predictable — the one genuinely uncertain piece.

## Reproduction backtest (corrected) + S/R entry + timing
- **Crude clone is ~breakeven, NOT UG's edge.** A fade scalp (direction = fade vs
  M5 SMA34, entry = limit ~1×ATR from price, TP1=5/SL=10) on 30k M5 bars: after
  fixing a fill-bar lookahead (Codex — don't score TP/SL on the fill bar itself),
  TP1-before-SL is only **~67–71%**, expectancy **~0..+0.07R**. So geometry +
  naive direction does NOT reproduce UG → **direction/entry-quality is the edge.**
- **Entry IS anchored to S/R (swing highs/lows) — confirmed.** UG scalp entries sit
  median **0.51×ATR** from the nearest M5/M30/H1 swing (72% within 1 ATR) vs a
  random-bar **baseline of 1.13×ATR** (44%). Significantly closer → the entry is a
  fade AT a real level, not a fixed offset. (Wide/PRI-GOLD weaker: 1.12 ATR.)
- **Detect→post delay is empirically small here.** Using UG's stated MA as a clock
  (find the bar whose recomputed SMA best matches), the message-ts bar is the best
  match 98% of the time (median delay 0 min; one outlier 40 min). UG admits late
  posts ("Telegram post chậm") but it doesn't materially confound this sample. The
  ~0.3/MA residual is feed/computation micro-difference, not latency.

## S/R-fade reproduction + honest bottom line
- Fading every swing-level retest (SHORT at last swing high, LONG at last swing
  low; TP1=5/SL=10) is **TP1 62–64%, expectancy NEGATIVE (−0.04..−0.07R)** — even
  worse than the ATR-fade. So entering AT S/R spatially is real, but **fading
  levels indiscriminately loses**: the edge is in SELECTING which level to fade
  (regime/direction filter), i.e. the opaque AI call we can't recover from outputs.
- **Both mechanical clones (ATR-fade, S/R-fade) are ~breakeven; neither nears the
  claimed TP1 95%.** A genuine tight-TP fade scalp is ~62–71% TP1 on 0.5R:1R
  payoff. UG's 95% is therefore either measured generously (e.g. TP1 touched
  ever), marketing, or driven by a level-selection edge that is not in the signal
  text. Going from ~67% to 95% via direction alone is implausible for a 0.5R TP.

## THE EDGE FOUND: S/R fade + rejection-candle CONFIRMATION
A real winning UG trade (user, 2026-06-01) revealed the true mechanic: the Entry
is a ZONE; you WAIT for price to pull back into it and **confirm**, then enter,
and TP1 is measured FROM your actual entry (not a fixed level). Modelling that as
"fade at a swing level only after a rejection close" (wick into the level, close
back across it) flips both methods from breakeven to clearly +EV (30k M5 bars,
conservative ties, non-overlapping):
- scalp (TP=5/SL=10, RR0.5): no-confirm WR 62% (−0.07R) → **confirm WR 80% (+0.21R)**
- PRI-GOLD (TP=15/SL=10, RR1.5): no-confirm 41% (+0.01R) → **confirm 55% (+0.37R)**
Adding an H1-trend filter on top does NOT help → the **rejection confirmation is
the edge**, not the regime. This matches UG's high claimed WR far better and is
exactly the user's discretionary "wait for the chart to confirm" step.

CAVEATS (before trusting): in-sample only; entry assumed at the level after a
confirmation known at bar close (slightly optimistic — realistic fill ≈ next-bar
open); no spread/slippage/commission (matters for a 5-price TP on XAU). Needs
walk-forward + realistic fills + costs via the repo's backtest engine.

## BOTTOM LINE (with current data, n=52 / 1 week)
Fully decoded & reproducible: SMA34/89 multi-TF inputs (feed≈Exness), entry LOCATION
(at S/R, ~0.5×ATR from swings), deterministic SL/TP/range geometry, fixed pip
templates per method. NOT recoverable from outputs: the level-SELECTION / direction
decision (~31% of direction unexplained; both naive clones breakeven) — almost
certainly the bot's opaque "AI" call. The fastest path to the exact remaining logic
is the bot's source/spec (it is reportedly Claude-built), not further black-box
inference.

## Next steps
- **Biggest lever: MORE DATA.** The export was ~1 week (52 unique). A longer
  history / live capture would tighten the CV and may lift the direction ceiling.
- **Reproduction backtest:** build a UG-clone from the solid parts (SMA regime +
  ~1×ATR fade limit + fixed geometry), backtest on MT5, compare WR to UG's claimed
  numbers (TP1 95% scalp). Tests whether the reproducible mechanics alone perform
  like UG — if so, the hidden 31% may not matter for profitability.
- Dedup + rule-fit script (formalize the hybrid rule as a tiny decision model).
- Run `scripts/analyze_ug.py --signals data/ug/signals.jsonl --tv|--parquet ...`
  once OHLC is available → verify MA feed + add the indicator/structure context
  features at each signal bar.

Related: `src/strategies/xau/ug_methods.py` (prior hypotheses: PP2 scalp fade +
deep pullback), `src/strategies/xau/ma34_cascade.py`.
