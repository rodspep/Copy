# ADR-001 — Backtest ↔ Live execution parity contract

**Status:** Adopted (2026-05-27, revised after codex review round 1)
**Scope:** Every strategy in this repo, every backtest run, every future live wiring (Binance Spot API, MT5/Exness).

## Why this document exists

The single largest source of failure in retail algo trading is **backtest-vs-live divergence**: the backtest assumes one thing about how orders fill, slippage, timing, or candle data, and the live bot does another. Profitable equity curves evaporate the moment real money hits a different fill model than the one the optimizer trained on.

This document is the **single source of truth** for execution semantics. Both `src/backtest/` and any future live execution adapter (`src/live/binance.py`, `src/live/mt5_exness.py`) MUST implement the rules below identically. Deviations require a new ADR.

When in doubt during implementation, re-read this file. Do not "improve" the contract in one place without updating it in the other.

---

## 1. Data conventions

| Concept | Rule |
|---|---|
| Timezone | All timestamps are **UTC tz-aware**. No naive datetimes anywhere. |
| Timestamp meaning | A bar with `timestamp = T` represents the period **[T, T + Δ)**. `open` happens at `T`; `close` is the last trade in the interval; bar **`available_at = T + Δ`**. |
| Bar close moment | A bar is **only considered "closed and available"** at time `available_at = T + Δ`. Indicators evaluated at bar `i` must use only bars `0..i` (no future leak). |
| Raw OHLC basis | **XAU**: midpoint OHLC `(bid+ask)/2` derived from Dukascopy ticks. **BTC**: Binance trade-price OHLC (no bid/ask). The synthetic spread model in §3.3 is applied on top of these raw prices consistently in both backtest and parity reporting. |
| BTC scope | `BTCUSDT` in v1 means **Binance Spot only**. No leverage, liquidation, funding, mark price, or futures order semantics are modeled. Futures requires a new ADR. |
| XAU quantity unit | Backtest `qty` for `XAUUSD` is denominated **in troy ounces**, with `contract_multiplier = 1.0` so `pnl = qty_oz * Δprice`. MT5/Exness adapters MUST convert `qty_oz ↔ broker lots` using the broker's contract size (commonly 100 oz/lot, so `0.01 lot = 1 oz`) **before** order submission, and convert back **before** parity accounting. Submitting `qty=0.5` to MT5 directly would be interpreted as `0.5 lots = 50 oz` — a 100× sizing bug. |
| BTC quantity unit | Backtest `qty` for `BTCUSDT` is in BTC, with `contract_multiplier = 1.0`. Binance REST accepts the same unit directly; no conversion needed. |
| Volume | Resampled bars use sum-of-tick-volume (XAU) or quote-asset volume (BTC). Treated as a feature, never as an order-fill assumption. |
| Gaps | Missing bars (weekends for XAU, exchange downtime for BTC) are **dropped**, not forward-filled. Indicators must be robust to non-uniform spacing. |
| Stale-signal rule | A signal generated at LTF bar `i` (timestamp `T_i`, duration `Δ_l`) is **executable only if the next available LTF row has timestamp exactly `T_i + Δ_l`**. If the next available row is later, the signal is **stale and skipped**. (Friday XAU close ⇒ no Monday-open entry.) Existing open positions remain open and are evaluated on the next available tradable bar under the gap rules in §3.2. |

## 2. Signal evaluation timing

| Concept | Rule |
|---|---|
| Signal evaluation | At `available_at` of LTF bar `i` (= `T_i + Δ_l`). Uses data through bar `i` inclusive. |
| Order placement | Market order submitted immediately. Backtest models the fill at the **open of bar `i+1`** (timestamp `T_i + Δ_l`). |
| Live note | Live code cannot literally force a fill at a historical candle-open price. The adapter submits the market order immediately after bar `i` is confirmed closed and records the **actual** broker/exchange fill. Backtest parity approximates this with bar `i+1` open plus configured slippage (§3.1). Reconciliation reports the realized slippage vs the modeled slippage; if realized slippage drifts > 1.5× modeled for 20+ trades, raise an alert. |
| No same-bar entry | A strategy never enters on the same bar whose close it just evaluated. If faster reaction is needed, drop to a lower timeframe. |

## 3. Fill model

### 3.1 Entry fill

- **Type:** Market order; in backtest, fills at the **open of bar `i+1`**, with synthetic spread + slippage baked into the recorded entry price (see 3.3).
- **Synthetic entry price** (backtest, both for accounting and for SL/TP comparison):
  - Long:  `entry_price = open[i+1] + (spread_pips + slippage_pips) * pip`
  - Short: `entry_price = open[i+1] − (spread_pips + slippage_pips) * pip`
- **Live:** trades real bid/ask; the adapter normalizes the realized fill to the same synthetic-entry-price convention before any PnL/R comparison so that backtest and live equity curves are apples-to-apples.
- No partial fills (see 3.6). Position size is filled in full at this single price.
- **Price rounding:**
  - SL/TP: long SL rounds **down**, long TP rounds **down**; short SL rounds **up**, short TP rounds **up** (always make the price the broker will accept *without* tightening risk).
  - **Entry: long rounds UP, short rounds DOWN** — i.e. *adverse* to the position, by one `min_tick` at most. Rationale: the backtest cannot know the broker's exact tick-grid fill, so it conservatively records an entry price no better than the live bot will plausibly achieve. Live adapters do NOT round the realized fill (they accept whatever the broker reports); instead, the parity reporting layer normalizes both sides to the same tick grid before comparing PnL. Expected net effect: one tick of pessimistic bias on entries in backtest, which is the safer direction for go/no-go decisions.

### 3.2 Stop loss / take profit (intra-bar)

- **Type:** Resting stop (SL) and limit (TP), both placed conceptually at entry-fill time of the entry order.
- **Validity check after fill:** once `entry_price` is known, the engine asserts:
  - Long:  `sl < entry_price < tp`
  - Short: `tp < entry_price < sl`

  If the assertion fails (e.g. an entry-bar gap moved price past a planned SL or TP), the entry is **canceled and no position is opened**. The strategy may re-emit a signal on a later bar.
- **Evaluation window:** SL/TP are evaluated **starting with the entry bar itself** (bar `i+1` in the engine's terms — the same bar in which the entry fills). This is essential for M1/M5 scalp realism. Subsequent bars are evaluated until one fills.
- **Per-bar fill rule** (applied to each bar's `[low, high]` range):
  - **Adverse open gap (long: `open ≤ sl_price`; short: `open ≥ sl_price`):** SL fills at `open − side * slippage_pips * pip` (i.e. worse than the stop). This captures weekend/news gaps.
  - **Favorable open gap (long: `open ≥ tp_price`; short: `open ≤ tp_price`):** TP fills at `tp_price` exactly (never better — limit orders do not get positive slippage in this model).
  - **Otherwise, if only SL is touched intra-bar:** SL fills at `sl_price − side * slippage_pips * pip`.
  - **Otherwise, if only TP is touched intra-bar:** TP fills at `tp_price` exactly.
  - **If both SL and TP fall inside the same bar's range:** assume **SL fills first** (pessimistic / worst-case). This is deliberate; it removes optimistic bias from the equity curve.
  - **If neither is touched:** position carries to next bar.
- **Sign convention for SL slippage:** `sl_exit_price = sl_price − side * slippage_pips * pip`, where `side = +1` long, `−1` short. Long SLs fill **below** the stop; short SLs fill **above** the stop.
- **No trailing stops in v1.** Adding them requires a new ADR; the live adapter must replicate exactly.

### 3.3 Spread (synthetic accounting model)

Backtest accounting uses a **synthetic all-in entry price** and **raw exit prices**.

- For longs: `entry_price = open + (spread_pips + slippage_pips) * pip` (see 3.1).
- For shorts: `entry_price = open − (spread_pips + slippage_pips) * pip`.
- **No additional spread is applied on exit** in the backtest — the full round-trip spread cost is loaded onto the entry.
- Live parity reporting must store **both** the actual broker fill prices **and** normalized parity prices. Normalized parity prices follow the same fields the backtest uses: synthetic entry under §3.1/§3.3, exit price under §3.2 / §3.4 / §3.7. Actual fills remain available for slippage-realized-vs-modeled diagnostics; they never feed PnL/R numbers used for equity comparison.
- Per-symbol values live in `src/config.py::SYMBOLS`. The live adapter reads the **same dict**, does not hardcode.

### 3.4 Slippage

- **Entry market orders:** apply `slippage_pips` (folded into synthetic entry price in 3.1).
- **SL fills (stop orders):** apply `slippage_pips` in the adverse direction (sign in 3.2).
- **TP fills (limit orders):** zero slippage. Limit orders generally do not get adverse slippage; if a TP gaps, it still fills at limit price (no improvement).

### 3.5 Commission

- Applied on **every executed order fill** (entry **and** exit), not entry-only.
- Percentage fee (BTC):  `commission = abs(fill_price * qty * contract_multiplier) * commission_pct` at entry, again at exit.
- Fixed/per-lot fee (XAU on most ECN brokers): apply per the symbol's fee model in `SYMBOLS`. If `commission_usd = 0` (default for spreads-only brokers like Exness Standard), no charge.
- Live adapter must use the **same numbers** from `SYMBOLS`. If the broker's actual fee differs, update `SYMBOLS` — never branch the math.

### 3.6 What the backtest does NOT model (yet)

These are known divergence risks. The live adapter MUST account for them with conservative real-world buffers; if any becomes material, a follow-up ADR adds them to backtest too.

- **Order book depth / market impact** — assumed infinite liquidity at the quoted price for scalp sizes.
- **Latency** — assumed zero. In live, expect 50–300ms broker round-trip; conservatively model by adding to `slippage_pips`.
- **Partial fills** — assumed never. For BTC scalp sizes this is realistic; for very large XAU sizes it is not.
- **Broker rejections / requotes** — assumed never. Live adapter must retry with bounded backoff and **abandon** the trade if not filled within N seconds (no chasing).
- **Funding rates / swap** — not modeled. Intraday scalp positions close same day; if a strategy holds overnight, add an ADR.
- **News-driven spread widening** — partially captured via `spread_pips`. Strategies should have a news filter (TODO in [`docs/research/news_filter.md`](../research/news_filter.md) when added).
- **Variable spread by liquidity** — currently assumes constant `spread_pips` per symbol.

### 3.6.1 Force-EOD close (backtest-only)

- At the end of a backtest run, any still-open position is **force-closed** at the last bar's `close` price, treated as a synthetic market exit: slippage applied, **no synthetic spread** (consistent with §3.7's manual-exit convention), commission applied on exit per §3.5.
- Recorded with `exit_reason = "force_eod"` so reports can optionally exclude these trades from parity stats (they have no live counterpart).
- The live adapter never performs this operation — live runs continuously until shut down, at which point any open position remains at the broker.
- Strategies should generally aim to exit positions before any expected data boundary; force-EOD is a safety net for clean accounting, not a strategy primitive.

### 3.7 Manual strategy exits

- A strategy may emit `action = 'exit'` at LTF bar close `i`. The exit is a **market order at the open of bar `i+1`**.
- **Exit price:** `exit_price = open[i+1] − side * slippage_pips * pip` (slippage only). **No synthetic spread is applied on strategy exits** — §3.3 already loaded the full round-trip spread onto entry; adding it again here would double-charge.
- **Commission** still applies on the exit fill (§3.5).
- **Precedence within bar `i+1`:** the strategy exit fills **at the open of bar `i+1` before any intra-bar SL/TP evaluation for that bar**. Once the market exit fills, the position is closed and the resting SL/TP are canceled. This ordering is intentional — letting the engine peek at the bar's `[low, high]` range *first* would constitute lookahead (the strategy committed to the market exit at bar `i` close, before bar `i+1` was observable).

### 3.8 Connectivity / restart / downtime (live only)

- **Restart:** On startup, the live adapter MUST reconstruct open position state from broker/exchange positions **and** locally persisted strategy state before evaluating new signals. If the two cannot be reconciled exactly, the bot refuses to trade that symbol/strategy until manually resolved.
- **Data or order connectivity loss:** no new entries opened during loss. Existing positions keep their broker-side SL/TP orders active where supported. After reconnect, the adapter reconciles fills first, then resumes signal evaluation **only from fully closed bars**.

## 4. Position sizing

| Concept | Rule |
|---|---|
| Risk per trade | Fixed fraction of equity, in account currency. Default 0.5% (configurable per strategy in `configs/strategies/<name>.yaml`). |
| Size formula | `qty = risk_amount / (stop_distance_in_price * contract_multiplier)`, where `risk_amount = equity * risk_pct`, `stop_distance = abs(entry_price − sl_price)`, and `contract_multiplier` is the per-symbol PnL-per-1.0-price-move-per-unit-qty (1 for BTC spot ounces / base units, 100 for XAU on MT5 standard lots, etc.). |
| Quantity rounding | Round `qty` **down** to the symbol's `qty_step`. Never round up. |
| Min size | If rounded `qty < min_qty`, **skip the trade**. Do not enlarge to meet minimum (would blow the risk model). |
| Price rounding | Done independently (see 3.1, directions favor broker acceptance over tighter risk). |
| Leverage | Sizing is independent of leverage. Leverage is a margin-availability concern only. |
| Equity update | After each closed trade, `equity += pnl`. **Compounding on by default**; configurable to fixed-equity mode for parameter robustness checks. |
| What R means | `risk_amount` is the **intended pre-fee, pre-SL-slippage** risk budget used for sizing. **Realized stop losses can exceed −1R** after spread, commission, SL slippage, or gap-through-stop fills. Reports must show both pre- and post-cost R-multiples. |

## 5. Multi-timeframe alignment

| Concept | Rule |
|---|---|
| HTF availability | For each HTF bar with open timestamp `T_h` and duration `Δ_h`, define `available_at = T_h + Δ_h`. |
| LTF↔HTF join | Use `merge_asof(left_on='signal_available_at', right_on='available_at', direction='backward', allow_exact_matches=True)`, where `signal_available_at = T_i + Δ_l` for LTF bar `i`. **Never** merge LTF rows directly against HTF bar-open timestamps — that leaks the in-progress HTF bar. |
| Live equivalent | The live adapter calls `get_klines(htf, limit=N+1)` and **drops the last (in-progress) bar** before computing HTF indicators. |

## 6. Strategy API contract

A strategy is a pure function:

```
signal(t) = f(ohlcv_ltf[0..t], ohlcv_htf[0..t (closed only)], indicators[0..t], open_position_state)
        -> {action: 'enter_long' | 'enter_short' | 'exit' | 'hold',
            sl: float | None,
            tp: float | None,
            meta: dict}
```

- `f` must be **deterministic** given the same inputs.
- `f` must not consult `ohlcv_ltf[t+1..]` or any future-leaking source.
- `f` is called at `available_at = T_t + Δ_l` of bar `t`; the action takes effect at bar `t+1`'s open under §3.
- **At most one position at a time per symbol per strategy.** No pyramiding in v1.
- If a strategy emits `enter_*` while a position is open, the new signal is **logged and ignored** (not actioned).

## 7. PnL accounting

- PnL per closed trade, in account currency:

  ```
  pnl = side * qty * contract_multiplier * (exit_price − entry_price) − commission_entry − commission_exit
  ```

  where `entry_price` includes synthetic spread+slippage (§3.1, §3.3) and `exit_price` includes SL slippage where applicable (§3.2, §3.4) or strategy-exit slippage (§3.7).
- Returns expressed in **R** (multiples of risked amount): `R_realized = pnl / risk_amount`. Note R can be < −1 (see §4 "What R means").
- Equity curve is the running cumulative sum of `pnl`, starting from `initial_equity` (default 10,000 USD; configurable).
- All currency math is **float64**. No `Decimal`. Acceptable for backtest precision; live adapter may upgrade to `Decimal` for order-size and price rounding only.

## 8. Reproducibility

- Every backtest run records: git commit hash, config YAML, full `SYMBOLS` dict, data file checksums, library versions (`pip freeze`), **SHA-256 of this file (`backtest_live_parity.md`)**. Output to `results/<timestamp>/manifest.json`.
- Live runs record the same manifest at startup. If the live adapter's parity-doc hash differs from the most recent backtest's, **the live bot refuses to start** (manual override requires `--ack-parity-drift <reason>`).

## 9. Forbidden patterns (will fail review)

- Reading `df['close'][i]` and entering at `df['open'][i]` (lookahead via same-bar close).
- Using `pd.Series.ewm(...)` (or any indicator with implicit warmup) without explicit `min_periods` / warmup handling — produces unstable early-sample values that diverge live vs backtest.
- Computing HTF indicator on full series then slicing — leaks future HTF bars.
- Merging HTF data on HTF bar-open timestamps instead of HTF `available_at` timestamps.
- Using the current still-forming live candle for signal generation.
- Forward-filling missing OHLC bars, or treating missing bars as flat candles.
- Entering after a missing-bar gap using a stale signal from the previous session.
- Rounding **quantity up** to satisfy broker minimums.
- Applying commission only in backtest, or only in live.
- Charging commission only on entry (must be both legs).
- Computing SL/TP from bar `i+1` data when the signal was generated from bar `i`.
- Sizing position from "available margin" — couples sizing to broker state. Use §4.
- Adjusting SL after entry without an ADR. (Trailing stops included.)
- Any branch like `if is_backtest: ... else: ...` in strategy code. Strategy is the same in both.
- Running live with unreconciled broker positions, open orders, or local strategy state.

## 10. Open questions / future ADRs

- [ ] Partial-fill modeling for large XAU sizes.
- [ ] News filter integration (Forex Factory calendar?).
- [ ] Funding-rate handling for BTC perpetual futures (if Binance Futures is added).
- [ ] Variable-spread modeling for low-liquidity hours (currently assumes constant).
- [ ] Tick-level backtest mode (currently bar-only).
- [ ] Trailing-stop semantics (currently forbidden).

---

**When you change anything here, you change the contract for the entire system. Always (a) update the backtest engine, (b) update or note the live adapter, (c) the SHA-256 hash check in §8 will trip and force you to acknowledge.**
