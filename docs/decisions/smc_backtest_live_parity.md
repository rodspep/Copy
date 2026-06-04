# SMC bot — backtest ↔ live parity contract

The standalone SMC bot (`scripts/smc_bot.py`) and the backtest that justified it
(`scripts/smc_sizing.py` → `optimize.smc_trades` path) MUST stay behaviourally
identical. This file is the written contract; `tests/test_smc_logic.py` pins it.

## Idealized vs REALIZABLE (read this first)
`smc_sizing.smc_legged` (+$5562 / maxDD −$644 / WR 32% / 519 trades) uses **look-ahead
fill detection** — it scans forward from the signal bar to find the retest. That is NOT
realizable live (you can't know the future). The authoritative number for deployment is
`scripts/smc_live_sim.py` — a faithful live-state-machine sim (place a pending on a closed
bar, fill only when price returns, cap concurrent setups, no look-ahead):

| model | #tr | net | WR | maxDD | MAR |
|---|---|---|---|---|---|
| idealized smc_legged (look-ahead) | 519 | +$5562 | 32% | −$644 | 8.6 — **not realizable** |
| one-pending gate (too conservative) | 418 | +$2449 | 27% | −$583 | 4.2 |
| **no-gate, cap 4 concurrent (DEPLOYED)** | 683 | **+$5019** | 28% | **−$915** | **5.5** |

The deployed bot = **no-gate, `--max-setups 4`**: place every fresh closed-bar setup as a
pending; up to 4 setups (pending+filled) live at once; this captures the edge a one-at-a-time
gate throws away. maxDD is correspondingly deeper (−$915, real concurrent exposure).

## Single source of truth
`src/exec/smc_logic.py` holds the detection (`swings`, `htf_trend`, `detect`,
`build_setup`). Both the backtest (`smc_sizing.smc_legged` calls `smc_logic.detect`)
and the live bot (`smc_bot` calls `smc_logic.decide`) use it. Tests assert:
- `smc_logic.swings`/`htf_trend` are byte-identical to `smc_replica.swings` / `optimize.htf_trend`.
- a `smc_logic.detect`-driven backtest reproduces the idealized **519 / +$5562 / −$644**.
- `scripts/smc_live_sim.py` (no-gate cap 4) reproduces the realizable **~+$5019 / −$915 / WR 28%**.
- `decide(window)` == `detect()` at the window's last bar.

## Frozen parameters (DO NOT change without a re-backtest + re-review)
| param | value | meaning |
|---|---|---|
| `W` | 2 | fractal swing half-window (pivot confirmed W bars later) |
| `SWEEP_WIN` | 8 | sweep + order-block lookback |
| `SL_BUF` | 2.0 | price beyond the OB edge |
| HTF filter | H1 EMA50 sign | only trade with the H1 trend |
| entry | 50% of OB→sweep range | OB-retest **limit** |
| `R_MAX` | 25.0 | reject setups with stop > 25 price pts |
| `RETEST_BARS` | 24 (=360 min) | pending lifetime; cancel if unfilled |
| `HORIZON` | 96 (=1440 min) | filled-trade time-stop (market close) |
| legs | 0.01 @ +4R (`tp1`) + 0.01 @ +10R (`tp3`) | 2-leg 50/50 |
| BE | after the near (4R) leg books | runner SL → entry |

## Lifecycle mapping
| live-sim (`smc_live_sim`) | live bot (`smc_bot`) |
|---|---|
| detect at each bar `i` if < cap concurrent | `decide()` on the **last CLOSED** M15 bar, once per new bar (`state.last_bar`), if `groups < --max-setups` |
| place pending; fill when a later bar touches `entry` | broker fills the resting limit; `fill_info` detects it |
| cancel pending after `RETEST_BARS` if never hit | pending **cancelled** after `expiry_min` |
| SL-first per bar; book legs at their TP | broker SL/TP fill the legs; `closed_info` records P/L |
| stop → entry after the near (4R) leg books | `modify_sl(runner, entry)` after the near leg `closed_tp` (genuine TP only) |
| close remaining at bar `fill+HORIZON` | `close_position` once a filled trade exceeds `horizon_min` |

## Orphan recovery (crash between order_send and DB insert)
Each leg's MT5 order **comment** is `smc<base36(epoch)><L|S>-tp1|tp3` (the same short id used
as `group_id`). `_adopt_orphans` parses it → relinks an orphaned leg to its EXACT bracket
(works even if the near already closed or other setups run concurrently; `siblings()` includes
closed rows so SL→BE still fires). Unrecognised comment → bounded standalone orphan. Real MT5
setup/fill times are used so a stale orphan expires/horizon-closes on the next pass. A periodic
sweep (every ~30 polls) catches orphans between restarts.

## Known, accepted gaps (small; documented so they're not surprises)
1. **Bar timestamps** are broker server time live vs CSV-UTC in backtest. Detection is
   tz-relative (swings/sweep/CHOCH/H1-resample), so this shifts labels, not decisions.
   No session/skip-hours filter is used (the vol-filter study showed filters don't help SMC).
2. **Fill model**: live-sim fills the limit when a later bar's range touches `entry`; live
   fills at the broker's actual touch (can be marginally better/worse). Same anchor, ±spread.
3. **Near-only degraded mode**: if the runner leg's order fails after the near landed, the
   bot KEEPS the bounded near-only trade (its own SL/TP) and warns — it does not cancel the
   near. This is strictly LESS exposure than the full bracket (rare; transient 2nd-order glitch).
4. **Concurrency cap**: `--max-setups 4` matches the cap the realizable sim was measured at.
   In a rare clustered-setup burst beyond 4, extra setups are skipped (logged), not queued.
5. **BE latency**: the sim moves the runner's stop to BE the instant the near leg books +4R.
   Live applies BE on the NEXT poll (after `closed_info` sees the near close, then `modify_sl`).
   If price hits +4R then fully reverses to the runner's original SL within ~1 poll (a ~5R whip
   in `--poll` seconds — extreme/rare), the runner can take its full loss instead of scratching
   at BE. `--poll 20` keeps the window small; a tick-level EA would remove it (out of scope v1).
6. **Hedging required**: the bot fails closed at startup on a non-hedging (netting) account —
   the 2-leg model needs independent per-leg positions. Fund SMC on a HEDGING account.

## Sizing / capital (REALIZABLE — see scripts/smc_live_sim.py, no-gate cap 4)
2-leg 50/50 (4R+10R) @ **0.02 lot/signal**, up to 4 concurrent setups: net **+$5019** over
18 months, WR 28%, **maxDD −$915** (real concurrent exposure). Run on a **separate account**:
- **$4500–5000** → maxDD ≈ 18–20% ✅ recommended
- **$3700** → maxDD ≈ 25% (minimum sane)
- below ~$3700 → **do not run SMC**; the deep, long (≈100+ day) drawdown will blow it up.

NOTE the earlier $2500–3000 figure was based on the idealized −$644 maxDD; the realizable
maxDD is −$915 (concurrency), so capital must be higher. Magic **770820** (≠ copier 770150)
→ full order isolation; ledger `data/smc_trades[_real_<login>].db`; telegram
`configs/smc_telegram.json`. The copier and SMC are independent processes/accounts.
