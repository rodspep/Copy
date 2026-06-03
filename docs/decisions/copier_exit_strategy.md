# UG Copier — exit strategy decisions & watch-list

Tracks how the live copier exits trades, the backtest evidence, and candidate changes
to revisit AFTER the current forward-test validates out-of-sample.

All P/L below = `scripts/ug_exit_strategy.py` on the collected UG signals, **in-sample
~1 week (2026-05-26 → 06-01), M1 XAU, deep-limit fill, 0.01 lot/leg, 3pip cost**,
Codex-reviewed (fill-bar TP-suppression artifact fixed). Numbers are DIRECTIONAL, not
gospel — 1 week, small n.

## Live config (current, forward-testing on demo)
- **Placement: DEEP-LIMIT** (`DEEP_LIMIT=True`) — place a pull-back limit when price is
  on the fillable side of entry; wait for the pull-back; do NOT chase-skip at TP1.
  (Backtest: 50pip 38%→83% fill, +$30.90→+$115.30 in-sample. Forward-testing.)
- **Exit: 50% TP1 + 50% runner to TP3, SL→BE after TP1.** = **+$142.80** (+0.232R), n=59.
- Methods: 50pip traded for real-money later; 100/150 demo observe-only.

## 50pip exit comparison (deep-limit fill, n=59, in-sample)
| strategy | net $ | meanR |
|---|---|---|
| TP1 full | +115.30 | +0.191 |
| **TP2 full (100%)** | **+165.80** | **+0.268** |
| TP3 full | +47.80 | +0.094 |
| TP4 full | +61.80 | +0.114 |
| 50% TP1 / 50% TP2 +BE | +137.80 | +0.224 |
| **50% TP1 / 50% TP3 +BE (LIVE)** | +142.80 | +0.232 |
| 50% TP1 / 50% TP4 +BE | +152.80 | +0.247 |
| 1/4 TP1/2/3/4 +BE | +159.05 | +0.256 |
| 50% TP1 / 50% trail100 | +119.33 | +0.197 |

Key insight: once the runner's SL is at break-even it cannot lose, so a FURTHER runner
target captures more upside at zero extra risk → runner-TP2 (+137.80) < runner-TP3
(+142.80) < runner-TP4 (+152.80). Do NOT shorten the runner.

## WATCH-LIST (revisit after deep-limit is validated out-of-sample)
1. **TP2-full (100% to TP2)** — best single-target in-sample (+$165.80) AND simplest (one
   order, no bracket/BE). User flagged it as RISKY on short data (single target, all-in,
   1-week sample). **Monitor**: once we have more demo/forward data, compare TP2-full vs
   the 50/50 runner. If it holds up, it could replace the bracket entirely (simpler).
2. **Runner → TP4** (+$152.80 vs TP3 +$142.80): minor in-sample improvement, within noise.
   Reconsider together with #1 after more data.
3. **1/4 ladder TP1/2/3/4 +BE** (+$159.05): close to TP4-runner; more partial closes
   (more spread cost in reality). Lower priority.

## Decision (2026-06-03)
Keep the LIVE config (deep-limit + 50% TP1 / 50% TP3 + BE) UNCHANGED while forward-testing
deep-limit on demo. The exit-target refinements above are all within a $137–166 in-sample
band over one week — do NOT optimize exits until deep-limit itself is validated on
out-of-sample (forward) data with a larger sample. Then revisit #1 (TP2-full) first.
