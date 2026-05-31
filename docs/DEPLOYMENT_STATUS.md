# XAU signal bot — deployment status

Live, monitored, single-strategy gold signal bot. Tracking phase (alerts only, no
order execution yet).

## Strategy
- **`ob_fvg_trend`** (`src/strategies/xau/ob_fvg_trend.py`): enter at the overlap of
  a same-direction Order Block ∩ Fair-Value-Gap, trend-aligned (EMA 50/100), fixed
  R:R 3 take-profit, ~1 ATR stop buffer. Long-only on gold (XAU is a one-way bull
  regime; shorts drag — see below).
- **Walk-forward (out-of-sample) validation**, H1, 38 windows:
  - XAU: exp **+0.236R**, PF 1.35, **positive every year** (2023..2026), +70–115% OOS.
  - BTC: marginal (+0.05R) → not deployed.
- Long-only XAU = best book: OOS CAGR ~+28%, MaxDD ~−12% at 1% risk.
- Frequency: ~1 trade / 2 trading days (H1+M30). Low and lumpy — clusters in trends.

## Runtime
- **Host**: Windows VPS (AWS Lightsail, Singapore, 8 GB).
- **Data**: live OHLCV straight from the **Exness MetaTrader 5** terminal
  (`DATA_SOURCE=mt5`). PAXG (Binance) is a fallback/dev source only.
- **Timeframes**: H1 + M30. Poll loop: 60 s.
- **Process layout**:
  - MT5 terminal: launched by a Task Scheduler ONLOGON task.
  - Bot: `bot_startup.bat` in the Startup folder (180 s delay, then
    `run_bot_vps.bat` → `python -m scripts.live_signal_bot --loop 60`), logging to
    `logs\bot.log`.
  - **Autologon** logs Windows in after a reboot → both auto-start → full
    self-recovery (verified).
- Code lives in `C:\mt5-bot` (clone of github.com/rodspep/Copy); DB at
  `data\signals.db`; managed remotely over SSH (key-based).

## Parity contract
- Signal printed at bar *i* close → entered at bar *i+1* open.
- Fixed bracket: SL = zone edge ∓ 1 ATR; TP = entry ± R:R·risk.
- Outcome resolution scans M5 from the **entry** bar onward (not inside the signal
  candle); conservative SL-first within an M5 bar.

## Telegram
- **Signals**: new entry (direction, entry/SL/TP, R:R) + close outcome (win/loss, R).
- **Monitoring**: alert only on error (≥3 consecutive failed passes, with the error
  text) + one recovery notice. No startup/heartbeat spam.
- **Commands** (dedicated long-poll thread, ~1 s reply): `/check` (live health:
  data-read age, last bar, price, DB summary), `/stats` (track record), `/last`
  (recent signals).

## Reviews / tests
- 3 rounds of independent (Codex) review; fixes: outcome-resolution parity,
  fail-fast config validation, whitespace-robust env, MT5 init retry/reconnect,
  DB schema migration + widened dedup `(source,symbol,strategy,timeframe,bar_ts)`.
- `pytest tests/ --ignore=tests/data` → 157 passed.

## Operate
- Health on demand: send `/check` in Telegram.
- Logs: `ssh … "type C:\mt5-bot\logs\bot.log"`.
- Track record: `python -X utf8 -m scripts.live_signal_bot --status`.

## Pending / roadmap
- Harden: restrict RDP (Lightsail firewall) to the operator IP; keep autologon
  password in sync if the Windows password changes.
- Consolidate the separate Binance-futures bot onto this VPS.
- Execution phase: `--execute` (place Exness orders sized by risk %) + a safe-mode
  that pauses + alerts on anomalies — only after the tracked record is trusted.
