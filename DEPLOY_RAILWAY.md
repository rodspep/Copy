# Deploy the XAU signal bot on Railway

The bot is a **worker** (no web port): it polls fresh gold price every 15 min,
sends Telegram signals for `ob_fvg_trend` (H1 + M30, long-only), and logs every
signal + its TP/SL outcome to a SQLite DB on a persistent volume.

## 1. Create the service
1. Railway → **New Project → Deploy from GitHub repo** → pick this repo.
2. Railway auto-detects Python (Nixpacks) and uses `railway.json` /`Procfile`
   start command: `python -X utf8 -m scripts.live_signal_bot --loop 900`.

## 2. Add a Volume (so signal history survives redeploys)
1. Service → **Settings → Volumes → New Volume**.
2. Mount path: **`/data`**.

## 3. Set environment variables
Service → **Variables**:

| Variable | Value | Notes |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | `123456:ABC...` | from @BotFather |
| `TELEGRAM_CHAT_ID` | `-100xxxxxxxxxx` | your channel/group id |
| `DB_PATH` | `/data/signals.db` | **must point at the volume** |
| `POLL_SECONDS` | `900` | poll interval (optional; start cmd already sets loop) |
| `PRICE_OFFSET` | `0` | `Exness_XAU − PAXG`; set after comparing your chart |
| `TIMEFRAMES` | `H1,M30` | optional |
| `RISK_PCT` | `1.0` | optional, shown in messages |

Do **not** commit the token — `configs/telegram.json` is gitignored; production
reads the env vars above.

## 4. Deploy & verify
- Deploy. Logs should show `polling every 900s ... checking...`.
- It only messages when a NEW H1/M30 bar prints a signal (can be quiet for hours
  in a range). To confirm wiring instantly, run locally once: `--test`.

## 5. Inspect the track record
The DB accrues a live record to compare against the backtest. Locally:
```
python -X utf8 -m scripts.live_signal_bot --status
```
On Railway, query `/data/signals.db` (table `signals`): each row has entry/SL/TP,
R:R, source price, and once resolved `status` (win/loss) + `result_r`.

## Notes
- **Binance PAXG is the DATA feed only** — you trade on your own broker (Exness).
  `PRICE_OFFSET` maps PAXG levels onto Exness XAU.
- Outcome resolution is conservative (if a 5m bar touches both SL and TP, it
  counts as SL) so the live win-rate is a *floor*, not inflated.
- Parity: a signal at bar i close is entered at the next bar's open.
