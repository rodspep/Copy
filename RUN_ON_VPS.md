# Run everything on the Windows VPS (real Exness prices, drop Railway)

Once you have a Windows VPS with the Exness MT5 terminal, run the **whole bot
there** with `DATA_SOURCE=mt5`. Bars come straight from your broker — no PAXG
proxy, no offset, no calibration sync. This is the clean setup and the base for
auto-execution later. Railway then becomes redundant.

## Setup
1. **VPS + MT5**: install the Exness MetaTrader 5 terminal, log in, add your gold
   symbol to Market Watch (often `XAUUSDm` — check the exact name).
2. **Python**: install 3.12+, then `pip install -r requirements.txt MetaTrader5`.
3. **Code**: clone the repo to the VPS.
4. **Secrets**: `copy configs\telegram.example.json configs\telegram.json` and put
   your `bot_token` + `chat_id` in it (gitignored — never committed).
5. **Symbol**: if your gold symbol isn't `XAUUSDm`, edit `run_bot_vps.bat`
   (`set MT5_SYMBOL=...`).

## Run
```
run_bot_vps.bat
```
This sets `DATA_SOURCE=mt5` and polls every 15 min. The terminal must stay logged
in. Verify the log shows `polling every 900s ... checking...` and the Telegram
messages say **"Exness MT5 (giá thật)"**.

### Keep it running across reboots
Task Scheduler → Create Task → Trigger: **At log on** → Action: start
`run_bot_vps.bat` → check "Run whether user is logged on or not". (Or wrap it as a
service with NSSM.) The bot self-loops, so it only needs starting once.

## Configuration (env vars, set by the .bat)
| Var | Value | Meaning |
|---|---|---|
| `DATA_SOURCE` | `mt5` | use the MT5 terminal feed (real Exness) |
| `MT5_SYMBOL` | `XAUUSDm` | your broker's gold symbol |
| `DB_PATH` | `data\signals.db` | local SQLite history (no volume needed) |
| `MT5_ACCOUNT`/`MT5_PASSWORD`/`MT5_SERVER` | *(optional)* | only if you want the bot to log in itself instead of attaching to the open terminal |

`PRICE_OFFSET` is **ignored** in MT5 mode (prices are already real).

## Turning Railway off
After the VPS bot is verified sending correct signals:
- Railway → service → **Settings → Remove / pause deployment**, or just delete the
  project. History there isn't needed (the VPS DB is now the source of truth).
- You no longer need `scripts/mt5_sync.py` (that was the PAXG→Exness calibrator
  for the Railway setup).

## Check the track record
```
python -X utf8 -m scripts.live_signal_bot --status
```
Every signal + its TP/SL outcome + realized R is in `data\signals.db` (table
`signals`) to compare against the backtest.

## Next: auto-execution
When you trust the tracked record, the same MT5 connection can place orders for
new signals (read open rows from the DB → `mt5.order_send`). Ask and we'll add
`--execute` with position sizing from your risk %.
