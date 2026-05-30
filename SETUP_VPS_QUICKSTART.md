# VPS quick start — get the bot live in ~5 minutes

For a fresh Windows VPS (AWS Lightsail / Vultr / etc.). Run these once the VPS is
ready and you're connected via Remote Desktop.

## Step 0 — install + log in to Exness MT5 (manual, can't be scripted)
1. Download the **Exness MetaTrader 5** terminal and install it.
2. Log in with your Exness MT5 credentials (Personal Area → trading account → MT5).
3. Add your gold symbol to **Market Watch** (right-click → Show All if hidden).
   Note the exact name — often `XAUUSDm`.

## Step 1 — one-command setup (Python + repo + dependencies)
Open **PowerShell as Administrator** and run:
```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
iwr -useb https://raw.githubusercontent.com/rodspep/Copy/main/setup_vps.ps1 | iex
```
This installs Python, downloads the repo to `C:\mt5-bot`, installs all packages
(including `MetaTrader5`), and asks you to paste your **Telegram bot token** and
**chat id** (kept local, never committed).

## Step 2 — verify the MT5 data feed
```
cd C:\mt5-bot
python -X utf8 scripts\check_mt5.py
```
You should see your account, the live gold price, and "ALL GOOD". If it reports a
different symbol than `XAUUSDm`, note it for the next step.

## Step 3 — set your symbol (only if not XAUUSDm)
Edit `C:\mt5-bot\run_bot_vps.bat`, change:
```
set MT5_SYMBOL=XAUUSDm     ->   set MT5_SYMBOL=YOUR_SYMBOL
```

## Step 4 — start the bot
```
run_bot_vps.bat
```
Logs should show `polling every 900s ... checking...`. A Telegram test/signal will
read **"Exness MT5 (giá thật)"**. It only messages when a new H1/M30 signal prints.

## Step 5 — keep it running across reboots
Task Scheduler → Create Task →
- **General:** "Run whether user is logged on or not"
- **Triggers:** New → "At log on"
- **Actions:** Start a program → `C:\mt5-bot\run_bot_vps.bat`

## Check the live track record anytime
```
cd C:\mt5-bot
python -X utf8 -m scripts.live_signal_bot --status
```

## Once verified → retire Railway
When the VPS bot sends correct signals, pause/delete the Railway project (the VPS
DB is now the source of truth). See RUN_ON_VPS.md.
