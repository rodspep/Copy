@echo off
REM Run the FULL signal bot on the Windows VPS using real Exness MT5 prices.
REM Secrets (TELEGRAM_BOT_TOKEN / CHAT_ID) come from configs\telegram.json
REM (gitignored) — create it from configs\telegram.example.json on the VPS.
cd /d %~dp0
set DATA_SOURCE=mt5
set MT5_SYMBOL=XAUUSDm
set DB_PATH=data\signals.db
set PYTHONIOENCODING=utf-8
python -X utf8 -m scripts.live_signal_bot --loop 900
