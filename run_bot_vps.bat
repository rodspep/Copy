@echo off
REM Run the XAU signal bot on the Windows VPS using real Exness MT5 prices.
REM Uses an ISOLATED venv (.venv) so its deps never clash with other bots on the
REM same VPS. Secrets come from configs\telegram.json (gitignored).
cd /d C:\mt5-bot
set DATA_SOURCE=mt5
set MT5_SYMBOL=XAUUSDm
set DB_PATH=data\signals.db
set PYTHONIOENCODING=utf-8
"C:\mt5-bot\.venv\Scripts\python.exe" -X utf8 -m scripts.live_signal_bot --loop 60
