@echo off
cd /d C:\mt5-bot
set MT5_SYMBOL=XAUUSDm
set PYTHONUNBUFFERED=yes
"C:\mt5-bot\.venv\Scripts\python.exe" -X utf8 -m scripts.ug_copier_selftest --symbol XAUUSDm
