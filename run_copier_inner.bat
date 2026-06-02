@echo off
REM LIVE copier on the DEMO account (--live). The copier ABORTS if the terminal
REM is NOT a demo account (needs --allow-real to ever trade real money). Reads the
REM listener feed, places/manages real orders on demo. vol 0.01, max-open 5, poll 5s.
cd /d C:\mt5-bot
set PYTHONUNBUFFERED=yes
"C:\mt5-bot\.venv\Scripts\python.exe" -X utf8 -m scripts.ug_copier --live --symbol XAUUSDm --poll 5
