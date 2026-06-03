@echo off
REM LIVE copier on the DEMO account (--live). The copier ABORTS if the terminal
REM is NOT a demo account (needs --allow-real to ever trade real money). Reads the
REM listener feed, places/manages real orders on demo. vol 0.01, max-open 12, poll 2s.
REM max-open raised to 12 for the deep-limit forward-test: limits rest up to expiry
REM (240min) waiting for the pull-back, so concurrency is higher; 12 keeps the cap from
REM bottlenecking the test (backtest had no cap). Demo only.
cd /d C:\mt5-bot
set PYTHONUNBUFFERED=yes
"C:\mt5-bot\.venv\Scripts\python.exe" -X utf8 -m scripts.ug_copier --live --symbol XAUUSDm --poll 2 --max-open 12
