@echo off
REM External watchdog — Windows Task Scheduler runs this every ~3 min (session 0,
REM independent of the MT5 session) to alert if a copier heartbeat goes stale.
cd /d C:\mt5-bot
"C:\mt5-bot\.venv\Scripts\python.exe" -X utf8 -m scripts.watchdog >> "C:\mt5-bot\logs\watchdog.log" 2>&1
