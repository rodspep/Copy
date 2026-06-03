@echo off
REM Shadow-log poller — Windows Task Scheduler runs this every ~15 min (session 0,
REM independent of MT5/the copier) to append new TCU API signals (receipt-time) +
REM refresh outcomes. Pure HTTP read; cannot affect live trading.
cd /d C:\mt5-bot
"C:\mt5-bot\.venv\Scripts\python.exe" -X utf8 -m scripts.tcu_shadow >> "C:\mt5-bot\logs\tcu_shadow.log" 2>&1
