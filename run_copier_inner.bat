@echo off
REM DRY-RUN copier (no --live → never places real orders; reads real prices +
REM the listener's feed, logs intended actions). Add --live ONLY when ready.
cd /d C:\mt5-bot
set PYTHONUNBUFFERED=yes
"C:\mt5-bot\.venv\Scripts\python.exe" -X utf8 -m scripts.ug_copier --symbol XAUUSDm --poll 30
