@echo off
REM Redirect INSIDE the inner bat so output/errors are captured even when launched
REM detached via PsExec -d (the outer redirect doesn't survive -d).
cd /d C:\mt5-bot
set PYTHONUNBUFFERED=yes
"C:\mt5-bot\.venv\Scripts\python.exe" -X utf8 -m scripts.ug_reader --listen >> "C:\mt5-bot\logs\listener.log" 2>&1
