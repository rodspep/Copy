@echo off
REM Live XAU signal bot — one pass (called by Task Scheduler every 15 min).
cd /d C:\Users\Admin\Desktop\TT
set PYTHONIOENCODING=utf-8
"C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe" -X utf8 -m scripts.live_signal_bot >> logs\signal_bot.log 2>&1
