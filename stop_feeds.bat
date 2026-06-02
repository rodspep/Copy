@echo off
REM Reliably STOP all listener + copier python (taskkill /F /T loop until zero,
REM scoped to MT5-session python running -m scripts.ug_reader|ug_copier under
REM C:\mt5-bot). Never touches the bot or volscan. Does NOT relaunch.
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\mt5-bot\scripts\process_control.ps1" -Action stop-feeds
