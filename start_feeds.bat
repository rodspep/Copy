@echo off
REM (Re)start the UG listener + copier reliably via process_control.ps1:
REM loop-kill any existing (taskkill /F /T until zero), delete locks, then launch
REM one of each and sanity-check the count. Run after update.bat.
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\mt5-bot\scripts\process_control.ps1" -Action start-feeds
