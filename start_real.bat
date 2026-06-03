@echo off
REM ===========================================================================
REM  Switch the copier to REAL money. PRECONDITION: the MT5 terminal must ALREADY
REM  be logged into the REAL account with AutoTrading ON. This stops any running
REM  feeds (demo copier + listener), then relaunches the listener + the REAL copier
REM  (run_copier_real.bat). Verify logs\copier.log shows "REAL · !! LIVE ...".
REM  To go back to demo: stop_feeds.bat, log the terminal back into demo, start_feeds.bat
REM ===========================================================================
setlocal
set "BOTDIR=C:\mt5-bot"
echo [real] stopping any running feeds...
powershell -NoProfile -ExecutionPolicy Bypass -File "%BOTDIR%\scripts\process_control.ps1" -Action stop-feeds
echo [real] launching listener...
call "%BOTDIR%\run_listener.bat"
echo [real] launching REAL copier...
call "%BOTDIR%\run_copier_real.bat"
echo [real] done. Check logs\copier.log + the new group for the REAL startup line.
endlocal
