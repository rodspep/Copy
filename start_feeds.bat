@echo off
REM (Re)start the UG listener + copier cleanly: kill any existing ones (matched by
REM module on the command line — never touches the bot or volscan), then relaunch
REM both. Run after update.bat (which now only restarts the bot). One command to
REM bring the feeds up with the latest code.
setlocal
set "BOTDIR=C:\mt5-bot"
cd /d "%BOTDIR%"
echo [feeds] stopping any existing listener/copier...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -match 'scripts.ug_reader' -or $_.CommandLine -match 'scripts.ug_copier') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
ping -n 6 127.0.0.1 >nul 2>&1
del /q "%BOTDIR%\data\ug\copier_live.lock" "%BOTDIR%\data\ug\copier_dry.lock" 2>nul
echo [feeds] starting listener + copier...
call "%BOTDIR%\run_listener.bat"
call "%BOTDIR%\run_copier.bat"
echo [feeds] done.
endlocal
