@echo off
REM Restart ONLY the XAU bot in its MT5 session — no full VPS reboot, and WITHOUT
REM touching any other python process (e.g. the Binance/vol bot on the same VPS).
setlocal
set "BOTDIR=C:\mt5-bot"
set "MARK=-m scripts.live_signal_bot"

REM 1) Kill ONLY the XAU bot's python (match its command line), then wait for exit.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*-m scripts.live_signal_bot*' }; $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; $p | ForEach-Object { Wait-Process -Id $_.ProcessId -Timeout 10 -ErrorAction SilentlyContinue }"

REM 2) Refuse to launch a second instance if the old one is somehow still alive.
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*-m scripts.live_signal_bot*' }) { exit 1 } else { exit 0 }"
if errorlevel 1 ( echo [restart] ERROR: old XAU bot still running -- abort to avoid double-launch. & exit /b 1 )

REM 3) Detect the session where MT5 runs; abort if MT5 is not running (never guess).
set "SID="
for /f "usebackq delims=" %%a in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'terminal64.exe' } | Select-Object -First 1; if ($p) { $p.SessionId }"`) do set "SID=%%a"
if not defined SID ( echo [restart] ERROR: terminal64.exe ^(MT5^) not running -- refuse to launch into a guessed session. & exit /b 1 )
echo [restart] MT5 session = %SID% -- launching XAU bot there...

REM 4) Launch the bot into MT5's interactive session via PsExec (detached).
"%BOTDIR%\PsExec64.exe" -accepteula -nobanner -i %SID% -d cmd /c "cd /d %BOTDIR% && set PYTHONUNBUFFERED=yes && call %BOTDIR%\run_bot_vps.bat >> %BOTDIR%\logs\bot.log 2>&1"
if errorlevel 1 ( echo [restart] ERROR: PsExec launch failed. & exit /b 1 )
echo [restart] launched in session %SID%.
endlocal
