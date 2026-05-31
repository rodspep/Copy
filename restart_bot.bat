@echo off
REM Restart ONLY the XAU bot, identified by its recorded PID (deterministic), in
REM its MT5 session, WITHOUT a reboot and WITHOUT touching any other bot.
setlocal
set "BOTDIR=C:\mt5-bot"
cd /d %BOTDIR%

REM 1) Stop the previous XAU bot by the PID it recorded -- ONLY if that PID is
REM    still a python.exe (so a reused PID can never hit another program/bot).
set "OLDPID="
if exist "%BOTDIR%\logs\xau_bot.pid" set /p OLDPID=<"%BOTDIR%\logs\xau_bot.pid"
if defined OLDPID (
  taskkill /F /FI "PID eq %OLDPID%" /FI "IMAGENAME eq python.exe" >nul 2>&1
  echo [restart] requested stop of old XAU bot PID %OLDPID%.
)
ping -n 3 127.0.0.1 >nul 2>&1

REM 2) Detect the session where MT5 (terminal64.exe) runs; abort if not running.
set "SID="
for /f "usebackq delims=" %%a in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'terminal64.exe' } | Select-Object -First 1; if ($p) { $p.SessionId }"`) do set "SID=%%a"
if not defined SID ( echo [restart] ERROR: terminal64.exe ^(MT5^) not running -- abort. & exit /b 1 )
echo [restart] MT5 session = %SID% -- launching XAU bot...

REM 3) Launch the bot into MT5's session; it rewrites logs\xau_bot.pid on startup.
"%BOTDIR%\PsExec64.exe" -accepteula -nobanner -i %SID% -d cmd /c "cd /d %BOTDIR% && set PYTHONUNBUFFERED=yes && call %BOTDIR%\run_bot_vps.bat >> logs\bot.log 2>&1"
if errorlevel 1 ( echo [restart] ERROR: PsExec launch failed. & exit /b 1 )
echo [restart] launched.
endlocal
