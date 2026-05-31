@echo off
REM Restart ONLY the XAU bot, with no reboot and without touching the vol bot.
REM Key idea: the XAU bot is the ONLY python in an INTERACTIVE session (it shares
REM MT5's session, != 0). The vol bot + any SSH python run in session 0, so they
REM are never touched. List PIDs via PowerShell (reliable), kill via taskkill
REM (reliable), then verify exactly one XAU bot ends up running.
setlocal
set "BOTDIR=C:\mt5-bot"
cd /d "%BOTDIR%"

REM 1) Stop every XAU bot (python.exe in a non-zero session). Converges even if
REM    duplicates exist. vol (session 0) is untouched.
for /f "usebackq" %%p in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.SessionId -ne 0 } | Select-Object -ExpandProperty ProcessId"`) do (
  echo [restart] stopping XAU bot PID %%p
  taskkill /F /PID %%p >nul 2>&1
)
ping -n 3 127.0.0.1 >nul 2>&1

REM 2) Find MT5's session -- require exactly one interactive terminal64.exe.
set "SID="
for /f "usebackq delims=" %%a in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'terminal64.exe' -and $_.SessionId -ne 0 } | Select-Object -ExpandProperty SessionId -Unique); if ($s.Count -eq 1) { $s[0] }"`) do set "SID=%%a"
if not defined SID ( echo [restart] ERROR: need exactly one interactive MT5 terminal -- abort. & exit /b 1 )
echo [restart] MT5 session = %SID% -- launching XAU bot...

REM 3) Launch the bot into MT5's session.
REM PsExec -d returns the spawned PID (>0) as exit code — NOT an error. Don't
REM check errorlevel here; the verify step below is the real success signal.
"%BOTDIR%\PsExec64.exe" -accepteula -nobanner -i %SID% -d cmd /c "cd /d ""%BOTDIR%"" && set PYTHONUNBUFFERED=yes && call ""%BOTDIR%\run_bot_vps.bat"" >> ""%BOTDIR%\logs\bot.log"" 2>&1" >nul 2>&1

REM 4) Verify the bot is up. A Python venv on Windows shows TWO python.exe per
REM    bot (a small launcher stub + the actual interpreter), so we count the
REM    REAL bot by command-line match.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=(Get-Date).AddSeconds(30); while((Get-Date) -lt $d){ $n=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.SessionId -ne 0 -and $_.CommandLine -and $_.CommandLine -match '-m scripts.live_signal_bot' }).Count; if($n -ge 1){ exit 0 }; Start-Sleep -Seconds 2 }; exit 1"
if errorlevel 1 ( echo [restart] WARN: bot not detected after 30s -- check log. & exit /b 1 )
echo [restart] OK: XAU bot running in session %SID%.
endlocal
