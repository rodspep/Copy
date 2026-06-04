@echo off
REM Launch the SMC bot DRY-RUN detached, in the MT5 session (reads prices from the
REM running terminal; PLACES NOTHING). Same proven PsExec pattern as the copier. Logs to
REM logs\smc_bot.log. Independent of the copier (own magic/ledger/lock) — a read-only 2nd
REM client; it will not place orders and cannot touch the copier's orders.
REM   For REAL trading use run_smc_bot_real_inner.bat instead (separate HEDGING account).
setlocal
set "BOTDIR=C:\mt5-bot"
cd /d "%BOTDIR%"

set "SID="
REM Require EXACTLY ONE interactive terminal64.exe (same guard the copier uses).
for /f "usebackq delims=" %%a in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'terminal64.exe' -and $_.SessionId -ne 0 }); if ($p.Count -eq 1) { $p[0].SessionId }"`) do set "SID=%%a"
if not defined SID ( echo [smc-DRY] ERROR: need exactly one interactive MT5 terminal -- abort. & exit /b 1 )
echo [smc-DRY] launching DRY-RUN in session %SID% (see logs\smc_bot.log)...

"%BOTDIR%\PsExec64.exe" -accepteula -nobanner -i %SID% -d cmd /c "cd /d ""%BOTDIR%"" && call ""%BOTDIR%\run_smc_bot.bat"" >> ""%BOTDIR%\logs\smc_bot.log"" 2>&1"
echo [smc-DRY] launched (PsExec returned %errorlevel% -- with -d this is the spawned PID).
endlocal
