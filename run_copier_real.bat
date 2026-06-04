@echo off
REM Launch the REAL-MONEY UG copier detached, in the MT5 session (Mt5 broker reads
REM prices + account from the running terminal). Same proven PsExec pattern as the
REM bot/listener. Logs to logs\copier_real.log. Reads the listener feed
REM data\ug\live_signals.jsonl.
REM  ! run_copier_real_inner.bat ALREADY passes --live --allow-real -- this is NOT a dry
REM  run. It places REAL orders as soon as the MT5 terminal is logged into a REAL
REM  account. The copier auto-detects demo vs real: on a DEMO login it trades demo, on a
REM  REAL login it trades real money (per-method 2-leg exit). Confirm MT5 is logged into
REM  the INTENDED account before launching.
setlocal
set "BOTDIR=C:\mt5-bot"
cd /d "%BOTDIR%"

set "SID="
REM Require EXACTLY ONE interactive terminal64.exe process (not just one session): with
REM two terminals open (e.g. demo + real) the copier could bind to the wrong account.
for /f "usebackq delims=" %%a in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'terminal64.exe' -and $_.SessionId -ne 0 }); if ($p.Count -eq 1) { $p[0].SessionId }"`) do set "SID=%%a"
if not defined SID ( echo [copier-REAL] ERROR: need exactly one interactive MT5 terminal -- abort. & exit /b 1 )
echo [copier-REAL] launching in session %SID% (see logs\copier_real.log)...

"%BOTDIR%\PsExec64.exe" -accepteula -nobanner -i %SID% -d cmd /c "cd /d ""%BOTDIR%"" && call ""%BOTDIR%\run_copier_real_inner.bat"" >> ""%BOTDIR%\logs\copier_real.log"" 2>&1"
REM With PsExec -d (detached), %errorlevel% is the spawned PID, NOT a success code.
echo [copier-REAL] launched (PsExec returned %errorlevel% -- with -d this is the spawned PID).
endlocal
