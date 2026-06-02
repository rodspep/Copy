@echo off
REM Launch the UG copier (DRY-RUN — see run_copier_inner.bat) detached, in the MT5
REM session (Mt5/DryRun broker reads prices from the running terminal). Same proven
REM PsExec pattern as the bot/listener. Logs to logs\copier.log. Reads the listener
REM feed data\ug\live_signals.jsonl. To go LIVE, edit run_copier_inner.bat to add
REM --live (places REAL orders) — keep on a DEMO account first.
setlocal
set "BOTDIR=C:\mt5-bot"
cd /d "%BOTDIR%"

set "SID="
for /f "usebackq delims=" %%a in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'terminal64.exe' -and $_.SessionId -ne 0 } | Select-Object -ExpandProperty SessionId -Unique); if ($s.Count -eq 1) { $s[0] }"`) do set "SID=%%a"
if not defined SID ( echo [copier] ERROR: need exactly one interactive MT5 terminal -- abort. & exit /b 1 )
echo [copier] launching in session %SID% (see logs\copier.log)...

"%BOTDIR%\PsExec64.exe" -accepteula -nobanner -i %SID% -d cmd /c "cd /d ""%BOTDIR%"" && call ""%BOTDIR%\run_copier_inner.bat"" >> ""%BOTDIR%\logs\copier.log"" 2>&1"
echo [copier] launched (PsExec exit %errorlevel%).
endlocal
