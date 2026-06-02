@echo off
REM Run the copier broker self-test INSIDE the MT5 interactive session (so the
REM MetaTrader5 lib attaches to the logged-in terminal). Places + cancels ONE
REM real pending order on the CURRENT account — make sure MT5 is on a DEMO account.
REM Logs to logs\copier_selftest.log. Same PsExec pattern as restart_bot/run_fetch.
setlocal
set "BOTDIR=C:\mt5-bot"
cd /d "%BOTDIR%"
set "SID="
for /f "usebackq delims=" %%a in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'terminal64.exe' -and $_.SessionId -ne 0 } | Select-Object -ExpandProperty SessionId -Unique); if ($s.Count -eq 1) { $s[0] }"`) do set "SID=%%a"
if not defined SID ( echo [selftest] ERROR: need exactly one interactive MT5 terminal -- abort. & exit /b 1 )
echo [selftest] MT5 session = %SID% -- running place/cancel test (see logs\copier_selftest.log)...
"%BOTDIR%\PsExec64.exe" -accepteula -nobanner -i %SID% cmd /c "cd /d ""%BOTDIR%"" && call ""%BOTDIR%\run_copier_selftest_inner.bat"" > ""%BOTDIR%\logs\copier_selftest.log"" 2>&1"
echo [selftest] finished (PsExec exit %errorlevel%).
endlocal
