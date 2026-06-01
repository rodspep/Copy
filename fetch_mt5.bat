@echo off
REM Export MT5 history to backtest parquet, IN the MT5 interactive session.
REM Like restart_bot.bat, the fetch must run where the logged-in terminal lives
REM (session != 0), so mt5.initialize() attaches to it instead of spawning a
REM fresh, not-logged-in terminal. Runs synchronously (no -d) and logs to
REM logs\fetch.log; read that over SSH afterwards. Read-only on MT5 — safe to run
REM while the bot is live.
setlocal
set "BOTDIR=C:\mt5-bot"
cd /d "%BOTDIR%"

set "SID="
for /f "usebackq delims=" %%a in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'terminal64.exe' -and $_.SessionId -ne 0 } | Select-Object -ExpandProperty SessionId -Unique); if ($s.Count -eq 1) { $s[0] }"`) do set "SID=%%a"
if not defined SID ( echo [fetch] ERROR: need exactly one interactive MT5 terminal -- abort. & exit /b 1 )
echo [fetch] MT5 session = %SID% -- exporting history (see logs\fetch.log)...

REM Call run_fetch.bat (not python.exe inline) — same proven pattern as
REM restart_bot.bat. Inlining python.exe inside the nested cmd /c broke the
REM quoting and python tried to parse python.exe as a script.
"%BOTDIR%\PsExec64.exe" -accepteula -nobanner -i %SID% cmd /c "cd /d ""%BOTDIR%"" && call ""%BOTDIR%\run_fetch.bat"" > ""%BOTDIR%\logs\fetch.log"" 2>&1"
echo [fetch] finished (PsExec exit %errorlevel%).
endlocal
