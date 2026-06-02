@echo off
REM Launch the UG Telegram listener detached, using the SAME proven pattern as
REM restart_bot.bat (PsExec -i <MT5 session> -d + call inner bat + outer redirect).
REM The listener is network-only (no MT5), but reusing the working launch avoids
REM the session-0 detach issues. Parses live UG msgs → data\ug\live_signals.jsonl.
REM Only ONE listener should run (Telegram allows one session consumer).
setlocal
set "BOTDIR=C:\mt5-bot"
cd /d "%BOTDIR%"

set "SID="
for /f "usebackq delims=" %%a in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'terminal64.exe' -and $_.SessionId -ne 0 } | Select-Object -ExpandProperty SessionId -Unique); if ($s.Count -eq 1) { $s[0] }"`) do set "SID=%%a"
if not defined SID ( echo [listener] ERROR: need exactly one interactive MT5 terminal -- abort. & exit /b 1 )
echo [listener] launching in session %SID% (see logs\listener.log)...

"%BOTDIR%\PsExec64.exe" -accepteula -nobanner -i %SID% -d cmd /c "cd /d ""%BOTDIR%"" && call ""%BOTDIR%\run_listener_inner.bat"" >> ""%BOTDIR%\logs\listener.log"" 2>&1"
echo [listener] launched (PsExec exit %errorlevel%).
endlocal
