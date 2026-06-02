@echo off
REM Launch the UG Telegram listener detached (network-only, no MT5 needed → runs
REM in session 0). It parses live UG messages → data\ug\live_signals.jsonl, which
REM the copier consumes. Logs to logs\listener.log. Re-running starts a NEW one;
REM make sure only ONE listener runs (Telegram allows one session consumer).
setlocal
set "BOTDIR=C:\mt5-bot"
cd /d "%BOTDIR%"
echo [listener] launching UG listener (see logs\listener.log)...
"%BOTDIR%\PsExec64.exe" -accepteula -nobanner -d cmd /c "cd /d ""%BOTDIR%"" && set PYTHONUNBUFFERED=yes && ""%BOTDIR%\.venv\Scripts\python.exe"" -X utf8 -m scripts.ug_reader --listen >> ""%BOTDIR%\logs\listener.log"" 2>&1"
echo [listener] launched (PsExec exit %errorlevel%).
endlocal
