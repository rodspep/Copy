@echo off
REM Launch the UG Telegram listener detached (network-only, no MT5 needed → runs
REM in session 0). It parses live UG messages → data\ug\live_signals.jsonl, which
REM the copier consumes. Logs to logs\listener.log. Re-running starts a NEW one;
REM make sure only ONE listener runs (Telegram allows one session consumer).
setlocal
set "BOTDIR=C:\mt5-bot"
cd /d "%BOTDIR%"
echo [listener] launching UG listener (see logs\listener.log)...
REM Call an inner .bat (not python.exe inline) — inline python in the nested
REM PsExec cmd breaks the quoting (same fix as run_fetch.bat).
"%BOTDIR%\PsExec64.exe" -accepteula -nobanner -d cmd /c "cd /d ""%BOTDIR%"" && call ""%BOTDIR%\run_listener_inner.bat"" >> ""%BOTDIR%\logs\listener.log"" 2>&1"
echo [listener] launched (PsExec exit %errorlevel%).
endlocal
