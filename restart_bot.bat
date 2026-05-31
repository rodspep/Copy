@echo off
REM Restart ONLY the XAU bot inside its MT5 session — no full VPS reboot.
REM Triggered over SSH (session 0); PsExec launches the bot back into the
REM interactive session where MetaTrader 5 runs.
echo [restart] killing old bot process...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

REM Find the session where terminal64.exe (MT5) runs — the bot must share it.
set SID=1
for /f "tokens=4" %%a in ('tasklist /FI "IMAGENAME eq terminal64.exe" /FO TABLE /NH 2^>nul') do set SID=%%a
echo [restart] MT5 session = %SID%, launching bot there...

"%~dp0PsExec64.exe" -accepteula -nobanner -i %SID% -d cmd /c "cd /d %~dp0 && set PYTHONUNBUFFERED=yes && call run_bot_vps.bat >> logs\bot.log 2>&1"
echo [restart] launched.
