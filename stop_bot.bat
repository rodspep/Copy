@echo off
REM Stop ONLY the XAU signal bot (scripts.live_signal_bot). Does NOT touch the UG
REM listener/copier or the vol bot. Matches the bot's python by command line, kills
REM it, then verifies none remain. To start it again later: restart_bot.bat.
setlocal
set "BOTDIR=C:\mt5-bot"
cd /d "%BOTDIR%"

for /f "usebackq" %%p in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.SessionId -ne 0 -and $_.CommandLine -match '-m scripts.live_signal_bot' } | Select-Object -ExpandProperty ProcessId"`) do (
  echo [stop] stopping XAU bot PID %%p
  taskkill /F /PID %%p >nul 2>&1
)
ping -n 3 127.0.0.1 >nul 2>&1

powershell -NoProfile -ExecutionPolicy Bypass -Command "$n=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.SessionId -ne 0 -and $_.CommandLine -and $_.CommandLine -match '-m scripts.live_signal_bot' }).Count; if($n -eq 0){ 'XAU bot stopped (0 running).'; exit 0 } else { \"WARN: $n still running\"; exit 1 }"
endlocal
