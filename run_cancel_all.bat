@echo off
REM Cancel ALL copier pending orders (magic 770150) in the MT5 session. Synchronous
REM (PsExec -i SID, no -d) so output lands in logs\cancel_all.log. Safety tool.
setlocal
set "BOTDIR=C:\mt5-bot"
cd /d "%BOTDIR%"
set "SID="
for /f "usebackq delims=" %%a in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'terminal64.exe' -and $_.SessionId -ne 0 } | Select-Object -ExpandProperty SessionId -Unique); if ($s.Count -eq 1) { $s[0] }"`) do set "SID=%%a"
if not defined SID ( echo [cancel] ERROR: need exactly one interactive MT5 terminal -- abort. & exit /b 1 )
echo [cancel] session %SID% -- cancelling magic pendings (see logs\cancel_all.log)...
"%BOTDIR%\PsExec64.exe" -accepteula -nobanner -i %SID% cmd /c "cd /d ""%BOTDIR%"" && call ""%BOTDIR%\run_cancel_all_inner.bat"" > ""%BOTDIR%\logs\cancel_all.log"" 2>&1"
echo [cancel] done (PsExec exit %errorlevel%).
endlocal
