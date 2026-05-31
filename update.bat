@echo off
REM ============================================================
REM update.bat -- Sync code from GitHub + restart XAU bot.
REM   Run on VPS (C:\mt5-bot) after `git push` from local.
REM   Idempotent: `git reset --hard` always converges to origin/main.
REM   Does NOT need admin. Bot is restarted via existing restart_bot.bat
REM   (which uses PsExec to enter MT5's interactive session).
REM   Uses absolute git path because SSH non-interactive PATH skips Program Files.
REM ============================================================
setlocal
set "GIT=C:\Program Files\Git\cmd\git.exe"
cd /d C:\mt5-bot || ( echo [update] ERROR: C:\mt5-bot missing & exit /b 1 )

echo [update] fetching origin/main...
"%GIT%" fetch origin main
if errorlevel 1 ( echo [update] ERROR: git fetch failed & exit /b 1 )

echo [update] incoming commits:
"%GIT%" log --oneline HEAD..origin/main
echo.

"%GIT%" reset --hard origin/main
if errorlevel 1 ( echo [update] ERROR: git reset failed & exit /b 1 )

echo [update] code synced to:
"%GIT%" log --oneline -1
echo.

echo [update] restarting XAU bot...
call C:\mt5-bot\restart_bot.bat
if errorlevel 1 ( echo [update] WARN: code synced but restart_bot.bat reported failure -- check logs\bot.log + MT5. & exit /b 1 )
echo [update] OK: code synced + bot restarted.
endlocal
