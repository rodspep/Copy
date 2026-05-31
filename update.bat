@echo off
REM ============================================================
REM update.bat -- Sync code from GitHub + restart XAU bot.
REM   Run on VPS (C:\mt5-bot) after `git push` from local.
REM   Idempotent: `git reset --hard` always converges to origin/main.
REM   Does NOT need admin. Bot is restarted via existing restart_bot.bat
REM   (which uses PsExec to enter MT5's interactive session).
REM ============================================================
setlocal
cd /d C:\mt5-bot || ( echo [update] ERROR: C:\mt5-bot missing & exit /b 1 )

echo [update] fetching origin/main...
git fetch origin main
if errorlevel 1 ( echo [update] ERROR: git fetch failed & exit /b 1 )

echo [update] incoming commits:
git log --oneline HEAD..origin/main
echo.

git reset --hard origin/main
if errorlevel 1 ( echo [update] ERROR: git reset failed & exit /b 1 )

echo [update] code synced to:
git log --oneline -1
echo.

echo [update] restarting XAU bot...
call C:\mt5-bot\restart_bot.bat
endlocal
