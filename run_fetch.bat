@echo off
REM Run the MT5 history export in the bot's isolated venv. Invoked by fetch_mt5.bat
REM via PsExec (must run in the MT5 interactive session). Kept as a separate .bat
REM — like run_bot_vps.bat — so PsExec only has to `call` a path (no python.exe in
REM the nested cmd quoting, which is what broke the inline version).
cd /d C:\mt5-bot
set MT5_SYMBOL=XAUUSDm
set PYTHONUNBUFFERED=yes
"C:\mt5-bot\.venv\Scripts\python.exe" -X utf8 -m scripts.fetch_mt5_history --tfs M1 M5 M30 H1
