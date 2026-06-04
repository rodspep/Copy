@echo off
REM ===========================================================================
REM  SMC bot — DRY-RUN (default). Reads REAL prices, generates SMC setups, but
REM  PLACES NOTHING. Use this to paper-validate the live signal stream against the
REM  backtest before committing real money. Independent of the UG copier (own magic
REM  770820, own ledger data\smc_trades.db, own telegram configs\smc_telegram.json).
REM ===========================================================================
cd /d C:\mt5-bot
set PYTHONUNBUFFERED=yes
"C:\mt5-bot\.venv\Scripts\python.exe" -X utf8 -m scripts.smc_bot ^
  --symbol XAUUSDm --poll 20 --max-setups 4 --volume 0.01
