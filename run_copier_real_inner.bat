@echo off
REM ===========================================================================
REM  REAL-MONEY copier. Only use this AFTER the MT5 terminal is logged into the
REM  REAL account. --allow-real permits trading a non-demo account; real_mode then
REM  trades ONLY the proven 50pip edge (100/150 auto-skipped). --tag real keeps a
REM  SEPARATE ledger (data\copier_trades_real.db) so real P/L never mixes with demo.
REM  Conservative: vol 0.01/leg (0.02/signal), max-open 6 (=3 concurrent signals),
REM  expiry 120min. Start SMALL; scale only after real trading proves out.
REM ===========================================================================
cd /d C:\mt5-bot
set PYTHONUNBUFFERED=yes
"C:\mt5-bot\.venv\Scripts\python.exe" -X utf8 -m scripts.ug_copier --live --allow-real --tag real ^
  --symbol XAUUSDm --poll 2 --max-open 6 --expiry-min 120 --volume 0.01
