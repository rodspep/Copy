@echo off
REM ===========================================================================
REM  REAL-MONEY copier. Only use this AFTER the MT5 terminal is logged into the
REM  REAL account. --allow-real permits trading a non-demo account; real_mode then
REM  trades ALL UG methods with the UNIFIED exit (TP1@50pip + runner@150pip + SL->BE). The
REM  bot AUTO-DETECTS demo vs real and uses a SEPARATE ledger per real account
REM  (data\copier_trades_real_<login>.db) so real P/L never mixes with demo stats.
REM  Conservative: vol 0.01/leg (0.02/signal), max-open 6 (=3 concurrent signals),
REM  expiry 120min. Start SMALL; scale only after real trading proves out.
REM ===========================================================================
cd /d C:\mt5-bot
set PYTHONUNBUFFERED=yes
REM  --max-daily-loss 70: stop NEW entries after ~-$70 net realized in a day (~3 bad
REM  signals, ~4.6%% of a $1500 account). TUNE to your real balance (≈4-5%% of equity).
"C:\mt5-bot\.venv\Scripts\python.exe" -X utf8 -m scripts.ug_copier --live --allow-real ^
  --symbol XAUUSDm --poll 2 --max-open 6 --expiry-min 120 --volume 0.01 --max-daily-loss 70
