@echo off
REM ===========================================================================
REM  REAL-MONEY copier. Only use this AFTER the MT5 terminal is logged into the
REM  REAL account. --allow-real permits trading a non-demo account; real_mode then
REM  trades ALL UG methods with a PER-METHOD 2-leg exit (near TP: 150->100pip, 50/100->50pip;
REM  runner: 150->200pip, 50/100->150pip; SL->BE after the near leg). The
REM  bot AUTO-DETECTS demo vs real and uses a SEPARATE ledger per real account
REM  (data\copier_trades_real_<login>.db) so real P/L never mixes with demo stats.
REM  Conservative: vol 0.01/leg (0.02/signal), max-open 6 (=3 concurrent signals),
REM  expiry 240min (4h — backtest: limit-at-mid fills more good-priced pullbacks, +~20-27%
REM  $/signal vs 120min, WR unchanged ~80%). Start SMALL; scale only after real proves out.
REM ===========================================================================
cd /d C:\mt5-bot
set PYTHONUNBUFFERED=yes
REM  Daily-loss circuit-breaker DISABLED (no --max-daily-loss → default 0 = off): only a
REM  small amount is deposited (not full capital) and 0.02/signal risk is acceptable, so
REM  no daily-loss cap is needed for now. To re-enable, append e.g. --max-daily-loss 70
REM  (stops NEW entries once today's NET realized P/L <= -70 USD; ~4-5%% of equity).
REM  --max-signal-age-min 2: only place if the signal is < 2 min old (lag = UG-post →
REM  bot). A scalp entered late is "chắc chắn lỗ"; this also stops a soft-blocked signal
REM  from being re-placed stale on a restart. Tune tighter/looser to taste.
REM  US-session filter (--skip-hours 13,14,15,16) DISABLED per user 2026-06-04: a winning
REM  13-16 UTC signal got blocked; user opted to trade ALL hours. The backtest had shown
REM  skipping 13-16 = +35-50%% avg $/signal, but it's a statistical bet (blocks some winners
REM  too). To restore, append: --skip-hours 13,14,15,16
"C:\mt5-bot\.venv\Scripts\python.exe" -X utf8 -m scripts.ug_copier --live --allow-real ^
  --symbol XAUUSDm --poll 2 --max-open 6 --expiry-min 240 --volume 0.01 --max-signal-age-min 2
