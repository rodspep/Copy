@echo off
REM ===========================================================================
REM  SMC bot — REAL MONEY. Only run AFTER an MT5 terminal is logged into the
REM  SEPARATE, FUNDED SMC account (NOT the copier's account — SMC must be isolated).
REM  Required capital (REALIZABLE maxDD -$915 @ 0.02 lot, no-gate cap 4 — see
REM  scripts\smc_live_sim.py): $4500-5000 (maxDD ~18-20%); minimum $3700 (~25%). Below
REM  that, do NOT run SMC — its deep ~100+ day drawdown will blow up a small account.
REM
REM  --live --allow-real : place real orders on a non-demo account.
REM  --volume 0.01       : lot PER LEG (2 legs/signal => 0.02/signal, as backtested).
REM  --max-setups 4      : at most 4 concurrent live setups.
REM  Exit is fixed by smc_logic: +4R books -> runner SL->BE, +10R runner, retest
REM  expiry 360min, 24h horizon time-stop. Magic 770820, ledger smc_trades_real_<login>.db.
REM
REM  IMPORTANT: the SMC account needs its OWN MT5 terminal instance (a 2nd terminal
REM  logged into the SMC account). mt5_feed must attach to THAT terminal. Confirm the
REM  startup line prints the SMC account login before letting it trade.
REM ===========================================================================
cd /d C:\mt5-bot
set PYTHONUNBUFFERED=yes
"C:\mt5-bot\.venv\Scripts\python.exe" -X utf8 -m scripts.smc_bot --live --allow-real ^
  --symbol XAUUSDm --poll 20 --max-setups 4 --volume 0.01
