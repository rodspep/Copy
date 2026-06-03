@echo off
REM LIVE copier on the DEMO account (--live). The copier ABORTS if the terminal
REM is NOT a demo account (needs --allow-real to ever trade real money). Reads the
REM listener feed, places/manages real orders on demo. vol 0.01, max-open 12, poll 2s,
REM expiry 120min. expiry 240->120: in-sample the 120-240min late fills were net-negative
REM (worse pull-back context), so 120 raised both WR (83->85.5%) and net (+$115->+$129).
REM Forward-testing on demo. max-open 12 so the cap doesn't bottleneck the test.
cd /d C:\mt5-bot
set PYTHONUNBUFFERED=yes
"C:\mt5-bot\.venv\Scripts\python.exe" -X utf8 -m scripts.ug_copier --live --symbol XAUUSDm --poll 2 --max-open 12 --expiry-min 120
