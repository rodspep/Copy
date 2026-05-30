# MT5 price sync on a Windows VPS (track real Exness price)

The cloud bot (Railway) generates signals from PAXG. This helper runs on a
**Windows VPS** where the Exness **MetaTrader 5** terminal is logged in, reads
the real Exness XAU price, and keeps Railway's `PRICE_OFFSET` calibrated so every
signal's entry/SL/TP matches your Exness chart.

```
Railway (24/7)              Windows VPS (24/7)
 signal bot (PAXG)   <----  mt5_sync.py  --loop 300
 uses PRICE_OFFSET          reads Exness XAU via MT5 → offset = Exness − PAXG
                            pushes offset to Railway when it drifts > threshold
```

## 1. Get your Exness MT5 login (you only use the app today)
Exness gives every account MT5 credentials even if you trade on mobile:
1. Exness **Personal Area** → your trading account → **Settings / "Trading
   Terminals"**.
2. Note: **account number**, **server** (e.g. `Exness-MT5Real8`), and set/!reset
   the **MT5 password** if needed.

## 2. Get a Windows VPS
Any Windows VPS (Contabo / Vultr / Kamatera, ~$5–10/mo). Remote-desktop in.

## 3. Install on the VPS
1. Download + install the **Exness MetaTrader 5** terminal; log in with the
   credentials from step 1. Add **XAUUSD** (or `XAUUSDm`) to Market Watch.
2. Install Python 3.12+, then:
   ```
   pip install MetaTrader5 requests pandas
   ```
3. Copy this repo (or just `scripts/mt5_sync.py` + `configs/`) to the VPS.
4. `copy configs\mt5.example.json configs\mt5.json` and fill it:
   - `mt5.account / password / server` from step 1
   - `mt5.symbol` = your broker's gold symbol (check Market Watch; often `XAUUSDm`)
   - `railway.token` = the Railway **project token** (already wired to project ids)

## 4. Run
```
python -X utf8 scripts\mt5_sync.py --selftest     # no MT5 — checks PAXG + Railway
python -X utf8 scripts\mt5_sync.py --price        # live Exness mid price
python -X utf8 scripts\mt5_sync.py --calibrate    # show offset, no push
python -X utf8 scripts\mt5_sync.py --loop 300     # sync to Railway every 5 min
```
Keep `--loop 300` running (Task Scheduler, or a terminal that stays open).

## How calibration works
- `offset = Exness_mid − PAXG_mid`, pushed to Railway `PRICE_OFFSET` only when it
  moves past `push_threshold` (default $0.5) — so the cloud bot isn't redeployed
  on every tick. Raise the threshold if you see frequent redeploys.
- After the first push, your Telegram signals show Exness-aligned prices.

## Next phase (execution)
This helper is the foundation for auto-execution: once you're confident in the
tracked record, the same MT5 connection can read open signals from the bot's DB
and place Exness orders. When you move to execution, the cleaner setup is to run
the **whole bot on the VPS** (direct MT5 prices, no PAXG/offset). Ask and we'll
wire `--execute`.
