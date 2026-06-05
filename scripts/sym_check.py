"""Liet ke symbol XAU tren terminal hien tai + thu lay tick. Chay trong session 1 (PsExec).
Dung de tim dung ten symbol vang cua tai khoan (real dung XAUUSDm, demo co the khac)."""
from src.data import mt5_feed
mt5_feed.init()
import MetaTrader5 as m

acc = m.account_info()
print("ACCT", acc.login if acc else None, "|", acc.server if acc else None)
syms = m.symbols_get() or []
xau = [s.name for s in syms if "XAU" in s.name.upper()]
print("XAU symbols:", xau)
for nm in list(dict.fromkeys(xau + ["XAUUSDm", "XAUUSD"])):
    m.symbol_select(nm, True)
    t = m.symbol_info_tick(nm)
    print(f"  {nm} -> bid {t.bid if t else None}")
