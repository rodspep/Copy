# UG signal generation — intelligence from the source channel

Source: Telethon dump of "AI Trading Signal Underground" (chat -1002006543603),
394 text + 28 image messages, 2026-03-25 → 2026-06-03. Stored:
`data/ug/tcu_chat/messages.jsonl` + `data/ug/tcu_chat/media/`. The admin's own posts
(90 non-signal) + shared methodology images reveal HOW the signals are made — far more
than price-only reverse engineering could.

## The big reveal — it's an LLM ENSEMBLE, not a fixed algorithm (msg #481, 03-25)
> "indicator + Các điều kiện thoả mãn thì AI sẽ gửi về cho 4 AI model
> **Claude + GPT + Grok + Deepseek để voting** đưa ra entry + khuyến nghị ít rủi ro nhất"

So the pipeline is: **indicator/condition triggers → 4 LLMs vote → entry + SL + TP + risk
+ recommendation.** This is why a deterministic rule only ever hit ~91%: the final entry
/risk/method is decided by a stochastic LLM vote, not a formula. Implications:
- **Non-determinism is inherent** — same indicators can yield different entries run-to-run.
- Admin says **Claude > GPT > Grok/Deepseek** in quality (#583: "Deepseek với Grok ảo
  tưởng lắm, ko xịn bằng Claude").
- **LLM token budget directly degrades signals** (#554, #556): when rate-limited / buying
  fewer tokens, "TP yếu", signals "phập phù", "15' đớp 60% cạn phước". Quality varies with
  their API spend + 5-hourly rate-limit resets. → our observed day-to-day variance is partly
  THEIR infra, not the market.

## Multiple BOTS / methods run in parallel (≈5, regime-switched) (#591, #608, #771)
The channel multiplexes several independent strategy bots; the displayName families we see
in the TCU API map to these:
1. **PP2 / Ai Scalp** — MA34/MA89 fade+trend, TP template {50,100,150,200}. (our main edge)
2. **PRI GOLD** — wider TP {150,200,300,400}.
3. **SMC Bot** (Vàng/Bạc/Dầu) — pure SMC, RR **1:3**, higher TF. Method infographic shared in
   msg_655: BOS / CHOCH / IDM (internal liquidity) / FVG / Order Block; entry = sweep IDM →
   retest OB+FVG → price-action confirm → SL beyond OB → TP ≥ 3R. (#654 "setup con bot smc mới")
4. **VGT — "Thợ Săn entry Vùng Giá Trị"** (msg_502) — Fibonacci 61.8% retrace + Elliott Wave
   (wave-C target) + EMA89 wick-sweep + a **Confidence score /100**; emits an "AI Analysis"
   verdict (quality grade, risk, recommendation, "prioritize TP1 over far TP2").
5. A 5th experimental bot (rotating).
- **Method is selected by market regime; winrate is tracked per bot** and a new signal is
  issued by whichever bot's setup fires (#591: "hệ thống ghi nhận winrate và cho kèo mới chứ
  ko phải hedge"). Two simultaneous opposite-side signals are two different bots, not a hedge.
- On a trend change the AI "tự đổi phương pháp" — accepts a few SL during the switch (#608).

## Parameters they tuned over time (matches our data)
- **TP too far → reverts** (#493, 03-26): "bot chỉ ăn đc 10 giá lại quay đầu… chắc chỉnh cho
  TP 10-20-30." Direct confirmation of our finding that far native TPs rarely reach → our
  unified TP1@50 + runner@150 cap is the right call.
- **SL distance experimented** (#495): "Nên SL 6 giá thì dễ khớp" (they tried 6 vs the 10
  we decoded). SL is a tuned knob, not sacred.
- **Entry offset / "điều chỉnh"** (#502, #611): they shift the entry to dodge liquidity/noise
  ("điều chỉnh entry lên 4460"), and admit a bug where "entry đến channel là 60… R:R < 1:1".
- **Session/news filter** (#486, #537, #588): a rule to NOT signal within 30 min around
  session open / news ("biến động mạnh, dễ ăn SL 2 đầu"); US session = "ngáo", turn bot off.
- **Price source** (#505): Binance vs MT5 mismatch caused bad entries; they moved to MT5.

## Operating reality / caveats (from the admin)
- Signals are "tham khảo" (reference only); the bot is "vô cảm", misses news/regime turns.
- The same signal is re-posted; an MT5 block once made it re-post a STALE old signal (#546:
  "kèo mới đăng lỗi… nó lấy kèo cũ… đừng vào theo") — exactly the stale-repost hazard our
  copier now guards (2-min freshness + freshest-per-key).
- Heavy SL clusters in choppy/sideway + post-news; admin repeatedly says scalp = "hit and run".

## What this means for OUR copier (hypotheses to validate, not act on blindly)
1. We cannot "solve" the entry formula — it's an LLM vote. Best we can do is (a) filter by
   the disclosed features + method, (b) lean on the proven scalp exit (already done), (c)
   treat day-to-day quality variance as partly their token/infra noise.
2. **Method matters more than a universal rule.** The TCU API `displayName` is the cleanest
   method label; edge analysis should be PER method (PP2 scalp is the proven one).
3. A **regime/session filter** on our side (skip US-session-open / high-vol windows) may
   mirror their own rule and cut the SL-heavy chop trades.
4. A **confidence/risk gate**: VGT emits a confidence score and the PP2 posts carry "Rủi ro
   x/10" + CAUTION/FOLLOW — we could weight or skip low-confidence / high-risk signals.

NONE of the above is implemented — these are leads for the decode-v2 + edge work.

## CORRECTION (important) — DON'T pool history; UG's version changed mid-May
First pass pooled 6 weeks of the API set under our exit and concluded the 50pip scalp was
weak/overfit. That was WRONG — it mixed UG's OLD (buggy) and CURRENT (improved) versions.
A clean weekly replay shows a sharp quality breakpoint ~W21 (mid-May), consistent with the
channel's constant "đang sửa / bot mới":

  week  W17  W18  W19  W20 | W21  W22  W23      (WR of filled)
  ALL   67%  73%  78%  70% | 83%  90%  80%
  50pip 67%  69%  69%  69% | 82%  88%  79%

Split OLD (<W21) vs CURRENT (≥W21), net $/received signal under our exit:
  TP1=50 : OLD −$0.85 (WR 69%)  →  CURRENT +$1.42 (WR 83%)
  TP1=150: OLD +$1.12          →  CURRENT +$0.41
  TP1=200: OLD +$0.44          →  CURRENT +$1.85 (WR 92%, n=21)
  ALL    : OLD −$0.23          →  CURRENT +$1.34 (WR 84%)

Conclusions:
- The 50pip scalp was a LOSER in the old version but is one of the BEST in the current one.
  So the live copier (50/100/150 + unified exit) is trading the IMPROVED version — reasonable.
- The earlier "Telegram week" wasn't a cherry-picked fluke — it was the START of the improved
  regime. Templates/SL barely changed week-to-week; what changed is QUALITY (entry/timing) →
  a logic/code improvement, not a template change.
- CAVEAT: "better now" = code improvement OR favourable market regime — can't fully separate;
  UG keeps changing, so it can break again. The shadow-log (forward, receipt-time) is the
  safeguard that will show a quality drop immediately.

DECISION: analysis is now RECENCY-WEIGHTED by default (scripts/tcu_edge.py RECENT_DAYS=28 +
weekly trajectory; full-pooled shown only as context). Do NOT make filter decisions on pooled
history. TP1=200 (currently filtered out by the copier) is the strongest CURRENT bucket but
n=21 — confirm on the shadow-log before opening it.
