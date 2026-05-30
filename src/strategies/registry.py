"""Strategy registry: per-symbol catalog of strategy candidates + their parameter
spaces for the optimizer.

Each entry is keyed by `(symbol, strategy_name)` and maps to:
  - 'strategy_cls'   : the Strategy subclass to instantiate
  - 'param_space'    : Optuna ParamSpace dict

Per `[[feedback-per-symbol-strategies]]`: the same strategy class can appear
under both symbols with DIFFERENT param spaces tuned to the asset's microstructure.
"""
from __future__ import annotations

from src.strategies.xau.ema_pullback import XauEmaPullback
from src.strategies.xau.session_breakout import XauSessionBreakout
from src.strategies.xau.smc_orderblock import XauSmcOrderBlock
from src.strategies.xau.liquidity_sweep_reversal import XauLiquiditySweepReversal
from src.strategies.xau.smc_confluence import XauSmcConfluence
from src.strategies.xau.zone_anticipator import XauZoneAnticipator
from src.strategies.xau.mtf_smc_entry import XauMtfSmcEntry
from src.strategies.xau.ma34_cascade import XauMa34Cascade
from src.strategies.xau.htf_trend_reversal import XauHtfTrendReversal
from src.strategies.xau.ltf_ob_entry import XauLtfObEntry, XauM15ObEntry
from src.strategies.xau.reaction_level import XauReactionLevel
from src.strategies.xau.ug_methods import XauScalpFade, XauDeepPullback
from src.strategies.xau.ob_fvg_trend import XauObFvgTrend
from src.strategies.xau.trend_follow import XauTrendFollow
from src.strategies.btc.trend_following import BtcTrendFollowing
from src.strategies.btc.bollinger_squeeze import BtcBollingerSqueeze
from src.strategies.btc.rsi_mean_revert import BtcRsiMeanRevert
from src.strategies.btc.smc_orderblock import BtcSmcOrderBlock


REGISTRY: dict[tuple[str, str], dict] = {

    # ---------------- XAU ----------------

    # NOTE: tp_atr_mult lower bounds widened to 0.2-0.3 across all strategies
    # to allow the optimizer to discover "tight TP / high WR" scalp setups
    # (matching the UG Trading bot reference: TP1 ≈ 0.5×SL with ~95% hit rate
    # gives expectancy ≈ +0.45R). Without this the optimizer can never explore
    # WR-dominated regimes; it could only chase larger R:R which fail OOS.

    ("XAUUSD", "ema_pullback"): {
        "strategy_cls": XauEmaPullback,
        "param_space": {
            "htf_ema_fast":           {"type": "int",   "low": 20,  "high": 100, "step": 10},
            "htf_ema_slow":           {"type": "int",   "low": 100, "high": 300, "step": 20},
            "htf_adx_min":            {"type": "float", "low": 12.0, "high": 35.0, "step": 1.0},
            "ltf_ema_pullback":       {"type": "int",   "low": 10,  "high": 50,  "step": 5},
            "ltf_pullback_atr_mult":  {"type": "float", "low": 0.2, "high": 1.5},
            "sl_atr_mult":            {"type": "float", "low": 0.5, "high": 3.0},
            "tp_atr_mult":            {"type": "float", "low": 0.2, "high": 5.0},
        },
    },

    ("XAUUSD", "session_breakout"): {
        "strategy_cls": XauSessionBreakout,
        "param_space": {
            "trade_start_hour":       {"type": "int",   "low": 6,   "high": 9},
            "trade_end_hour":         {"type": "int",   "low": 13,  "high": 20},
            "sl_atr_mult":            {"type": "float", "low": 0.5, "high": 2.0},
            "tp_atr_mult":            {"type": "float", "low": 0.3, "high": 4.0},
            "breakout_buffer_atr":    {"type": "float", "low": 0.0, "high": 0.5},
        },
    },

    ("XAUUSD", "smc_orderblock"): {
        "strategy_cls": XauSmcOrderBlock,
        "param_space": {
            "htf_swing_left":         {"type": "int",   "low": 2,   "high": 6},
            "htf_swing_right":        {"type": "int",   "low": 2,   "high": 6},
            "sl_atr_mult":            {"type": "float", "low": 0.3, "high": 1.5},
            "tp_atr_mult":            {"type": "float", "low": 0.3, "high": 5.0},
            "ob_proximity_pct":       {"type": "float", "low": 0.0005, "high": 0.005, "log": True},
        },
    },

    ("XAUUSD", "liquidity_sweep_reversal"): {
        "strategy_cls": XauLiquiditySweepReversal,
        "param_space": {
            "swing_left":             {"type": "int",   "low": 3,   "high": 10},
            "swing_right":            {"type": "int",   "low": 3,   "high": 10},
            "sl_atr_mult":            {"type": "float", "low": 0.3, "high": 1.2},
            "tp_atr_mult":            {"type": "float", "low": 0.2, "high": 2.5},
            "vwap_filter":            {"type": "categorical", "choices": [True, False]},
        },
    },

    ("XAUUSD", "zone_anticipator"): {
        "strategy_cls": XauZoneAnticipator,
        "param_space": {
            # Entry mode: optimizer picks limit (pending) vs reaction (market+confirm)
            "mode":                   {"type": "categorical", "choices": ["limit", "reaction"]},

            # Which zones to use (combinations matter)
            "use_ob_zone":            {"type": "categorical", "choices": [True, False]},
            "use_fvg_zone":           {"type": "categorical", "choices": [True, False]},
            "use_swing_zone":         {"type": "categorical", "choices": [True, False]},
            "use_asian_zone":         {"type": "categorical", "choices": [True, False]},

            "swing_left":             {"type": "int",   "low": 3,   "high": 8},
            "swing_right":            {"type": "int",   "low": 3,   "high": 8},
            "zone_touch_atr":         {"type": "float", "low": 0.0, "high": 0.3},

            # Confirmation gates
            "require_ma_align":       {"type": "categorical", "choices": [True, False]},
            "ma_fast":                {"type": "int",   "low": 13,  "high": 55,  "step": 1},
            "ma_slow":                {"type": "int",   "low": 60,  "high": 144, "step": 1},
            "require_htf_smc":        {"type": "categorical", "choices": [True, False]},
            "session_filter":         {"type": "categorical", "choices": [True, False]},
            "trade_start_hour":       {"type": "int",   "low": 6,   "high": 10},
            "trade_end_hour":         {"type": "int",   "low": 13,  "high": 20},

            # Reaction mode
            "reaction_max_wait":      {"type": "int",   "low": 1,   "high": 4},

            # SL/TP — wide range so optimizer can find sweet spot
            "sl_buffer_atr":          {"type": "float", "low": 0.2, "high": 1.5},
            "tp_atr_mult":            {"type": "float", "low": 0.2, "high": 2.5},
        },
    },

    ("XAUUSD", "trend_follow_h4"): {   # H4 trend-following, ride trend, exit on reversal
        "strategy_cls": XauTrendFollow,
        "param_space": {
            "entry_mode":   {"type": "categorical", "choices": ["ema_cross", "donchian"]},
            "ema_fast":     {"type": "int", "low": 8,  "high": 60, "step": 1},
            "ema_slow":     {"type": "int", "low": 30, "high": 150, "step": 5},
            "donchian_n":   {"type": "int", "low": 15, "high": 55, "step": 5},
            "atr_period":   {"type": "int", "low": 10, "high": 20},
            "sl_atr_mult":  {"type": "float", "low": 2.0, "high": 5.0},
            "allow_short":  {"type": "categorical", "choices": [True, False]},
            "min_hold_bars": {"type": "int", "low": 0, "high": 6},
        },
    },

    ("XAUUSD", "scalp_fade"): {   # Method A — tight TP scalp
        "strategy_cls": XauScalpFade,
        "param_space": {
            "ref_ema":          {"type": "int",   "low": 20,  "high": 89,  "step": 1},
            "stretch_atr":      {"type": "float", "low": 0.3, "high": 1.5},
            "use_rsi":          {"type": "categorical", "choices": [True, False]},
            "rsi_os":           {"type": "float", "low": 20.0, "high": 40.0, "step": 1.0},
            "rsi_ob":           {"type": "float", "low": 60.0, "high": 80.0, "step": 1.0},
            "session_filter":   {"type": "categorical", "choices": [True, False]},
            "trade_start_hour": {"type": "int",   "low": 6,   "high": 10},
            "trade_end_hour":   {"type": "int",   "low": 13,  "high": 20},
            "sl_atr_mult":      {"type": "float", "low": 0.5, "high": 2.5},
            "tp_rr":            {"type": "float", "low": 0.3, "high": 1.5},
        },
    },

    ("XAUUSD", "deep_pullback"): {   # Method B — deep limit, wide TP
        "strategy_cls": XauDeepPullback,
        "param_space": {
            "h1_ema_fast":      {"type": "categorical", "choices": [21, 34, 50]},
            "h1_ema_slow":      {"type": "categorical", "choices": [55, 89, 144, 200]},
            "require_h1_trend": {"type": "categorical", "choices": [True, False]},
            "swing_left":       {"type": "int",   "low": 3,   "high": 8},
            "swing_right":      {"type": "int",   "low": 3,   "high": 8},
            "touch_atr":        {"type": "float", "low": 0.05, "high": 0.5},
            "session_filter":   {"type": "categorical", "choices": [True, False]},
            "trade_start_hour": {"type": "int",   "low": 6,   "high": 10},
            "trade_end_hour":   {"type": "int",   "low": 13,  "high": 20},
            "sl_buffer_atr":    {"type": "float", "low": 0.2, "high": 1.0},
            "tp_rr":            {"type": "float", "low": 1.0, "high": 3.0},
            "min_sl_atr":       {"type": "float", "low": 0.3, "high": 1.0},
        },
    },

    ("XAUUSD", "ob_fvg_trend"): {   # BEST EDGE — OB∩FVG overlap, trend-aligned, wide TP
        "strategy_cls": XauObFvgTrend,
        "param_space": {
            "swing_left":  {"type": "int",   "low": 2,   "high": 6},
            "swing_right": {"type": "int",   "low": 2,   "high": 6},
            "ema_fast":    {"type": "categorical", "choices": [21, 34, 50]},
            "ema_slow":    {"type": "categorical", "choices": [89, 100, 144, 200]},
            "tol_atr":     {"type": "float", "low": 0.1, "high": 0.6},
            "sl_buf_atr":  {"type": "float", "low": 0.5, "high": 1.5},
            "tp_rr":       {"type": "float", "low": 2.0, "high": 4.0},
        },
    },

    ("XAUUSD", "reaction_level"): {
        "strategy_cls": XauReactionLevel,
        "param_space": {
            # HTF trend reference
            "h1_ema_fast":         {"type": "categorical", "choices": [21, 34, 50]},
            "h1_ema_slow":         {"type": "categorical", "choices": [55, 89, 144, 200]},
            "require_h1_trend":    {"type": "categorical", "choices": [True, False]},
            # Reaction-level detection
            "swing_left":          {"type": "int",   "low": 2,   "high": 5},
            "swing_right":         {"type": "int",   "low": 2,   "high": 5},
            "lookback":            {"type": "int",   "low": 200, "high": 1500, "step": 100},
            "band_atr_mult":       {"type": "float", "low": 0.1, "high": 0.6},
            "min_reactions":       {"type": "int",   "low": 2,   "high": 8},
            # Filters (the remaining options)
            "require_confirm_candle": {"type": "categorical", "choices": [True, False]},
            "require_m5_trend":    {"type": "categorical", "choices": [True, False]},
            "session_filter":      {"type": "categorical", "choices": [True, False]},
            "trade_start_hour":    {"type": "int",   "low": 6,   "high": 10},
            "trade_end_hour":      {"type": "int",   "low": 13,  "high": 20},
            "require_sr":          {"type": "categorical", "choices": [True, False]},
            "sr_round_step":       {"type": "categorical", "choices": [5.0, 10.0, 20.0]},
            "sr_tolerance_atr":    {"type": "float", "low": 0.3, "high": 2.0},
            # Geometry
            "sl_buffer_atr":       {"type": "float", "low": 0.1, "high": 0.8},
            "tp_rr":               {"type": "float", "low": 1.0, "high": 3.0},
            "min_sl_atr":          {"type": "float", "low": 0.2, "high": 1.0},
        },
    },

    ("XAUUSD", "ltf_ob_entry"): {
        "strategy_cls": XauLtfObEntry,
        "param_space": {
            "h1_ema_fast":         {"type": "categorical", "choices": [21, 34, 50]},
            "h1_ema_slow":         {"type": "categorical", "choices": [55, 89, 144, 200]},
            "require_h1_trend":    {"type": "categorical", "choices": [True, False]},
            "m5_swing_left":       {"type": "int",   "low": 2,   "high": 6},
            "m5_swing_right":      {"type": "int",   "low": 2,   "high": 6},
            "ob_proximity_pct":    {"type": "float", "low": 0.0005, "high": 0.005, "log": True},
            "require_m5_trend":    {"type": "categorical", "choices": [True, False]},
            "session_filter":      {"type": "categorical", "choices": [True, False]},
            "trade_start_hour":    {"type": "int",   "low": 6,   "high": 10},
            "trade_end_hour":      {"type": "int",   "low": 13,  "high": 20},
            "require_sr":          {"type": "categorical", "choices": [True, False]},
            "sr_round_step":       {"type": "categorical", "choices": [5.0, 10.0, 20.0]},
            "sr_tolerance_atr":    {"type": "float", "low": 0.3, "high": 2.0},
            "sl_buffer_atr":       {"type": "float", "low": 0.1, "high": 0.8},
            "tp_rr":               {"type": "float", "low": 1.0, "high": 3.0},
            "min_sl_atr":          {"type": "float", "low": 0.2, "high": 1.0},
        },
    },

    ("XAUUSD", "m15_ob_entry"): {
        "strategy_cls": XauM15ObEntry,
        "param_space": {
            "h1_ema_fast":         {"type": "categorical", "choices": [21, 34, 50]},
            "h1_ema_slow":         {"type": "categorical", "choices": [55, 89, 144, 200]},
            "require_h1_trend":    {"type": "categorical", "choices": [True, False]},
            "m5_swing_left":       {"type": "int",   "low": 2,   "high": 6},
            "m5_swing_right":      {"type": "int",   "low": 2,   "high": 6},
            "ob_proximity_pct":    {"type": "float", "low": 0.0005, "high": 0.005, "log": True},
            "require_m5_trend":    {"type": "categorical", "choices": [True, False]},
            "session_filter":      {"type": "categorical", "choices": [True, False]},
            "trade_start_hour":    {"type": "int",   "low": 6,   "high": 10},
            "trade_end_hour":      {"type": "int",   "low": 13,  "high": 20},
            "require_sr":          {"type": "categorical", "choices": [True, False]},
            "sr_round_step":       {"type": "categorical", "choices": [5.0, 10.0, 20.0]},
            "sr_tolerance_atr":    {"type": "float", "low": 0.3, "high": 2.0},
            "sl_buffer_atr":       {"type": "float", "low": 0.1, "high": 0.8},
            "tp_rr":               {"type": "float", "low": 1.0, "high": 3.0},
            "min_sl_atr":          {"type": "float", "low": 0.2, "high": 1.0},
        },
    },

    ("XAUUSD", "htf_trend_reversal"): {
        "strategy_cls": XauHtfTrendReversal,
        "param_space": {
            # HTF gate
            "h1_ema_fast":           {"type": "categorical", "choices": [21, 34, 50]},
            "h1_ema_slow":           {"type": "categorical", "choices": [55, 89, 144, 200]},
            "h1_adx_min":            {"type": "float", "low": 12.0, "high": 30.0, "step": 1.0},
            "require_h1_above_fast": {"type": "categorical", "choices": [True, False]},

            # Zone selection
            "zone_use_ema":          {"type": "categorical", "choices": [True, False]},
            "zone_ema":              {"type": "categorical", "choices": [21, 34, 50, 89]},
            "zone_ema_atr":          {"type": "float", "low": 0.2, "high": 1.0},
            "zone_use_swing":        {"type": "categorical", "choices": [True, False]},
            "swing_left":            {"type": "int",   "low": 3,   "high": 8},
            "swing_right":           {"type": "int",   "low": 3,   "high": 8},
            "zone_swing_atr":        {"type": "float", "low": 0.2, "high": 1.0},

            # Reversal patterns
            "pattern_engulfing":     {"type": "categorical", "choices": [True, False]},
            "pattern_pin_bar":       {"type": "categorical", "choices": [True, False]},
            "pattern_rsi_cross":     {"type": "categorical", "choices": [True, False]},
            "rsi_oversold":          {"type": "float", "low": 25.0, "high": 40.0, "step": 1.0},
            "rsi_overbought":        {"type": "float", "low": 60.0, "high": 75.0, "step": 1.0},

            # Session
            "session_filter":        {"type": "categorical", "choices": [True, False]},
            "trade_start_hour":      {"type": "int",   "low": 6,   "high": 10},
            "trade_end_hour":        {"type": "int",   "low": 13,  "high": 20},

            # SL/TP geometry
            "sl_buffer_atr":         {"type": "float", "low": 0.1, "high": 0.8},
            "tp_rr":                 {"type": "float", "low": 0.4, "high": 2.5},
            "sl_min_atr":            {"type": "float", "low": 0.3, "high": 1.5},
        },
    },

    ("XAUUSD", "ma34_cascade"): {
        "strategy_cls": XauMa34Cascade,
        "param_space": {
            "ma_fast":            {"type": "categorical", "choices": [13, 21, 34, 55]},
            "ma_slow":            {"type": "categorical", "choices": [50, 89, 100, 144, 200]},
            "pullback_atr":       {"type": "float", "low": 0.1, "high": 1.0},
            "min_tf_agree":       {"type": "int",   "low": 2,   "high": 4},
            "require_smc":        {"type": "categorical", "choices": [True, False]},
            "session_filter":     {"type": "categorical", "choices": [True, False]},
            "trade_start_hour":   {"type": "int",   "low": 6,   "high": 10},
            "trade_end_hour":     {"type": "int",   "low": 13,  "high": 20},
            "sl_atr_mult":        {"type": "float", "low": 0.5, "high": 2.5},
            "tp_atr_mult":        {"type": "float", "low": 0.3, "high": 2.5},
        },
    },

    ("XAUUSD", "mtf_smc_entry"): {
        "strategy_cls": XauMtfSmcEntry,
        "param_space": {
            # H4 OB
            "h4_swing_left":          {"type": "int",   "low": 2,   "high": 5},
            "h4_swing_right":         {"type": "int",   "low": 2,   "high": 5},
            "h4_ob_max_age_bars":     {"type": "int",   "low": 10,  "high": 60},

            # M15 FVG
            "m15_fvg_lookback":       {"type": "int",   "low": 4,   "high": 24},
            "m15_fvg_min_size_atr":   {"type": "float", "low": 0.05, "high": 0.5},

            # M5 BOS
            "m5_swing_left":          {"type": "int",   "low": 2,   "high": 5},
            "m5_swing_right":         {"type": "int",   "low": 2,   "high": 5},

            # Session
            "session_filter":         {"type": "categorical", "choices": [True, False]},
            "trade_start_hour":       {"type": "int",   "low": 6,   "high": 10},
            "trade_end_hour":         {"type": "int",   "low": 13,  "high": 20},

            # S/R confluence
            "require_sr_confluence":  {"type": "categorical", "choices": [True, False]},
            "sr_round_step":          {"type": "categorical", "choices": [5.0, 10.0, 20.0, 25.0]},
            "sr_tolerance_atr":       {"type": "float", "low": 0.3, "high": 2.0},

            # SL/TP — tp_mode controls geometry
            "sl_buffer_atr":          {"type": "float", "low": 0.1, "high": 1.0},
            "tp_mode":                {"type": "categorical", "choices": ["atr_absolute", "rr_relative"]},
            "tp_atr_mult":            {"type": "float", "low": 0.3, "high": 3.0},   # used when atr_absolute
            "tp_rr":                  {"type": "float", "low": 0.5, "high": 3.0},   # used when rr_relative
        },
    },

    ("XAUUSD", "smc_confluence"): {
        "strategy_cls": XauSmcConfluence,
        "param_space": {
            # Entry trigger (optimizer picks one per window)
            "entry_trigger":          {"type": "categorical",
                                       "choices": ["sweep", "ob_retest", "fvg_retest"]},

            # Swing detection
            "swing_left":             {"type": "int",   "low": 3,   "high": 8},
            "swing_right":            {"type": "int",   "low": 3,   "high": 8},

            # Trigger-specific
            "ob_proximity_pct":       {"type": "float", "low": 0.0005, "high": 0.003, "log": True},
            "fvg_max_age_bars":       {"type": "int",   "low": 10,  "high": 60},

            # Gates — each can be turned off independently
            "require_htf_smc":        {"type": "categorical", "choices": [True, False]},
            "require_ema_align":      {"type": "categorical", "choices": [True, False]},
            "session_filter":         {"type": "categorical", "choices": [True, False]},
            "vol_regime_filter":      {"type": "categorical", "choices": [True, False]},

            # EMA params (only used when require_ema_align=True)
            "ema_fast":               {"type": "int",   "low": 10,  "high": 30,  "step": 1},
            "ema_slow":               {"type": "int",   "low": 40,  "high": 100, "step": 5},

            # Session bounds (only used when session_filter=True)
            "trade_start_hour":       {"type": "int",   "low": 6,   "high": 9},
            "trade_end_hour":         {"type": "int",   "low": 13,  "high": 20},

            # Volatility regime (only used when vol_regime_filter=True)
            "vol_pctile_low":         {"type": "float", "low": 0.15, "high": 0.45},
            "vol_pctile_high":        {"type": "float", "low": 0.70, "high": 0.95},

            # SL/TP geometry — tight TP range so optimizer can explore high-WR scalp regime
            "sl_atr_mult":            {"type": "float", "low": 0.3, "high": 2.0},
            "tp_atr_mult":            {"type": "float", "low": 0.3, "high": 3.5},
        },
    },

    # ---------------- BTC ----------------

    ("BTCUSDT", "trend_following"): {
        "strategy_cls": BtcTrendFollowing,
        "param_space": {
            "htf_ema":                {"type": "int",   "low": 100, "high": 300, "step": 20},
            "htf_adx_min":            {"type": "float", "low": 18.0, "high": 35.0, "step": 1.0},
            "ltf_ema_pullback":       {"type": "int",   "low": 10,  "high": 50,  "step": 5},
            "pullback_proximity_atr": {"type": "float", "low": 0.1, "high": 1.0},
            "sl_atr_mult":            {"type": "float", "low": 0.8, "high": 3.5},
            "tp_atr_mult":            {"type": "float", "low": 0.3, "high": 6.0},
        },
    },

    ("BTCUSDT", "bollinger_squeeze"): {
        "strategy_cls": BtcBollingerSqueeze,
        "param_space": {
            "bb_period":              {"type": "int",   "low": 14,  "high": 40,  "step": 2},
            "bb_std":                 {"type": "float", "low": 1.5, "high": 3.0, "step": 0.1},
            "squeeze_lookback":       {"type": "int",   "low": 60,  "high": 240, "step": 20},
            "squeeze_pctile":         {"type": "float", "low": 0.05, "high": 0.40},
            "sl_atr_mult":            {"type": "float", "low": 0.5, "high": 2.5},
            "tp_atr_mult":            {"type": "float", "low": 0.3, "high": 5.0},
        },
    },

    ("BTCUSDT", "rsi_mean_revert"): {
        "strategy_cls": BtcRsiMeanRevert,
        "param_space": {
            "rsi_period":             {"type": "int",   "low": 7,   "high": 21},
            "rsi_oversold":           {"type": "float", "low": 15.0, "high": 30.0, "step": 1.0},
            "rsi_overbought":         {"type": "float", "low": 70.0, "high": 85.0, "step": 1.0},
            "htf_adx_max":            {"type": "float", "low": 15.0, "high": 25.0, "step": 1.0},
            "ema_target":             {"type": "int",   "low": 10,  "high": 40,  "step": 2},
            "sl_atr_mult":            {"type": "float", "low": 0.5, "high": 2.0},
            "min_tp_atr":             {"type": "float", "low": 0.1, "high": 1.0},
        },
    },

    ("BTCUSDT", "ob_fvg_trend"): {   # same OB∩FVG trend edge — positive 3/4 yrs on BTC
        "strategy_cls": XauObFvgTrend,
        "param_space": {
            "swing_left":  {"type": "int",   "low": 2,   "high": 6},
            "swing_right": {"type": "int",   "low": 2,   "high": 6},
            "ema_fast":    {"type": "categorical", "choices": [21, 34, 50]},
            "ema_slow":    {"type": "categorical", "choices": [89, 100, 144, 200]},
            "tol_atr":     {"type": "float", "low": 0.1, "high": 0.6},
            "sl_buf_atr":  {"type": "float", "low": 0.5, "high": 1.5},
            "tp_rr":       {"type": "float", "low": 2.0, "high": 4.0},
        },
    },

    ("BTCUSDT", "smc_orderblock"): {
        "strategy_cls": BtcSmcOrderBlock,
        "param_space": {
            "htf_swing_left":         {"type": "int",   "low": 2,   "high": 6},
            "htf_swing_right":        {"type": "int",   "low": 2,   "high": 6},
            "sl_atr_mult":            {"type": "float", "low": 0.5, "high": 2.0},
            "tp_atr_mult":            {"type": "float", "low": 0.3, "high": 6.0},
            "ob_proximity_pct":       {"type": "float", "low": 0.0005, "high": 0.01, "log": True},
        },
    },
}


def get_strategies_for_symbol(symbol: str) -> dict[str, dict]:
    """Return {strategy_name: {strategy_cls, param_space}} for a given symbol."""
    return {name: entry for (sym, name), entry in REGISTRY.items() if sym == symbol}
