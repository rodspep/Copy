"""Dedup / re-post handling for the UG copier (_reconsider_signal).

REGRESSION: UG re-posts the SAME entry/SL/TP hours apart as genuinely new signals.
A morning post that was stale-skipped (soft block) used to silently swallow the
evening re-post (no order, no notification). _reconsider_signal must let a FRESH
re-post through after the window, while still:
  - never double-placing a setup we already placed (hard block), and
  - suppressing UG's burst re-posts seconds apart + per-poll re-reads of the same line.
"""
from __future__ import annotations

import scripts.ug_copier as cp

WIN = 900.0          # staleness / re-eval window (15 min), in seconds


def test_first_seen_evaluates():
    assert cp._reconsider_signal(None, lag_sec=2, block_age_sec=0, window_sec=WIN) is True


def test_placed_block_never_reconsiders():
    prev = {"placed": [{"trade_id": 1, "ticket": 5, "leg": "tp1"}]}
    # even a fresh re-post, even long after → never re-place the same levels
    assert cp._reconsider_signal(prev, lag_sec=2, block_age_sec=99999, window_sec=WIN) is False


def test_placed_empty_list_still_hard_block():
    # all legs reconciled to already-tracked tickets → {"placed": []} (falsy!) must STILL
    # be a hard block by key presence, not be reconsidered as a soft block.
    prev = {"placed": []}
    assert cp._reconsider_signal(prev, lag_sec=2, block_age_sec=WIN + 1, window_sec=WIN) is False


def test_placing_block_never_reconsiders():
    # crashed mid-place: some legs may have landed → must not double-place
    prev = {"status": "placing", "at": "2026-06-03T00:00:00+00:00"}
    assert cp._reconsider_signal(prev, lag_sec=2, block_age_sec=99999, window_sec=WIN) is False


def test_soft_stale_fresh_repost_after_window_reconsiders():
    # THE BUG: morning stale-skip (soft), evening fresh re-post → must reconsider
    prev = {"stale": 7337.0, "at": "2026-06-03T09:47:00+00:00"}
    assert cp._reconsider_signal(prev, lag_sec=2, block_age_sec=9 * 3600, window_sec=WIN) is True


def test_soft_skip_fresh_repost_after_window_reconsiders():
    prev = {"skipped": "voided", "at": "2026-06-03T09:47:00+00:00"}
    assert cp._reconsider_signal(prev, lag_sec=5, block_age_sec=WIN + 1, window_sec=WIN) is True


def test_soft_block_burst_repost_within_window_suppressed():
    # UG's 12s-apart re-posts: the 2nd must NOT re-act (block too recent)
    prev = {"skipped": "voided", "at": "2026-06-03T09:47:00+00:00"}
    assert cp._reconsider_signal(prev, lag_sec=12, block_age_sec=12, window_sec=WIN) is False


def test_soft_block_stale_repost_not_fresh_suppressed():
    # the same OLD line re-read every poll: lag > window (not fresh) → no churn
    prev = {"stale": 7337.0, "at": "2026-06-03T09:47:00+00:00"}
    assert cp._reconsider_signal(prev, lag_sec=7337, block_age_sec=9 * 3600, window_sec=WIN) is False


def test_soft_block_bad_ts_negative_lag_suppressed():
    prev = {"stale": -1.0, "at": "2026-06-03T09:47:00+00:00"}
    assert cp._reconsider_signal(prev, lag_sec=-1, block_age_sec=9 * 3600, window_sec=WIN) is False


def test_soft_loss_halt_fresh_after_window_reconsiders():
    prev = {"loss_halt": True, "at": "2026-06-03T09:47:00+00:00"}
    assert cp._reconsider_signal(prev, lag_sec=2, block_age_sec=WIN + 100, window_sec=WIN) is True


def test_soft_place_failed_fresh_after_window_reconsiders():
    prev = {"status": "place_failed", "at": "2026-06-03T09:47:00+00:00"}
    assert cp._reconsider_signal(prev, lag_sec=2, block_age_sec=WIN + 100, window_sec=WIN) is True


def _sig(ts, lo=4459, hi=4462, sl=4472, tp1=50, direction="short"):
    return {"direction": direction, "entry_low": lo, "entry_high": hi, "sl": sl,
            "ts": ts, "tps_pip": {1: tp1, 2: 100, 3: 150}}


def test_freshest_per_key_collapses_old_before_fresh():
    # The Codex finding: feed in arrival order has an OLD same-key line before a FRESH
    # re-post. _freshest_per_key must yield only the FRESH one (so the old line can't
    # stale-mark the key and suppress the fresh re-post).
    old = _sig("2026-06-03T12:17:18+00:00")
    fresh = _sig("2026-06-03T12:46:30+00:00")
    out = cp._freshest_per_key([old, fresh])           # file order: old first
    assert len(out) == 1 and out[0]["ts"] == fresh["ts"]


def test_freshest_per_key_keeps_distinct_keys():
    a = _sig("2026-06-03T12:00:00+00:00", lo=4459, hi=4462, sl=4472)
    b = _sig("2026-06-03T12:01:00+00:00", lo=4400, hi=4403, sl=4413)   # different levels
    out = cp._freshest_per_key([a, b])
    assert len(out) == 2


def test_freshest_per_key_orders_by_selected_line_not_first_seen():
    # [old A, distinct B, fresh re-post A]: collapse must keep fresh A + B, and process B
    # BEFORE A (A's SELECTED line is the 12:46 repost at index 2, after B at index 1) — so a
    # later re-post can't pre-empt an earlier distinct signal for max-open capacity.
    old_a = _sig("2026-06-03T12:00:00+00:00", lo=4459, hi=4462, sl=4472)
    b = _sig("2026-06-03T12:30:00+00:00", lo=4400, hi=4403, sl=4413)
    fresh_a = _sig("2026-06-03T12:46:00+00:00", lo=4459, hi=4462, sl=4472)
    out = cp._freshest_per_key([old_a, b, fresh_a])
    assert len(out) == 2
    assert out[0]["ts"] == b["ts"]                     # B first (selected index 1)
    assert out[1]["ts"] == fresh_a["ts"]               # fresh A second (selected index 2)


def test_freshest_per_key_bad_ts_prefers_later():
    a = _sig("garbage-ts")
    b = _sig("also-bad")
    out = cp._freshest_per_key([a, b])                 # unparseable → later (file order) wins
    assert len(out) == 1 and out[0] is b


def test_incident_891s_stale_not_reconsidered_with_tight_window():
    # THE REAL INCIDENT (2min window): a place_failed signal (AutoTrading off) was
    # reconsidered on restart when block_age crossed the window while lag was still under
    # the OLD loose 15min window → placed 891s late ("chắc chắn lỗ"). With a 2min window
    # (120s), lag 891s is NOT fresh → suppressed, never placed stale.
    prev = {"status": "place_failed", "at": "2026-06-03T12:17:07+00:00"}
    assert cp._reconsider_signal(prev, lag_sec=891, block_age_sec=902, window_sec=120) is False


def test_incident_fresh_repost_after_tight_window_still_places():
    # The legit fresh re-post (same levels, ~29min later, lag ~3s) MUST still reconsider
    # → place, with the tight 2min window.
    prev = {"status": "place_failed", "at": "2026-06-03T12:17:07+00:00"}
    assert cp._reconsider_signal(prev, lag_sec=3, block_age_sec=29 * 60, window_sec=120) is True


def test_fresh_at_exact_window_boundary_reconsiders():
    # lag == window is still "fresh" (<=), block_age == window is "expired" (>=)
    prev = {"skipped": "voided", "at": "x"}
    assert cp._reconsider_signal(prev, lag_sec=WIN, block_age_sec=WIN, window_sec=WIN) is True
