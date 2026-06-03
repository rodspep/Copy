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


def test_fresh_at_exact_window_boundary_reconsiders():
    # lag == window is still "fresh" (<=), block_age == window is "expired" (>=)
    prev = {"skipped": "voided", "at": "x"}
    assert cp._reconsider_signal(prev, lag_sec=WIN, block_age_sec=WIN, window_sec=WIN) is True
