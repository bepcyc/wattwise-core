"""Run-scoped reducer unit tests for the durable-resume turn boundary (CKPT-R5).

These exercise the contracts.py reducers that make ONE durable checkpoint reusable across
many turns without leaking turn-1 evidence into turn-2 (the reverted force-degrade bug):

* F-MONO: :func:`_turn_monotonic` is monotonic WITHIN a turn (a mid-turn decrease raises,
  preserving the EVAL-R7 bounded-termination guard) but a decrease to the sentinel floor 0
  is allowed (the head node's new-turn reset). Verified on all three run-scoped counters
  (``node_visits`` is INCLUDED — else the cross-turn reset raises on it).
* F-RESET: the turn-keyed ``retrieved`` / ``coverage_gaps`` reducers SELF-RESET when an
  incoming write names a different turn (the leak backstop), and merge/union within a turn.
* The stamp/reader helpers round-trip and keep the readers' plain dict/set view intact.

Offline + self-contained: pure reducer functions, no graph/saver/DB.
"""

from __future__ import annotations

import pytest

from wattwise_core.agent.contracts import (
    COVERAGE_GAPS_TURN_PREFIX,
    RETRIEVED_TURN_KEY,
    TURN_COUNTER_FLOOR,
    _last_write_wins,
    _turn_keyed_merge,
    _turn_keyed_union,
    _turn_monotonic,
    stamp_coverage_gaps,
    stamp_retrieved,
    turn_gaps,
    turn_records,
)

pytestmark = pytest.mark.unit


# --- F-MONO: turn-monotonic counter reducer ---------------------------------------------


def test_turn_monotonic_allows_increase_within_turn() -> None:
    """An increase (or no change) within a turn is accepted, like the strict counter."""
    assert _turn_monotonic(0, 1) == 1
    assert _turn_monotonic(4, 5) == 5
    assert _turn_monotonic(5, 5) == 5


def test_turn_monotonic_mid_turn_decrease_raises() -> None:
    """F-MONO: a mid-turn decrease (5 -> 3) still raises (EVAL-R7 monotonic guard).

    The bounded-termination guard must hold WITHIN a turn so a reflect/redraft/visit loop
    cannot evade its budget by rewinding the counter.
    """
    with pytest.raises(ValueError, match="mid-turn"):
        _turn_monotonic(5, 3)


def test_turn_monotonic_reset_to_floor_allowed() -> None:
    """F-MONO: a decrease to the sentinel floor 0 is allowed (head-node new-turn reset).

    ``(5, 0) -> 0`` is the single head node resetting the counter at a turn boundary; any
    OTHER decrease is rejected. ``(5, 6) -> 6`` confirms a post-reset increase is fine.
    """
    assert _turn_monotonic(5, 0) == TURN_COUNTER_FLOOR
    assert _turn_monotonic(5, 0) == 0
    assert _turn_monotonic(5, 6) == 6
    # A decrease to 1 (not the floor) is NOT a reset and still raises.
    with pytest.raises(ValueError, match="mid-turn"):
        _turn_monotonic(5, 1)


def test_turn_monotonic_applies_to_all_three_counters() -> None:
    """F-MONO: identical semantics on node_visits / reflection_count / redraft_count.

    All three run-scoped counters share :func:`_turn_monotonic`; node_visits is INCLUDED so
    the cross-turn reset does not raise on it.
    """
    for stored in (3, 7, 12):
        assert _turn_monotonic(stored, 0) == 0
        assert _turn_monotonic(stored, stored + 1) == stored + 1
        with pytest.raises(ValueError):
            _turn_monotonic(stored, stored - 1)


def test_run_epoch_last_write_wins() -> None:
    """``run_epoch`` keeps the last non-empty write (head node stamps it on reset)."""
    assert _last_write_wins("", "turn-1") == "turn-1"
    assert _last_write_wins("turn-1", "turn-2") == "turn-2"
    assert _last_write_wins("turn-2", "") == "turn-2"


# --- F-RESET: turn-keyed retrieved reducer ----------------------------------------------


def test_retrieved_same_turn_merges() -> None:
    """Within one turn, ``retrieved`` writes merge by key (the _keyed_merge behaviour)."""
    first = stamp_retrieved("t1", {"a": {"v": 1}})
    second = stamp_retrieved("t1", {"b": {"v": 2}})
    merged = _turn_keyed_merge(first, second)
    assert turn_records(merged) == {"a": {"v": 1}, "b": {"v": 2}}
    assert merged[RETRIEVED_TURN_KEY] == "t1"


def test_retrieved_different_turn_replaces() -> None:
    """F-RESET: a turn-2 ``retrieved`` write DROPS every turn-1 record (leak backstop).

    The reducer is the backstop against an evidence leak: even if a head-node reset were
    missed, turn-2's first stamped write replaces (does not union) the stored turn-1 records.
    """
    turn1 = stamp_retrieved("t1", {"old_rec": {"v": 1}, "other": {"v": 9}})
    turn2 = stamp_retrieved("t2", {"new_rec": {"v": 2}})
    merged = _turn_keyed_merge(turn1, turn2)
    assert turn_records(merged) == {"new_rec": {"v": 2}}
    assert "old_rec" not in turn_records(merged)
    assert merged[RETRIEVED_TURN_KEY] == "t2"


def test_retrieved_first_write_into_empty_channel() -> None:
    """The first stamped write into the seeded empty dict establishes the turn marker."""
    merged = _turn_keyed_merge({}, stamp_retrieved("t1", {"a": {"v": 1}}))
    assert turn_records(merged) == {"a": {"v": 1}}
    assert merged[RETRIEVED_TURN_KEY] == "t1"


def test_retrieved_unstamped_update_inherits_stored_turn() -> None:
    """An unstamped same-channel update merges and keeps the stored turn (no reset)."""
    stored = stamp_retrieved("t1", {"a": {"v": 1}})
    merged = _turn_keyed_merge(stored, {"b": {"v": 2}})
    assert turn_records(merged) == {"a": {"v": 1}, "b": {"v": 2}}
    assert merged[RETRIEVED_TURN_KEY] == "t1"


def test_retrieved_marker_invisible_to_payload_reader() -> None:
    """The turn marker rides in-band but :func:`turn_records` hides it from readers."""
    stamped = stamp_retrieved("t1", {"a": {"v": 1}})
    assert RETRIEVED_TURN_KEY in stamped
    assert RETRIEVED_TURN_KEY not in turn_records(stamped)
    # Re-stamping never duplicates or strands a prior marker.
    restamped = stamp_retrieved("t2", stamped)
    assert restamped[RETRIEVED_TURN_KEY] == "t2"
    assert turn_records(restamped) == {"a": {"v": 1}}


# --- F-RESET: turn-keyed coverage_gaps reducer ------------------------------------------


def test_coverage_gaps_same_turn_unions() -> None:
    """Within one turn, ``coverage_gaps`` writes union (the _set_union behaviour)."""
    first = stamp_coverage_gaps("t1", {"gap_a"})
    second = stamp_coverage_gaps("t1", {"gap_b"})
    merged = _turn_keyed_union(first, second)
    assert turn_gaps(merged) == {"gap_a", "gap_b"}


def test_coverage_gaps_different_turn_replaces() -> None:
    """F-RESET: a turn-2 ``coverage_gaps`` write drops every turn-1 gap (leak backstop)."""
    turn1 = stamp_coverage_gaps("t1", {"stale_gap"})
    turn2 = stamp_coverage_gaps("t2", {"fresh_gap"})
    merged = _turn_keyed_union(turn1, turn2)
    assert turn_gaps(merged) == {"fresh_gap"}
    assert "stale_gap" not in turn_gaps(merged)


def test_coverage_gaps_marker_invisible_to_reader() -> None:
    """The coverage-gaps turn token is stripped by :func:`turn_gaps` and never duplicated."""
    stamped = stamp_coverage_gaps("t1", {"gap_a"})
    assert any(g.startswith(COVERAGE_GAPS_TURN_PREFIX) for g in stamped)
    assert turn_gaps(stamped) == {"gap_a"}
    restamped = stamp_coverage_gaps("t2", stamped)
    markers = [g for g in restamped if g.startswith(COVERAGE_GAPS_TURN_PREFIX)]
    assert markers == [f"{COVERAGE_GAPS_TURN_PREFIX}t2"]
    assert turn_gaps(restamped) == {"gap_a"}


def test_coverage_gaps_first_write_into_empty_channel() -> None:
    """The first stamped gap write into the seeded empty set establishes the turn marker."""
    merged = _turn_keyed_union(set(), stamp_coverage_gaps("t1", {"gap_a"}))
    assert turn_gaps(merged) == {"gap_a"}
    assert _turn_keyed_union(merged, stamp_coverage_gaps("t1", {"gap_b"}))  # same turn unions
