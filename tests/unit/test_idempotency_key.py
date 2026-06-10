"""The deterministic per-turn idempotency key + dedup window (CKPT-R4).

CKPT-R4: submitting the SAME request turn twice MUST resume/return the existing run rather
than starting a duplicate, WITHIN A CONFIGURABLE DEDUP WINDOW. The fix derives the durable
thread's conversation id DETERMINISTICALLY from the turn (athlete + trigger + request +
window bucket) so two identical turns inside the window collapse onto one thread (where the
saver's ``resolve_idempotent`` then finds the existing run); a random id minted per turn — the
deviated behaviour — would spawn a duplicate thread on every re-submission.
"""

from __future__ import annotations

import pytest

from wattwise_core.agent.projection import idempotent_conversation_id

pytestmark = pytest.mark.unit

ATHLETE = "00000000-0000-7000-8000-00000000000a"


def _key(*, request_text: str, window: int, now: float) -> str:
    return idempotent_conversation_id(
        athlete_id=ATHLETE,
        trigger="user_turn",
        request_text=request_text,
        dedup_window_seconds=window,
        now=now,
    )


def test_same_turn_in_same_window_resolves_to_same_id() -> None:
    """The SAME turn re-submitted inside the window maps to the SAME thread (CKPT-R4).

    Both timestamps fall in the 60s bucket ``[1020, 1080)`` (``floor(t/60) == 17``).
    """
    a = _key(request_text="How am I doing?", window=60, now=1_020.0)
    b = _key(request_text="How am I doing?", window=60, now=1_079.0)
    assert a == b


def test_same_turn_after_window_resolves_to_new_id() -> None:
    """Once the window elapses the bucket advances, so a later turn opens a NEW thread."""
    a = _key(request_text="How am I doing?", window=60, now=1_020.0)
    later = _key(request_text="How am I doing?", window=60, now=1_090.0)  # bucket 18
    assert a != later


def test_different_question_resolves_to_different_id() -> None:
    """A DIFFERENT turn (different text) is not the same turn — distinct thread."""
    a = _key(request_text="How am I doing?", window=60, now=1_020.0)
    other = _key(request_text="What about tomorrow?", window=60, now=1_020.0)
    assert a != other


def test_window_is_configurable_not_hardcoded() -> None:
    """The window is a parameter (CKPT-R4 'configurable'): widening it widens the bucket.

    With a wide window two timestamps fall in one bucket (same id); with a narrow window the
    same two timestamps straddle two buckets (different id) — proving the window governs dedup.
    """
    wide_a = _key(request_text="q", window=3600, now=1_000.0)
    wide_b = _key(request_text="q", window=3600, now=2_500.0)
    assert wide_a == wide_b  # both within one 3600s bucket
    narrow_a = _key(request_text="q", window=600, now=1_000.0)
    narrow_b = _key(request_text="q", window=600, now=2_500.0)
    assert narrow_a != narrow_b  # straddle two 600s buckets


def test_zero_window_is_pure_content_dedup() -> None:
    """A non-positive window disables time-bucketing: identical turns always dedup."""
    a = _key(request_text="q", window=0, now=1_000.0)
    b = _key(request_text="q", window=0, now=9_999_999.0)
    assert a == b
