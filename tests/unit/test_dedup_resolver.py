"""Unit tests for the OSS dedup/conflict resolver (CONF-R2, MAP-R10, DEDUP-R7)."""

from __future__ import annotations

import datetime as _dt

import pytest

from wattwise_core.domain.candidate import FieldCandidate
from wattwise_core.domain.enums import Fidelity
from wattwise_core.ingestion.dedup import resolve_activity_identity, resolve_field

UTC = _dt.UTC


def _fc(value: object, tier: Fidelity, sid: str, **kw: object) -> FieldCandidate:
    return FieldCandidate(value=value, trust_tier=tier, source_descriptor_id=sid, **kw)  # type: ignore[arg-type]


def test_no_contributor_returns_none() -> None:
    """No candidate -> None so the caller records a typed gap, never a zero (CONF-R5)."""
    assert resolve_field([]) is None


def test_trust_tier_wins_first() -> None:
    """Highest fidelity wins regardless of recency/confidence (CONF-R2 step 1)."""
    raw = _fc(250.0, Fidelity.RAW_STREAM, "b", confidence=0.5)
    summ = _fc(260.0, Fidelity.SUMMARY_ONLY, "a", confidence=1.0)
    res = resolve_field([summ, raw])
    assert res is not None
    assert res.value == 250.0
    assert res.winning_source_descriptor_id == "b"


def test_confidence_breaks_tie_within_tier() -> None:
    """Within one tier, higher confidence wins (CONF-R2 step 2)."""
    a = _fc(100.0, Fidelity.DEVICE_COMPUTED, "a", confidence=0.6)
    b = _fc(110.0, Fidelity.DEVICE_COMPUTED, "b", confidence=0.9)
    res = resolve_field([a, b])
    assert res is not None
    assert res.value == 110.0


def test_recency_then_completeness_then_stable_tiebreak() -> None:
    """Recency, then completeness, then lowest source id as the final stable tiebreak."""
    older = _fc(
        1.0, Fidelity.MODELED, "z", confidence=1.0,
        observed_at=_dt.datetime(2026, 1, 1, tzinfo=UTC), completeness=1.0,
    )
    newer = _fc(
        2.0, Fidelity.MODELED, "y", confidence=1.0,
        observed_at=_dt.datetime(2026, 2, 1, tzinfo=UTC), completeness=1.0,
    )
    assert resolve_field([older, newer]).value == 2.0  # type: ignore[union-attr]

    # Identical on tiers 1-4 -> lowest source_descriptor_id wins (byte-reproducible).
    a = _fc(5.0, Fidelity.SUMMARY_ONLY, "aaa")
    b = _fc(6.0, Fidelity.SUMMARY_ONLY, "bbb")
    assert resolve_field([b, a]).winning_source_descriptor_id == "aaa"  # type: ignore[union-attr]


def test_disputed_flag_when_materially_disagree() -> None:
    """disputed=True when two numeric candidates differ beyond tolerance (CONF-R5)."""
    a = _fc(200.0, Fidelity.RAW_STREAM, "a")
    b = _fc(260.0, Fidelity.RAW_STREAM, "b")
    res = resolve_field([a, b], dispute_tolerance=0.1)
    assert res is not None
    assert res.disputed is True
    # Still picks the best, never averages.
    assert res.value in (200.0, 260.0)


def test_resolution_is_deterministic() -> None:
    """Same candidate set -> same winner regardless of input order (CONF-R4)."""
    cs = [
        _fc(1.0, Fidelity.MODELED, "m"),
        _fc(2.0, Fidelity.RAW_STREAM, "r"),
        _fc(3.0, Fidelity.SUMMARY_ONLY, "s"),
    ]
    first = resolve_field(cs)
    second = resolve_field(list(reversed(cs)))
    assert first == second


def test_identity_fingerprint_matches_regardless_of_window() -> None:
    """A shared strong fingerprint matches even outside the time window (MAP-R10)."""
    t1 = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
    t2 = _dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC)  # 1h apart
    assert resolve_activity_identity(
        t1, 3600, "cycling", "fit-abc", t2, 3600, "cycling", "fit-abc"
    )


def test_identity_window_and_duration_tolerance() -> None:
    """Within window + duration tolerance and same sport -> match; else not (MAP-R10)."""
    t1 = _dt.datetime(2026, 6, 1, 8, 0, 0, tzinfo=UTC)
    t2 = _dt.datetime(2026, 6, 1, 8, 1, 30, tzinfo=UTC)  # 90 s < 120 s window
    assert resolve_activity_identity(t1, 3600, "cycling", None, t2, 3650, "cycling", None)
    # Different sport never matches without a fingerprint.
    assert not resolve_activity_identity(t1, 3600, "cycling", None, t1, 3600, "running", None)
    # Outside the start window with no fingerprint -> no match.
    t3 = _dt.datetime(2026, 6, 1, 8, 5, 0, tzinfo=UTC)  # 300 s > window
    assert not resolve_activity_identity(t1, 3600, "cycling", None, t3, 3600, "cycling", None)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
