"""Canonical local-date projection helper — GBO-R33/R34, CFG-R1a fail-closed.

The single normative owner of turning a UTC instant into the athlete's LOCAL calendar
date (spec doc 20 §3.8). These pin the pure projection contract the ingest write path and
the analytics day-buckets both consume:

* GBO-R33 — ``local_date`` is the calendar date of the UTC instant projected into the
  athlete's reference timezone (``athlete.reference_timezone``), using the timezone in
  effect at that instant per the as-of metadata ``reference_timezone_effective_from``. A
  non-UTC athlete whose UTC instant lands on a DIFFERENT local calendar day than the UTC
  day buckets to the LOCAL day, not the UTC day.
* GBO-R34 — recomputation is reproducible: the same UTC instant + the same as-of tz
  metadata always yields the same ``local_date``; an instant BEFORE a non-NULL
  ``reference_timezone_effective_from`` resolves against the prior projection, never the
  new tz (a relocation must not retroactively re-bucket prior days).
* CFG-R1a / CFG-R6 — a missing/blank reference timezone FAILS CLOSED (a typed error),
  NEVER a silent code-baked ``UTC`` default. No tz literal is baked into the code.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from wattwise_core.persistence.localdate import (
    MissingReferenceTimezone,
    project_local_date,
    project_local_wall_clock,
)

pytestmark = pytest.mark.unit

UTC = _dt.UTC


class _TzRef:
    """Minimal stand-in carrying only the two as-of tz fields the projector reads."""

    def __init__(self, tz: str | None, eff: _dt.datetime | None = None) -> None:
        self.reference_timezone = tz
        self.reference_timezone_effective_from = eff


def test_non_utc_instant_buckets_to_local_day_not_utc_day() -> None:
    """A UTC instant on a different LOCAL calendar day buckets to the LOCAL day (GBO-R33).

    2026-06-01 23:30Z in America/New_York (UTC-4 in June) is 19:30 on 2026-06-01 local —
    same date here. The decisive case is the OTHER side of midnight: 2026-06-02 03:00Z is
    2026-06-01 23:00 local, so it MUST bucket to 2026-06-01 (local), NOT 2026-06-02 (UTC).
    """
    athlete = _TzRef("America/New_York")
    instant = _dt.datetime(2026, 6, 2, 3, 0, tzinfo=UTC)
    assert instant.date() == _dt.date(2026, 6, 2)  # the UTC date (the wrong bucket)
    assert project_local_date(instant, athlete) == _dt.date(2026, 6, 1)  # LOCAL date


def test_positive_offset_crosses_forward_over_midnight() -> None:
    """A +tz instant late in the UTC day buckets to the NEXT local day (GBO-R33).

    2026-06-01 23:00Z in Asia/Tokyo (UTC+9) is 2026-06-02 08:00 local → buckets 06-02.
    """
    athlete = _TzRef("Asia/Tokyo")
    instant = _dt.datetime(2026, 6, 1, 23, 0, tzinfo=UTC)
    assert instant.date() == _dt.date(2026, 6, 1)
    assert project_local_date(instant, athlete) == _dt.date(2026, 6, 2)


def test_projection_is_reproducible_for_same_instant_and_tz() -> None:
    """Recomputing local_date for one instant + tz is stable (GBO-R34 reproducibility)."""
    athlete = _TzRef("America/New_York")
    instant = _dt.datetime(2026, 6, 2, 3, 0, tzinfo=UTC)
    first = project_local_date(instant, athlete)
    again = project_local_date(instant, athlete)
    assert first == again == _dt.date(2026, 6, 1)


def test_dst_spring_forward_boundary_buckets_correctly() -> None:
    """A DST spring-forward instant projects to the correct local day (GBO-R33, DST).

    US DST 2026 begins 2026-03-08 02:00 local (clocks jump to 03:00); the UTC offset goes
    from -5 to -4. The pre-jump instant 2026-03-08 04:30Z is 2026-03-07 23:30 EST → buckets
    to 03-07, not the UTC 03-08; the post-jump 2026-03-08 07:30Z is 03:30 EDT → still 03-08.
    """
    athlete = _TzRef("America/New_York")
    pre_jump = _dt.datetime(2026, 3, 8, 4, 30, tzinfo=UTC)
    assert pre_jump.date() == _dt.date(2026, 3, 8)  # UTC date
    assert project_local_date(pre_jump, athlete) == _dt.date(2026, 3, 7)  # local, offset -5
    post_jump = _dt.datetime(2026, 3, 8, 7, 30, tzinfo=UTC)
    assert project_local_date(post_jump, athlete) == _dt.date(2026, 3, 8)  # local, offset -4


def test_dst_fall_back_boundary_buckets_correctly() -> None:
    """A DST fall-back instant projects to the correct local day (GBO-R33, DST)."""
    athlete = _TzRef("America/New_York")
    # 2026-11-01 fall-back: 2026-11-01 03:30Z = 2026-10-31 23:30 EDT (offset -4) → 10-31.
    instant = _dt.datetime(2026, 11, 1, 3, 30, tzinfo=UTC)
    assert instant.date() == _dt.date(2026, 11, 1)
    assert project_local_date(instant, athlete) == _dt.date(2026, 10, 31)


def test_tz_change_over_time_uses_as_of_effective_from() -> None:
    """An instant BEFORE a relocation resolves against the PRIOR projection (GBO-R34 as-of).

    The athlete relocated to Asia/Tokyo effective 2026-06-15 00:00Z. A PRIOR activity at
    2026-06-02 03:00Z (before the relocation) MUST NOT be re-bucketed under Tokyo (which
    would give 06-02 12:00 → 06-02); it keeps the projection it had under the prior tz.
    The current-tz path applies only at/after effective_from.
    """
    eff = _dt.datetime(2026, 6, 15, 0, 0, tzinfo=UTC)
    athlete = _TzRef("Asia/Tokyo", eff)
    # Instant AT/AFTER effective_from → current (Tokyo) tz: 2026-06-20 23:00Z = 06-21 08:00.
    after = _dt.datetime(2026, 6, 20, 23, 0, tzinfo=UTC)
    assert project_local_date(after, athlete) == _dt.date(2026, 6, 21)
    # Instant BEFORE effective_from → NOT re-bucketed under the new tz. The persisted prior
    # projection is authoritative; the projector returns it rather than the new-tz date.
    before = _dt.datetime(2026, 6, 2, 3, 0, tzinfo=UTC)
    assert project_local_date(before, athlete, prior_local_date=_dt.date(2026, 6, 1)) == _dt.date(
        2026, 6, 1
    )


def test_missing_reference_timezone_fails_closed() -> None:
    """A missing reference timezone FAILS CLOSED, never a silent UTC default (CFG-R1a/R6)."""
    instant = _dt.datetime(2026, 6, 2, 3, 0, tzinfo=UTC)
    with pytest.raises(MissingReferenceTimezone):
        project_local_date(instant, _TzRef(None))
    with pytest.raises(MissingReferenceTimezone):
        project_local_date(instant, _TzRef("   "))


def test_unknown_reference_timezone_fails_closed() -> None:
    """An unresolvable IANA zone fails closed, never a guessed offset (CFG-R6 fail-closed)."""
    instant = _dt.datetime(2026, 6, 2, 3, 0, tzinfo=UTC)
    with pytest.raises(MissingReferenceTimezone):
        project_local_date(instant, _TzRef("Not/AZone"))


def test_naive_instant_is_treated_as_utc() -> None:
    """A naive instant is read as UTC (GBO-R32: all stored instants are UTC)."""
    athlete = _TzRef("America/New_York")
    naive = _dt.datetime(2026, 6, 2, 3, 0)
    assert project_local_date(naive, athlete) == _dt.date(2026, 6, 1)


def test_local_wall_clock_preserves_local_time_for_display() -> None:
    """``start_time_local`` carries the LOCAL wall-clock time (GBO-R13/§3.8 display)."""
    athlete = _TzRef("America/New_York")
    instant = _dt.datetime(2026, 6, 2, 3, 0, tzinfo=UTC)
    local = project_local_wall_clock(instant, athlete)
    # 2026-06-02 03:00Z = 2026-06-01 23:00 local (offset -4); the wall-clock fields are local.
    assert (local.year, local.month, local.day, local.hour, local.minute) == (2026, 6, 1, 23, 0)
    assert local.date() == project_local_date(instant, athlete)
