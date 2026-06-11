"""Intervals.icu discover/map helper edges (ADP-R5/R6/R7, MAP-R4, IDS-R3).

Pure-function branch coverage for the adapter's stateless sync helpers: cursor
decoding rejects an unknown stage; an unusable last-modified hint never silently
skips a ref (fail-closed toward re-checking); a listing row without an id yields no
ref; an absent/naive/unparseable source instant is a typed gap, never a guess.
Offline, no I/O (TST-R1).
"""

from __future__ import annotations

import datetime as _dt

import pytest

from wattwise_core.ingestion.adapters import _intervals_sync as isync
from wattwise_core.ingestion.adapters._intervals_map import sport_code
from wattwise_core.ingestion.adapters.intervals_icu import _parse_utc

pytestmark = pytest.mark.unit


def test_parse_cursor_rejects_unknown_stage() -> None:
    """ADP-R7: an opaque cursor with an unknown stage fails closed with ValueError."""
    with pytest.raises(ValueError, match="unknown discover cursor stage"):
        isync.parse_cursor("bogus:3")


def test_parse_cursor_start_and_offset_forms() -> None:
    """ADP-R7: None means the start; ``stage:offset`` round-trips; a bare stage is 0."""
    assert isync.parse_cursor(None) == ("act", 0)
    assert isync.parse_cursor("well:7") == ("well", 7)
    assert isync.parse_cursor("act:") == ("act", 0)


def test_parse_hint_unusable_values_return_none() -> None:
    """ADP-R6: empty / unparseable / naive hints are unusable -> None (never a guess)."""
    assert isync._parse_hint(None) is None
    assert isync._parse_hint("") is None
    assert isync._parse_hint("not-a-date") is None
    assert isync._parse_hint("2026-06-01T10:00:00") is None  # naive -> rejected


def test_activity_refs_skip_only_provably_current_rows() -> None:
    """ADP-R6: a row without an id yields no ref; a hint-less row is always yielded."""
    watermark = _dt.datetime(2026, 6, 1, tzinfo=_dt.UTC)
    raw: list[dict[str, object]] = [
        {"start_date": "2026-05-01T00:00:00Z"},  # no id -> skipped entirely
        {"id": "a1", "start_date": "2026-05-01T00:00:00Z"},  # provably current -> skipped
        {"id": "a2"},  # hint-less -> must be yielded (cannot prove current)
        {"id": "a3", "start_date": "2026-06-02T00:00:00Z"},  # past watermark -> yielded
    ]
    refs = isync.activity_refs(raw, watermark)
    assert [r.source_native_id for r in refs] == ["a2", "a3"]
    assert refs[0].last_modified is None


def test_wellness_refs_carry_no_change_hint() -> None:
    """ADP-R6: wellness day refs are always yielded with last_modified=None."""
    refs = isync.wellness_refs(["2026-06-01", "2026-06-02"])
    assert [r.source_native_id for r in refs] == ["2026-06-01", "2026-06-02"]
    assert all(r.last_modified is None for r in refs)


def test_sport_code_maps_absent_type_to_other() -> None:
    """MAP-R4: an absent source activity type maps to the canonical 'other' code."""
    assert sport_code(None) == "other"
    assert sport_code("Ride") == "cycling"


def test_parse_utc_rejects_unparseable_and_naive_instants() -> None:
    """IDS-R3/MAP-R3: unparseable or naive source instants are typed gaps (None)."""
    assert _parse_utc(None) is None
    assert _parse_utc("garbage") is None
    assert _parse_utc("2026-06-01T10:00:00") is None  # naive -> rejected
    parsed = _parse_utc("2026-06-01T10:00:00Z")
    assert parsed == _dt.datetime(2026, 6, 1, 10, 0, tzinfo=_dt.UTC)
