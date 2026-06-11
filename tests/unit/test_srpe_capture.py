"""Athlete-reported exertion capture at the canonical mappers (SRPE-R1, MAP-R2/R3/R5).

The two activity mappers (file upload, intervals.icu) must carry the athlete-reported
``perceived_exertion`` (normalized to the canonical CR-10 scale) and ``feel`` (the 1..5
ordinal) into the canonical payload, with every malformed or out-of-range report dropped
to a typed gap ``None`` (MAP-R5) — never clamped into validity, because a clamped
exertion report is a fabricated one.
"""

from __future__ import annotations

import datetime as _dt
import math

import pytest

from wattwise_core.ingestion.adapters._asbo import ActivityAsbo
from wattwise_core.ingestion.adapters._intervals_asbo import IntervalsActivityAsbo
from wattwise_core.ingestion.adapters._map_activity import (
    activity_payload,
    build_streams,
    feel_value,
    rpe_value,
)
from wattwise_core.ingestion.adapters.intervals_icu import _activity_payload

pytestmark = pytest.mark.unit

_START = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=_dt.UTC)


def test_rpe_cr10_values_pass_through_unscaled() -> None:
    """A report already on the CR-10 scale ([0, 10]) is carried verbatim (MAP-R3)."""
    assert rpe_value(7) == 7.0
    assert rpe_value(7.5) == 7.5
    assert rpe_value(0) == 0.0
    assert rpe_value(10) == 10.0  # boundary: CR-10 maximum, NOT the percent encoding


def test_rpe_percent_encoding_normalizes_by_ten() -> None:
    """A (10, 100] report is the FIT percent-of-scale encoding and divides by 10 (MAP-R3).

    Garmin writes the 1..10 self-evaluation as 10..100; CR-10 cannot exceed 10, so the
    two encodings never overlap above 10 and the normalization is a deterministic unit
    conversion, not a guess.
    """
    assert rpe_value(70) == 7.0
    assert rpe_value(100) == 10.0
    assert rpe_value(10.5) == 1.05


def test_rpe_out_of_range_or_malformed_is_typed_gap() -> None:
    """An out-of-range or non-numeric report is None (MAP-R5), never clamped."""
    assert rpe_value(-1) is None
    assert rpe_value(101) is None
    assert rpe_value(math.nan) is None
    assert rpe_value(math.inf) is None
    assert rpe_value("7") is None
    assert rpe_value(True) is None
    assert rpe_value(None) is None


def test_feel_ordinal_validates_one_to_five() -> None:
    """The feel report is the 1..5 ordinal carried verbatim; anything else is None (MAP-R5)."""
    assert feel_value(1) == 1
    assert feel_value(5) == 5
    assert feel_value(3.0) == 3  # integral float is the same ordinal
    assert feel_value(0) is None
    assert feel_value(6) is None
    assert feel_value(2.5) is None  # non-integral is malformed, not rounded
    assert feel_value(math.nan) is None
    assert feel_value("3") is None
    assert feel_value(None) is None


def test_file_upload_payload_carries_reported_exertion() -> None:
    """The file-upload payload maps FIT session perceived_exertion/feel to canonical keys.

    The FIT decoder passes the session message through verbatim, so a device that wrote
    the percent-scale self-evaluation surfaces here normalized to CR-10 (SRPE-R1).
    """
    asbo = ActivityAsbo(
        session={
            "sport": "training",
            "start_time": _START,
            "perceived_exertion": 70,
            "feel": 2,
        }
    )
    payload = activity_payload(asbo, _START, build_streams(asbo), [])
    assert payload["sport"] == "strength"
    assert payload["perceived_exertion"] == 7.0
    assert payload["feel"] == 2


def test_file_upload_payload_unreported_exertion_is_typed_absence() -> None:
    """A session with no exertion report yields None for both keys — never a default (MAP-R5)."""
    asbo = ActivityAsbo(session={"sport": "cycling", "start_time": _START})
    payload = activity_payload(asbo, _START, build_streams(asbo), [])
    assert payload["perceived_exertion"] is None
    assert payload["feel"] is None


def test_intervals_payload_carries_icu_rpe_and_feel() -> None:
    """The intervals.icu payload maps icu_rpe (already CR-10) and feel to canonical keys."""
    act = IntervalsActivityAsbo(id="i1", type="Ride", icu_rpe=8, feel=2)
    payload = _activity_payload(act, _START, {})
    assert payload["perceived_exertion"] == 8.0
    assert payload["feel"] == 2


def test_intervals_payload_malformed_reports_are_typed_gaps() -> None:
    """An absent or out-of-range intervals.icu report is None in the payload (MAP-R5)."""
    act = IntervalsActivityAsbo(id="i2", type="Ride", icu_rpe=-3, feel=9)
    payload = _activity_payload(act, _START, {})
    assert payload["perceived_exertion"] is None
    assert payload["feel"] is None
    bare = IntervalsActivityAsbo(id="i3", type="Ride")
    bare_payload = _activity_payload(bare, _START, {})
    assert bare_payload["perceived_exertion"] is None
    assert bare_payload["feel"] is None
