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
    RpeEncoding,
    activity_payload,
    build_streams,
    feel_value,
    rpe_value,
)
from wattwise_core.ingestion.adapters.intervals_icu import _activity_payload

pytestmark = pytest.mark.unit

_START = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=_dt.UTC)


def test_rpe_cr10_encoding_passes_through_unscaled() -> None:
    """A CR-10-tagged report ([0, 10]) is carried verbatim (MAP-R3); >10 is a typed gap."""
    assert rpe_value(7, RpeEncoding.CR10) == 7.0
    assert rpe_value(7.5, RpeEncoding.CR10) == 7.5
    assert rpe_value(0, RpeEncoding.CR10) == 0.0
    # Boundary: under CR-10 a 10 IS the legitimate maximum effort — passed through.
    assert rpe_value(10, RpeEncoding.CR10) == 10.0
    # Default encoding is CR-10 (intervals.icu / manual), so the same holds bare.
    assert rpe_value(8) == 8.0
    # A CR-10 source never sends a percent value; >10 cannot be CR-10 → typed gap.
    assert rpe_value(70, RpeEncoding.CR10) is None
    assert rpe_value(11, RpeEncoding.CR10) is None


def test_rpe_percent_encoding_normalizes_by_ten() -> None:
    """A [10, 100] percent-of-scale report (Garmin FIT) divides by 10 (MAP-R3; SRPE-R2)."""
    assert rpe_value(70, RpeEncoding.PERCENT) == 7.0
    assert rpe_value(100, RpeEncoding.PERCENT) == 10.0
    assert rpe_value(10.5, RpeEncoding.PERCENT) == 1.05


def test_rpe_percent_boundary_ten_is_min_effort_not_max() -> None:
    """SRPE-R2 boundary: a percent-source 10 is CR-10 1.0 (minimum), NEVER 10.0 (maximum).

    The old shared sink read raw 10 as CR-10 maximum effort, a 10x encoding error for the
    Garmin FIT source. With the encoding source-tagged, a percent-scale 10 decodes to 1.0.
    """
    assert rpe_value(10, RpeEncoding.PERCENT) == 1.0


def test_rpe_percent_ambiguous_band_fails_closed() -> None:
    """SRPE-R2: a percent-source value in the ambiguous (0, 10) band is a typed gap (MAP-R5).

    Below 10 a percent-scale reading cannot be told apart from a native CR-10 reading, so it
    fails closed rather than be guessed — a misread would fabricate a max-effort session from
    a minimum-effort one. Only an exact 0 (unambiguous rest) and [10, 100] decode.
    """
    assert rpe_value(0, RpeEncoding.PERCENT) == 0.0  # unambiguous rest report
    for ambiguous in (1, 5, 7, 9, 9.9):
        assert rpe_value(ambiguous, RpeEncoding.PERCENT) is None


def test_rpe_out_of_range_or_malformed_is_typed_gap() -> None:
    """An out-of-range or non-numeric report is None (MAP-R5), never clamped — both encodings."""
    for enc in (RpeEncoding.CR10, RpeEncoding.PERCENT):
        assert rpe_value(-1, enc) is None
        assert rpe_value(math.nan, enc) is None
        assert rpe_value(math.inf, enc) is None
        assert rpe_value("7", enc) is None
        assert rpe_value(True, enc) is None
        assert rpe_value(None, enc) is None
    assert rpe_value(101, RpeEncoding.PERCENT) is None
    assert rpe_value(11, RpeEncoding.CR10) is None


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


def test_file_upload_payload_carries_workout_rpe_percent_scale() -> None:
    """The file-upload payload decodes the FIT ``workout_rpe`` session field to CR-10 (SRPE-R2).

    Garmin FIT records perceived exertion as ``workout_rpe`` (uint8, percent-of-scale =
    RPE x 10); the decoder passes the session message through verbatim, so a device that
    wrote 70 surfaces here normalized to CR-10 7.0.
    """
    asbo = ActivityAsbo(
        session={
            "sport": "training",
            "start_time": _START,
            "workout_rpe": 70,
            "feel": 2,
        }
    )
    payload = activity_payload(asbo, _START, build_streams(asbo), [])
    assert payload["sport"] == "strength"
    assert payload["perceived_exertion"] == 7.0
    assert payload["feel"] == 2


def test_file_upload_payload_workout_rpe_boundary_ten_is_min_effort() -> None:
    """SRPE-R2 contract-level boundary: FIT ``workout_rpe`` 10 surfaces as CR-10 1.0, not 10.0.

    A device reporting minimum effort (percent value 10) must NOT be priced as maximum
    effort — the regression the source-tagged decode fixes, asserted through the full payload.
    """
    asbo = ActivityAsbo(session={"sport": "training", "start_time": _START, "workout_rpe": 10})
    payload = activity_payload(asbo, _START, build_streams(asbo), [])
    assert payload["perceived_exertion"] == 1.0


def test_file_upload_payload_falls_back_to_cr10_perceived_exertion() -> None:
    """A non-Garmin uploader's pre-normalized CR-10 ``perceived_exertion`` is read as CR-10."""
    asbo = ActivityAsbo(
        session={"sport": "training", "start_time": _START, "perceived_exertion": 7}
    )
    payload = activity_payload(asbo, _START, build_streams(asbo), [])
    assert payload["perceived_exertion"] == 7.0


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
