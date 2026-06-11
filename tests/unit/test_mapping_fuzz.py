"""Fuzz tests for the ASBO→GBO MAPPING layer (TIER-R5 T-FUZZ — adapter map surface).

The decoder fuzz (``test_file_upload_fuzz``) covers bytes→ASBO; this module covers the
OTHER mandated T-FUZZ surface: the pure ``map`` of every shipped adapter driven with
arbitrary ASBO content. The contract (TIER-R5/CON-R3): the mapping MUST NOT crash,
hang, or raise an unhandled exception on ANY input, and MUST NOT silently emit a
wrong-but-plausible canonical record — the only permitted outcomes are well-formed
:class:`GboCandidate` records or an EMPTY result (a required field absent yields no
fabricated candidate, MAP-R5/ING-R3).

Bounded + deterministic (PR-gate mode (a)); the nightly campaign re-runs this same
corpus under the extended ``fuzz-nightly`` hypothesis profile (``just
test-fuzz-nightly``, CI-R4).
"""

from __future__ import annotations

import datetime as _dt
import math

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import SourceKind
from wattwise_core.ingestion.adapters._asbo import ActivityAsbo, AsboLap, AsboRecord
from wattwise_core.ingestion.adapters._intervals_asbo import (
    ActivityWithStreams,
    IntervalsActivityAsbo,
    IntervalsStreamAsbo,
    IntervalsWellnessAsbo,
)
from wattwise_core.ingestion.adapters.file_upload import FileUploadAdapter
from wattwise_core.ingestion.adapters.intervals_icu import IntervalsIcuAdapter
from wattwise_core.ingestion.base import FetchContext, SourceDescriptorRef

pytestmark = pytest.mark.fuzz

_DESCRIPTOR = SourceDescriptorRef(
    source_descriptor_id="00000000-0000-0000-0000-000000000001",
    source_key="fuzz_source",
    kind=SourceKind.FILE_UPLOAD,
)
_CONTEXT = FetchContext(
    ingest_run_id="00000000-0000-0000-0000-000000000002",
    fetched_at=_dt.datetime(2026, 6, 1, tzinfo=_dt.UTC),
)

_FUZZ_SETTINGS = settings(
    max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)

# Hostile scalar soup: junk types, NaN/Inf, huge magnitudes, empty/whitespace strings.
_scalar = st.one_of(
    st.none(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.integers(min_value=-(2**63), max_value=2**63),
    st.text(max_size=20),
    st.booleans(),
)
_maybe_dt = st.one_of(
    st.none(),
    st.datetimes(
        min_value=_dt.datetime(1970, 1, 1),
        max_value=_dt.datetime(2100, 1, 1),
        timezones=st.sampled_from([_dt.UTC, None]),  # type: ignore[list-item]
    ),
)
_maybe_float = st.one_of(st.none(), st.floats(allow_nan=True, allow_infinity=True))


def _assert_candidates(result: object) -> None:
    """Every mapping outcome is a list of WELL-FORMED candidates (or empty) — never junk."""
    assert isinstance(result, list)
    for cand in result:
        assert isinstance(cand, GboCandidate)
        assert cand.gbo_type in {"activity", "daily_wellness"}
        assert isinstance(cand.payload, dict)
        start = cand.payload.get("start_time")
        if cand.gbo_type == "activity":
            # A mapped activity NEVER carries a fabricated/absent start instant.
            assert isinstance(start, _dt.datetime)


_records = st.lists(
    st.builds(
        AsboRecord,
        timestamp=_maybe_dt,
        power_w=_maybe_float,
        hr_bpm=_maybe_float,
        cadence_rpm=_maybe_float,
        speed_mps=_maybe_float,
        altitude_m=_maybe_float,
        distance_m=_maybe_float,
        temp_c=_maybe_float,
    ),
    max_size=30,
).map(tuple)
_laps = st.lists(
    st.builds(
        AsboLap,
        lap_index=st.integers(min_value=-5, max_value=500),
        start_time=_maybe_dt,
        duration_s=_maybe_float,
        avg_power_w=_maybe_float,
    ),
    max_size=10,
).map(tuple)


@_FUZZ_SETTINGS
@given(
    records=_records,
    laps=_laps,
    session=st.dictionaries(st.text(max_size=15), _scalar, max_size=10),
    rr=st.lists(st.floats(allow_nan=True, allow_infinity=True), max_size=20).map(tuple),
)
# Distilled fuzz failure pinned per TIER-R5/TIER-R1: an AWARE lap followed by a NAIVE
# lap start (and a naive-only fallback) used to raise TypeError from naive-aware
# datetime subtraction in build_laps/start_time.
@example(
    records=(),
    laps=(
        AsboLap(lap_index=0, start_time=_dt.datetime(2000, 1, 1, tzinfo=_dt.UTC)),
        AsboLap(lap_index=1, start_time=_dt.datetime(2000, 1, 1)),
    ),
    session={},
    rr=(),
)
@example(
    records=(AsboRecord(timestamp=_dt.datetime(2000, 1, 1)),),
    laps=(AsboLap(lap_index=0, start_time=_dt.datetime(2000, 1, 1, tzinfo=_dt.UTC)),),
    session={},
    rr=(),
)
def test_file_upload_map_never_crashes_or_fabricates(
    records: tuple[AsboRecord, ...],
    laps: tuple[AsboLap, ...],
    session: dict[str, object],
    rr: tuple[float, ...],
) -> None:
    """Arbitrary decoded ASBO content maps to well-formed candidates or nothing —
    no unhandled exception, no fabricated start_time (TIER-R5 mapping mode (a))."""
    asbo = ActivityAsbo(records=records, session=session, laps=laps, rr_intervals_ms=rr)
    result = FileUploadAdapter().map(asbo, _DESCRIPTOR, _CONTEXT)
    _assert_candidates(result)


@_FUZZ_SETTINGS
@given(
    start_date=st.one_of(st.none(), st.text(max_size=30)),
    body=st.dictionaries(
        st.sampled_from(
            [
                "type",
                "moving_time",
                "elapsed_time",
                "distance",
                "icu_average_watts",
                "average_heartrate",
                "average_speed",
                "average_temp",
            ]
        ),
        _scalar,
        max_size=8,
    ),
    stream_data=st.lists(_scalar, max_size=25),
)
def test_intervals_activity_map_never_crashes_or_fabricates(
    start_date: str | None, body: dict[str, object], stream_data: list[object]
) -> None:
    """Arbitrary Intervals activity payloads (junk fields, hostile stream values, a
    malformed start_date) map fail-closed: typed candidates or nothing, never a crash."""
    try:
        activity = IntervalsActivityAsbo.model_validate(
            {"id": "fuzz-1", "start_date": start_date, **body}
        )
    except ValueError:
        return  # the typed boundary rejected the payload — the fail-closed outcome
    streams = [IntervalsStreamAsbo(type="watts", data=stream_data)]
    asbo = ActivityWithStreams(activity=activity, streams=streams)
    result = IntervalsIcuAdapter().map(asbo, _DESCRIPTOR, _CONTEXT)
    _assert_candidates(result)


@_FUZZ_SETTINGS
@given(
    record_id=st.text(max_size=24),
    body=st.dictionaries(
        st.sampled_from(
            ["restingHR", "hrv", "hrvSDNN", "sleepScore", "sleepSecs", "steps", "weight"]
        ),
        st.one_of(st.none(), st.floats(allow_nan=True, allow_infinity=True), st.integers()),
        max_size=7,
    ),
)
def test_intervals_wellness_map_never_crashes_or_fabricates(
    record_id: str, body: dict[str, object]
) -> None:
    """Arbitrary wellness payloads (junk date ids, NaN/Inf vitals) map fail-closed."""
    try:
        asbo = IntervalsWellnessAsbo.model_validate({"id": record_id, **body})
    except ValueError:
        return
    result = IntervalsIcuAdapter().map(asbo, _DESCRIPTOR, _CONTEXT)
    _assert_candidates(result)
    for cand in result:
        for value in cand.payload.values():
            if isinstance(value, float):
                # No NaN/Inf is promoted into a canonical wellness candidate value.
                assert math.isfinite(value)
