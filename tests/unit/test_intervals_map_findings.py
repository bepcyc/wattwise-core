"""Regression tests pinning the Intervals.icu adapter mapping fixes (one per finding).

* ADP-R10 / TIER-R5 — a malformed/dict/list stream element becomes a typed gap
  (``None``), never an un-typed blob in a canonical SI channel.
* MAP-R4 / MAP-R2  — ``sub_type`` is mapped through a vocab to a registry code (or
  ``None``), never passed through raw.
* MAP-R12          — ``device_class`` is not fabricated from speed/distance alone.
* MAP-R3           — the ``distance`` stream is TIME-sampled (matches file-upload).
"""

from __future__ import annotations

import datetime as _dt

from wattwise_core.domain.enums import DeviceClass, SourceKind
from wattwise_core.ingestion.adapters import _intervals_map as _im
from wattwise_core.ingestion.adapters.intervals_icu import (
    ActivityWithStreams,
    IntervalsActivityAsbo,
    IntervalsIcuAdapter,
    IntervalsStreamAsbo,
)
from wattwise_core.ingestion.base import FetchContext, SourceDescriptorRef

UTC = _dt.UTC


def _descriptor() -> SourceDescriptorRef:
    return SourceDescriptorRef("sd-1", "intervals_icu", SourceKind.OAUTH_API)


def _ctx() -> FetchContext:
    return FetchContext(
        ingest_run_id="run-1",
        fetched_at=_dt.datetime(2026, 6, 6, 12, 0, tzinfo=UTC),
        connection_id="conn-1",
    )


def _activity(**over: object) -> IntervalsActivityAsbo:
    base: dict[str, object] = {"id": "i1", "type": "Ride", "start_date": "2026-06-01T08:00:00Z"}
    base.update(over)
    return IntervalsActivityAsbo.model_validate(base)


# --------------------------------------------------------- ADP-R10 / TIER-R5: streams


def test_malformed_stream_elements_become_gaps_never_blobs() -> None:
    """A dict/list/str element in a source channel coerces to None (ADP-R10/TIER-R5)."""
    streams = [IntervalsStreamAsbo(type="watts", data=[{"evil": 1}, "str", None, [1, 2], 240])]
    cand = IntervalsIcuAdapter().map(
        ActivityWithStreams(activity=_activity(), streams=streams), _descriptor(), _ctx()
    )[0]
    values = cand.payload["streams"]["power_w"]["values"]
    # Every non-numeric blob is a typed gap; the one numeric sample is a float.
    assert values == [None, None, None, None, 240.0]
    assert not any(isinstance(v, dict | list | str) for v in values)


def test_latlng_pairs_survive_but_bad_pairs_become_gaps() -> None:
    """A clean lat/lon pair survives as floats; a malformed pair becomes None (ADP-R10)."""
    streams = [IntervalsStreamAsbo(type="latlng", data=[[51.5, -0.1], {"x": 1}, [1.0], None])]
    cand = IntervalsIcuAdapter().map(
        ActivityWithStreams(activity=_activity(), streams=streams), _descriptor(), _ctx()
    )[0]
    assert cand.payload["streams"]["latlng"]["values"] == [[51.5, -0.1], None, None, None]


# ---------------------------------------------------------------- MAP-R4 / MAP-R2: sub_sport


def test_sub_type_mapped_to_registry_code_not_passed_through() -> None:
    """A known sub_type maps to a seeded registry code; an unknown one -> None (MAP-R4/R2)."""
    mapped = IntervalsIcuAdapter().map(_activity(sub_type="Gravel"), _descriptor(), _ctx())[0]
    assert mapped.payload["sub_sport"] == "cycling_other"  # mapped, not raw "Gravel"
    unknown = IntervalsIcuAdapter().map(
        _activity(sub_type="SomethingExotic"), _descriptor(), _ctx()
    )[0]
    assert unknown.payload["sub_sport"] is None  # unmapped -> typed gap, never a raw token


def test_sub_sport_code_helper_never_echoes_raw_token() -> None:
    """The vocab helper returns a registry code or None, never the raw source token."""
    assert _im.sub_sport_code("Gravel") == "cycling_other"
    assert _im.sub_sport_code("not-a-known-subtype") is None
    assert _im.sub_sport_code("generic") is None
    assert _im.sub_sport_code(None) is None


# ----------------------------------------------------------------------- MAP-R12: device


def test_device_class_not_fabricated_from_speed_or_distance() -> None:
    """Speed/distance alone do NOT imply a GPS watch -> device_class stays UNKNOWN (MAP-R12)."""
    cand = IntervalsIcuAdapter().map(
        _activity(average_speed=5.0, distance=12000.0), _descriptor(), _ctx()
    )[0]
    assert cand.payload["device_class"] == DeviceClass.UNKNOWN.value
    # A real provenance signal still assigns a concrete class.
    powered = IntervalsIcuAdapter().map(_activity(power_meter=True), _descriptor(), _ctx())[0]
    assert powered.payload["device_class"] == DeviceClass.POWERMETER.value


# ------------------------------------------------------------------------ MAP-R3: basis


def test_distance_stream_is_time_sampled() -> None:
    """The distance channel is TIME-sampled, consistent with the file-upload adapter (MAP-R3)."""
    streams = [IntervalsStreamAsbo(type="distance", data=[0.0, 5.0, 10.0])]
    cand = IntervalsIcuAdapter().map(
        ActivityWithStreams(activity=_activity(), streams=streams), _descriptor(), _ctx()
    )[0]
    assert cand.payload["streams"]["distance_m"]["sample_basis"] == "time"
