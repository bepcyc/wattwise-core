"""Regression tests pinning the file-upload mapping fixes (MAP-R4/R2).

The file-upload adapter must translate a source ``sub_sport`` token through a vocab to
a canonical registry code (or ``None``), never echo the raw lowercased source token
into the canonical FK field (MAP-R4/MAP-R2) — mirroring the Intervals adapter.
"""

from __future__ import annotations

import datetime as _dt
import json
import math

from wattwise_core.ingestion.adapters._asbo import ActivityAsbo, AsboLap, AsboRecord
from wattwise_core.ingestion.adapters._map_activity import (
    _sub_sport,
    activity_payload,
    build_laps,
    build_streams,
)


def test_sub_sport_maps_known_token_to_registry_code() -> None:
    """A known source sub_sport token maps to a seeded registry code (MAP-R4)."""
    assert _sub_sport("gravel") == "cycling_other"
    assert _sub_sport("Trail") == "running_other"
    assert _sub_sport("open_water") == "swimming_other"


def test_sub_sport_unknown_token_is_typed_gap_not_passthrough() -> None:
    """An unmapped/absent token yields None, never the raw lowercased token (MAP-R2)."""
    assert _sub_sport("road_bike_with_unknown_suffix") is None
    assert _sub_sport("generic") is None
    assert _sub_sport("all") is None
    assert _sub_sport("") is None
    assert _sub_sport(None) is None
    assert _sub_sport(42) is None  # type: ignore[arg-type]


def test_non_finite_session_and_stream_values_become_typed_gaps() -> None:
    """A NaN/inf from ANY adapter is dropped to a typed gap at the canonical mapper (MAP-R5).

    ``_map_activity`` is the one sink every adapter's scalars and streams flow through (a raw
    FIT field, an Intervals.icu value, a malformed XML token), so a non-finite value must
    become ``None`` here — keeping the payload strict JSON (Postgres JSONB rejects
    NaN/Infinity) and deterministic (``nan != nan`` would break the re-decode, GBO-AC-1).
    """
    start = _dt.datetime(2024, 1, 2, 10, 0, tzinfo=_dt.UTC)
    asbo = ActivityAsbo(
        records=(
            AsboRecord(timestamp=start, power_w=math.nan, hr_bpm=140.0, latlng=(math.nan, 7.0)),
            AsboRecord(timestamp=start, power_w=210.0, hr_bpm=142.0, latlng=(45.0, 7.0)),
        ),
        session={
            "sport": "cycling",
            "start_time": start,
            "avg_power": math.nan,
            "total_work": math.inf,
        },
    )
    streams = build_streams(asbo)
    payload = activity_payload(asbo, start, streams, [])
    assert payload["avg_power_w"] is None  # non-finite session scalar -> typed gap
    assert payload["total_work_j"] is None
    assert payload["energy_kj"] is None
    assert streams["power_w"]["values"] == [None, 210.0]  # non-finite sample dropped, finite kept
    assert streams["latlng"]["values"][0] is None  # (nan, 7.0) dropped whole
    assert streams["latlng"]["values"][1] == [45.0, 7.0]
    rendered = json.dumps(payload, default=str)  # strict JSON: no NaN/Infinity tokens
    json.loads(rendered)
    assert "NaN" not in rendered and "Infinity" not in rendered


def test_non_finite_lap_aggregates_become_typed_gaps_without_crashing() -> None:
    """A non-finite LAP aggregate is dropped to a typed gap at the mapper (MAP-R5).

    Lap scalars bypass ``build_streams``/``activity_payload``, so the canonical sink must also
    cover ``build_laps``: the FIT decoder's lenient float parse can pass a NaN/inf lap field
    (e.g. corrupt-FIT recovery) straight into an ``AsboLap``. It must become ``None`` here — a
    leaked NaN/inf makes the payload invalid JSONB and non-deterministic, and ``int(inf/nan)``
    for ``duration_s`` would otherwise raise an uncaught OverflowError inside the pure map.
    """
    start = _dt.datetime(2024, 1, 2, 10, 0, tzinfo=_dt.UTC)
    asbo = ActivityAsbo(
        records=(AsboRecord(timestamp=start, power_w=200.0),),
        session={"sport": "cycling", "start_time": start},
        laps=(
            AsboLap(
                lap_index=0,
                start_time=start,
                duration_s=math.inf,  # would crash int(inf) without the finite sink
                distance_m=math.inf,
                avg_power_w=math.nan,
                max_power_w=math.inf,
                avg_hr_bpm=math.nan,
                max_hr_bpm=200.0,  # a finite sibling is preserved
                avg_cadence_rpm=math.inf,
            ),
        ),
    )
    laps = build_laps(asbo, start)
    lap = laps[0]
    assert lap["duration_s"] is None  # int(inf) did not crash; dropped to a typed gap
    assert lap["distance_m"] is None
    assert lap["avg_power_w"] is None
    assert lap["max_power_w"] is None
    assert lap["avg_hr_bpm"] is None
    assert lap["avg_cadence_rpm"] is None
    assert lap["max_hr_bpm"] == 200.0  # the finite value survives
    payload = activity_payload(asbo, start, build_streams(asbo), laps)
    rendered = json.dumps(payload, default=str)
    json.loads(rendered)  # strict JSON: lap NaN/inf must not leak into the payload
    assert "NaN" not in rendered and "Infinity" not in rendered
