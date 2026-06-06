"""Contract tests for the Intervals.icu ``api_key`` adapter (ADP-R17, TST-R1, CON-R3).

Offline-only: ``respx`` mocks the ``httpx`` transport so the typed client's
fetch/probe/discover paths run with NO live network (TIER-R1). Recorded, sanitized
fixtures (PII stripped) under ``fixtures/intervals/`` are fed through fetch -> map and
asserted on:

* ASBO -> GBO canonical mapping (MAP-R2/R3/R4): canonical field names, SI units,
  canonical sport codes, canonical stream channels.
* Provenance / trust tiers (PRV-R7): a real per-sample stream -> ``raw_stream``; a
  summary-only wellness scalar -> ``summary_only``.
* Free text tagged untrusted (MAP-R7) for prompt-injection quarantine.
* Real gaps preserved as ``None`` never ``0`` (MAP-R5).
* Graceful degrade (CON-R3): a malformed payload fails closed as a typed validation
  error at the client boundary (CLI-R2); a naive/absent ``start_date`` yields no
  fabricated candidate (ING-R3) rather than a crash.
* The mandatory read-only probe (AUT-R17) succeeds before ``connected``.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import httpx
import pydantic
import pytest
import respx

from wattwise_core.domain.enums import Fidelity, SourceKind
from wattwise_core.ingestion.adapters.intervals_icu import (
    ActivityWithStreams,
    IntervalsActivityAsbo,
    IntervalsIcuAdapter,
    IntervalsIcuClient,
    IntervalsStreamAsbo,
    IntervalsWellnessAsbo,
)
from wattwise_core.ingestion.base import FetchContext, SourceDescriptorRef

pytestmark = pytest.mark.contract

_FIXTURES = Path(__file__).parent / "fixtures" / "intervals"
_BASE = "https://intervals.icu"
_ATHLETE = "i00000"
_ACTIVITY = "i111111111"
_FETCHED_AT = _dt.datetime(2026, 6, 6, 12, 0, tzinfo=_dt.UTC)


def _load(name: str) -> Any:
    return json.loads((_FIXTURES / name).read_text())


def _descriptor() -> SourceDescriptorRef:
    return SourceDescriptorRef("sd-uuid-1", "intervals_icu", SourceKind.OAUTH_API)


def _ctx() -> FetchContext:
    return FetchContext(ingest_run_id="run-1", fetched_at=_FETCHED_AT, connection_id="conn-1")


def _client() -> IntervalsIcuClient:
    # A non-secret placeholder key; respx intercepts so nothing leaves the process.
    return IntervalsIcuClient("test-key", _ATHLETE, base_url=_BASE)


# --------------------------------------------------------------------------- probe


@respx.mock
async def test_probe_succeeds_before_connected() -> None:
    """The mandatory read-only probe (AUT-R17) returns the athlete profile on 200."""
    respx.get(f"{_BASE}/api/v1/athlete/{_ATHLETE}").mock(
        return_value=httpx.Response(200, json=_load("athlete_profile.json"))
    )
    async with _client() as client:
        profile = await client.probe()
    assert profile["id"] == _ATHLETE


@respx.mock
async def test_probe_raises_on_unauthorized() -> None:
    """A bad key (401) MUST raise so the caller never reports ``connected`` (AUT-R17)."""
    respx.get(f"{_BASE}/api/v1/athlete/{_ATHLETE}").mock(return_value=httpx.Response(401))
    async with _client() as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.probe()


# ------------------------------------------------------------------- discover/fetch


@respx.mock
async def test_discover_lists_activities() -> None:
    """discover_activities pages by ISO date window (ADP-R5)."""
    respx.get(f"{_BASE}/api/v1/athlete/{_ATHLETE}/activities").mock(
        return_value=httpx.Response(200, json=_load("activities_list.json"))
    )
    async with _client() as client:
        rows = await client.discover_activities("2024-01-01", "2026-12-31")
    assert len(rows) == 2
    assert rows[0]["id"] == _ACTIVITY


@respx.mock
async def test_fetch_activity_then_map_recorded() -> None:
    """Recorded walk activity (HR-only) maps to a canonical activity candidate."""
    respx.get(f"{_BASE}/api/v1/activity/{_ACTIVITY}").mock(
        return_value=httpx.Response(200, json=_load("activity_detail.json"))
    )
    respx.get(f"{_BASE}/api/v1/activity/{_ACTIVITY}/streams").mock(
        return_value=httpx.Response(200, json=_load("activity_streams.json"))
    )
    async with _client() as client:
        asbo = await client.fetch_activity(_ACTIVITY)
    candidates = IntervalsIcuAdapter().map(asbo, _descriptor(), _ctx())

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.gbo_type == "activity"
    assert cand.source_native_id == _ACTIVITY
    assert cand.source_descriptor_id == "sd-uuid-1"
    assert cand.connection_id == "conn-1"
    assert cand.fetched_at == _FETCHED_AT
    # Canonical SI fields only — no source-named keys (MAP-R2/R3).
    assert cand.payload["sport"] == "other"  # source "Walk" -> "other"
    assert cand.payload["start_time"] == _dt.datetime(2026, 5, 31, 17, 29, 35, tzinfo=_dt.UTC)
    assert cand.payload["avg_hr_bpm"] == 92
    assert cand.payload["moving_time_s"] == 1960
    assert set(cand.payload) >= {"start_time", "sport", "streams", "device_class"}
    assert not any(k.startswith("icu_") or k.startswith("source") for k in cand.payload)
    # HR stream present + canonical channel name; a real stream -> raw_stream fidelity.
    assert "hr_bpm" in cand.payload["streams"]
    assert cand.payload["streams"]["hr_bpm"]["sample_basis"] == "time"
    assert cand.trust_tier is Fidelity.RAW_STREAM
    # The activity carried a (redacted) free-text name -> untrusted (MAP-R7).
    assert cand.untrusted_content is True


@respx.mock
async def test_fetch_handles_missing_streams_endpoint() -> None:
    """A non-200 streams response degrades to a summary-only activity, not a crash."""
    respx.get(f"{_BASE}/api/v1/activity/{_ACTIVITY}").mock(
        return_value=httpx.Response(200, json=_load("activity_detail.json"))
    )
    respx.get(f"{_BASE}/api/v1/activity/{_ACTIVITY}/streams").mock(
        return_value=httpx.Response(404)
    )
    async with _client() as client:
        asbo = await client.fetch_activity(_ACTIVITY)
    cand = IntervalsIcuAdapter().map(asbo, _descriptor(), _ctx())[0]
    assert cand.payload["streams"] == {}
    # No real stream -> platform-computed summary fidelity, not raw_stream (PRV-R7).
    assert cand.trust_tier is Fidelity.PLATFORM_COMPUTED


# --------------------------------------------------------- full-channel mapping


def test_map_synthetic_cycling_all_channels() -> None:
    """A power-bearing cycling activity maps every canonical channel + SI scalars."""
    act = IntervalsActivityAsbo.model_validate(_load("synthetic_cycling_activity.json"))
    streams = [
        IntervalsStreamAsbo.model_validate(s)
        for s in _load("synthetic_cycling_streams.json")
    ]
    cand = IntervalsIcuAdapter().map(
        ActivityWithStreams(activity=act, streams=streams), _descriptor(), _ctx()
    )[0]

    assert cand.payload["sport"] == "cycling"
    assert cand.payload["device_class"] == "trainer"
    assert cand.payload["energy_kj"] == 900.0  # icu_joules / 1000 (SI conversion)
    assert cand.payload["total_work_j"] == 900000.0
    assert cand.payload["has_power"] is True
    assert cand.payload["has_gps"] is True
    assert cand.payload["has_cadence"] is True
    # Every source channel maps to its canonical name; unmappable one is dropped.
    assert sorted(cand.payload["streams"]) == [
        "altitude_m", "cadence_rpm", "distance_m", "hr_bpm",
        "latlng", "power_w", "speed_mps", "temp_c",
    ]
    assert "unmappable_channel" not in cand.payload["streams"]
    assert cand.trust_tier is Fidelity.RAW_STREAM


def test_map_preserves_real_gaps_as_none_never_zero() -> None:
    """A mid-stream missing sample stays ``None`` — never coerced to 0 (MAP-R5)."""
    act = IntervalsActivityAsbo.model_validate(_load("synthetic_cycling_activity.json"))
    streams = [
        IntervalsStreamAsbo.model_validate(s)
        for s in _load("synthetic_cycling_streams.json")
    ]
    cand = IntervalsIcuAdapter().map(
        ActivityWithStreams(activity=act, streams=streams), _descriptor(), _ctx()
    )[0]
    power = cand.payload["streams"]["power_w"]["values"]
    cadence = cand.payload["streams"]["cadence_rpm"]["values"]
    assert power == [240, 255, None, 260, 250, 245]
    assert None in power and 0 not in power
    assert cadence[3] is None


def test_map_is_pure_and_deterministic() -> None:
    """Same ASBO + context -> byte-identical candidates incl. content_hash (MAP-R1/R8)."""
    act = IntervalsActivityAsbo.model_validate(_load("synthetic_cycling_activity.json"))
    streams = [
        IntervalsStreamAsbo.model_validate(s)
        for s in _load("synthetic_cycling_streams.json")
    ]
    awith = ActivityWithStreams(activity=act, streams=streams)
    first = IntervalsIcuAdapter().map(awith, _descriptor(), _ctx())[0]
    second = IntervalsIcuAdapter().map(awith, _descriptor(), _ctx())[0]
    assert first.content_hash == second.content_hash
    assert len(first.content_hash) == 64  # sha256 hex
    assert first.payload == second.payload


# --------------------------------------------------------------------- wellness


@respx.mock
async def test_fetch_wellness_then_map() -> None:
    """Wellness rows map to daily_wellness candidates (rmssd, summary_only fidelity)."""
    respx.get(f"{_BASE}/api/v1/athlete/{_ATHLETE}/wellness").mock(
        return_value=httpx.Response(200, json=_load("wellness.json"))
    )
    async with _client() as client:
        rows = await client.fetch_wellness("2026-05-01", "2026-06-06")
    cands = [c for row in rows for c in IntervalsIcuAdapter().map(row, _descriptor(), _ctx())]

    assert len(cands) == len(rows) >= 1
    first = cands[0]
    assert first.gbo_type == "daily_wellness"
    assert first.payload["local_date"] == _dt.date(2026, 5, 1)
    assert first.payload["hrv_rmssd_ms"] == _load("wellness.json")[0]["hrv"]
    assert first.payload["resting_hr_bpm"] == 48
    assert first.trust_tier is Fidelity.SUMMARY_ONLY
    assert first.untrusted_content is False
    # observed_at is derived deterministically from the date (no clock read).
    assert first.observed_at == _dt.datetime(2026, 5, 1, tzinfo=_dt.UTC)


def test_map_wellness_with_invalid_date_yields_no_candidate() -> None:
    """A wellness id that is not an ISO date yields no candidate, not a crash (ING-R3)."""
    bad = IntervalsWellnessAsbo.model_validate({"id": "not-a-date", "restingHR": 50})
    assert IntervalsIcuAdapter().map(bad, _descriptor(), _ctx()) == []


# --------------------------------------------------------------- graceful degrade


def test_malformed_payload_fails_closed_at_boundary() -> None:
    """A payload missing the required ``id`` fails closed as a typed error (CLI-R2)."""
    with pytest.raises(pydantic.ValidationError):
        IntervalsActivityAsbo.model_validate(_load("malformed_activity.json"))


def test_map_absent_start_time_yields_no_fabricated_candidate() -> None:
    """No usable start_time -> no candidate (fail-closed, never a defaulted time)."""
    act = IntervalsActivityAsbo.model_validate(
        {"id": "i999", "type": "Ride", "start_date": None, "start_date_local": None}
    )
    assert IntervalsIcuAdapter().map(act, _descriptor(), _ctx()) == []


def test_map_naive_start_time_is_rejected() -> None:
    """A naive (tz-less) instant is rejected, not silently stored as UTC (IDS-R3)."""
    act = IntervalsActivityAsbo.model_validate(
        {"id": "i998", "type": "Ride", "start_date": "2026-06-01T06:00:00"}
    )
    # Only start_date is naive; with no usable tz-aware start_time -> no candidate.
    assert IntervalsIcuAdapter().map(act, _descriptor(), _ctx()) == []


def test_unknown_object_type_maps_to_empty() -> None:
    """An ASBO the adapter does not recognize maps to no candidates (defensive)."""
    assert IntervalsIcuAdapter().map(object(), _descriptor(), _ctx()) == []


# ------------------------------------------------------------ adapter identity


def test_adapter_satisfies_source_adapter_protocol() -> None:
    """The class declares the required identity attrs + a pure map (SourceAdapter)."""
    adapter = IntervalsIcuAdapter()
    assert adapter.source_key == "intervals_icu"
    assert adapter.auth_archetype.value == "api_key"
    assert adapter.adapter_version and adapter.mapping_version
    assert callable(adapter.map)
