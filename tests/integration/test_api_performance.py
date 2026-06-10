"""Integration tests for the performance + activities routers (API-R30/R31/R48/R49/R50).

Builds a minimal ASGI app that mounts the two routers and overrides their dependency
seams (server-derived identity AUTH-R3, ``read`` scope AUTH-R11, the request-scoped
:class:`AnalyticsService` + session) against a seeded in-memory canonical store. Asserts
that:

* every ``/v1/performance/*`` endpoint returns the chart-ready, source-agnostic shape
  (``items`` + X-axis key + per-point ``coverage`` + precomputed ``summary``, API-R31)
  and that an absent input fails closed — a typed ``null`` per point or a
  ``422 analytics-precondition-unmet`` machine code (ERR-R9), never a fabricated ``0``;
* the activities surface lists (cursor-paginated, PAGE-R8), serves canonical detail, the
  column-oriented streams bundle (gaps as explicit ``null``, API-R48), the RDP map track
  (no-GPS → typed empty map not ``404``, API-R49), and the full lap table (API-R50);
* no response point carries a source/provider name (AUTH-R15) and degradation is surfaced,
  never an error (API-R29).

Runs on in-memory SQLite (the portable substrate, GBO-R8b).
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.api.auth import Principal, Scope, authenticate
from wattwise_core.api.errors import install_error_handlers
from wattwise_core.api.ratelimit import RateLimiter
from wattwise_core.api.routers import activities as act_router
from wattwise_core.api.routers import performance as perf_router
from wattwise_core.api.routers import performance_history as perf_history_router
from wattwise_core.domain.enums import (
    SampleBasis,
    SignatureOrigin,
    StreamChannelName,
    StreamSetKind,
)
from wattwise_core.persistence.models import (
    Activity,
    ActivityLap,
    ActivityStreamSet,
    Athlete,
    Base,
    FitnessSignature,
    Sport,
    StreamChannel,
)

UTC = _dt.UTC
_START = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)


@dataclass
class Env:
    """The wired app + its client/session for one seeded scenario."""

    client: AsyncClient
    app: FastAPI
    session: AsyncSession
    athlete_id: str
    activity_id: str


@pytest_asyncio.fixture
async def seeded() -> AsyncIterator[Env]:
    """An app wired to a seeded canonical store, with its client/session exposed."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        athlete_id, activity_id = await _seed_ride(session)
        app = _build_app(session, athlete_id)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t"
        ) as client:
            yield Env(client, app, session, athlete_id, activity_id)
    await engine.dispose()


def _build_app(session: AsyncSession, athlete_id: str) -> FastAPI:
    """Mount both routers and override the identity/scope/service/session/cursor seams.

    Installs the uniform RFC 9457 error handlers so a raised :class:`ProblemError`
    renders as the same ``application/problem+json`` document production emits (ERR-R1),
    and wires the deterministic cursor signing key the activities cursor signs with
    (PAGE-R5).
    """
    app = FastAPI()
    app.state.rate_limiter = RateLimiter()  # per-athlete read bucket the routers debit (LIMIT-R1)
    install_error_handlers(app)
    app.include_router(perf_router.router)
    app.include_router(perf_history_router.router)
    app.include_router(act_router.router)
    # The routers attach the per-subject RateLimit gate, which derives identity from
    # ``authenticate`` (AUTH-R18); bind it to the seeded owner so the read bucket is keyed
    # server-side, mirroring how the assembled app wires it (LIMIT-R1/R6).
    app.dependency_overrides[authenticate] = lambda: Principal(
        subject=athlete_id, scopes=frozenset(Scope)
    )
    app.dependency_overrides[perf_router.require_read_scope] = lambda: None
    app.dependency_overrides[perf_router.current_athlete_id] = lambda: athlete_id
    app.dependency_overrides[perf_router.analytics_service] = lambda: AnalyticsService(session)
    app.dependency_overrides[act_router.current_session] = lambda: session
    app.dependency_overrides[act_router.cursor_signing_key] = lambda: "perf-test-cursor-key"
    return app


async def _seed_ride(session: AsyncSession) -> tuple[str, str]:
    """Seed one athlete + signature + a 1-hour constant-FTP ride with HR, GPS, and a lap."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    athlete = Athlete(sex="male", reference_timezone="UTC")
    session.add(athlete)
    await session.flush()
    aid = athlete.athlete_id
    session.add(
        FitnessSignature(
            athlete_id=aid, signature_type="cycling", effective_date=_dt.date(2026, 1, 1),
            ftp_w=250.0, cp_w=250.0, w_prime_j=20000.0, max_hr_bpm=190.0, resting_hr_bpm=45.0,
            origin=SignatureOrigin.MEASURED,
        )
    )
    activity = Activity(
        athlete_id=aid, start_time=_START, sport="cycling", elapsed_time_s=3600,
        moving_time_s=3600, avg_power_w=250.0, max_power_w=400.0, avg_hr_bpm=150,
        has_power=True, has_hr=True, has_gps=True,
    )
    session.add(activity)
    await session.flush()
    await _seed_streams(session, activity.activity_id)
    session.add(
        ActivityLap(
            activity_id=activity.activity_id, lap_index=0, start_offset_s=0, duration_s=1800,
            distance_m=15000.0, avg_power_w=250.0, max_power_w=400.0, avg_hr_bpm=150,
        )
    )
    await session.commit()
    return str(aid), str(activity.activity_id)


async def _seed_streams(session: AsyncSession, activity_id: uuid.UUID) -> None:
    """Seed power/HR/latlng channels with a deliberate HR gap (a ``None`` sample)."""
    stream_set = ActivityStreamSet(
        activity_id=activity_id, sample_basis=SampleBasis.TIME, sample_rate_hz=1.0,
        sample_count=3600, t0=_START,
    )
    session.add(stream_set)
    await session.flush()
    sid = stream_set.stream_set_id
    hr_vals: list[object] = [150] * 3600
    hr_vals[100] = None  # explicit gap: must survive as null, never 0 (ANL-R7)
    channels = {
        StreamChannelName.POWER_W: [250.0] * 3600,
        StreamChannelName.HR_BPM: hr_vals,
        StreamChannelName.LATLNG: [[52.5 + i / 1e5, 13.4 + i / 1e5] for i in range(3600)],
    }
    for channel, values in channels.items():
        session.add(
            StreamChannel(
                stream_set_id=sid, set_kind=StreamSetKind.ACTIVITY, channel=channel,
                sample_basis=SampleBasis.TIME, values=values, coverage={},
            )
        )


def _range() -> dict[str, str]:
    """The 7-day query range covering the seeded ride."""
    return {"from": "2026-06-01", "to": "2026-06-07"}


def _assert_no_source_name(payload: object) -> None:
    """No response field anywhere may carry a source/provider name (AUTH-R15)."""
    text = repr(payload).lower()
    for banned in ("garmin", "strava", "intervals", "wahoo", "source_descriptor", "provider"):
        assert banned not in text


@pytest.mark.integration
async def test_pmc_chart_ready_shape(seeded: Env) -> None:
    """The PMC endpoint returns one chart point per day with coverage + a precomputed summary."""
    client = seeded.client
    resp = await client.get("/v1/performance/load-fitness", params=_range())
    assert resp.status_code == 200
    body = resp.json()
    assert body["x_axis"] == "local_date"
    assert len(body["items"]) == 7
    first = body["items"][0]
    assert "coverage" in first and "fitness" in first["values"]
    assert first["coverage"]["present"] is True
    assert body["summary"]["ewma_constants"] == {"tau_fitness": 42, "tau_fatigue": 7}
    _assert_no_source_name(body)


@pytest.mark.integration
async def test_coggan_per_activity_points_carry_activity_id(
    seeded: Env
) -> None:
    """Per-activity Coggan points carry ``activity_id`` and a computed TSS (~100)."""
    client, activity_id = seeded.client, seeded.activity_id
    resp = await client.get("/v1/performance/coggan", params=_range())
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    point = items[0]
    assert point["activity_id"] == activity_id
    assert point["values"]["tss"] == pytest.approx(100.0, abs=1.0)


@pytest.mark.integration
async def test_critical_power_fails_closed_when_underdetermined(
    seeded: Env
) -> None:
    """A single constant ride cannot fit CP → 422 cp_insufficient_points, not a number (ERR-R9)."""
    client = seeded.client
    resp = await client.get("/v1/performance/critical-power", params=_range())
    assert resp.status_code == 422
    body = resp.json()
    # ERR-R9: the closed-catalog type + machine errors[].code reach the wire (not a
    # framework-discarded detail dict).
    assert body["type"].endswith("/analytics-precondition-unmet")
    assert body["errors"][0]["code"] == "cp_insufficient_points"


@pytest.mark.integration
async def test_trimp_computed_and_no_source_name(seeded: Env) -> None:
    """TRIMP is computed for an HR ride with thresholds and surfaces a per-day point."""
    client = seeded.client
    resp = await client.get("/v1/performance/trimp", params=_range())
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"][0]["values"]["trimp_points"] is not None
    assert body["summary"]["load_model"] == "hr_load"
    _assert_no_source_name(body)


@pytest.mark.integration
async def test_bad_range_is_422(seeded: Env) -> None:
    """A reversed ``from > to`` range is rejected 422 (PAGE-R8), not silently accepted."""
    client = seeded.client
    resp = await client.get(
        "/v1/performance/load-fitness", params={"from": "2026-06-07", "to": "2026-06-01"}
    )
    assert resp.status_code == 422


@pytest.mark.integration
async def test_activities_list_is_paginated(seeded: Env) -> None:
    """The activity list returns the PAGE-R4 envelope with the seeded ride and no source name."""
    client, activity_id = seeded.client, seeded.activity_id
    resp = await client.get("/v1/activities", params={"sport": "cycling"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["page"]["has_more"] is False
    assert [a["activity_id"] for a in body["data"]] == [activity_id]
    _assert_no_source_name(body)


@pytest.mark.integration
async def test_activity_detail_has_canonical_load_bundle(
    seeded: Env
) -> None:
    """Activity detail composes the canonical scalars + the per-activity load bundle (§13)."""
    client, activity_id = seeded.client, seeded.activity_id
    resp = await client.get(f"/v1/activities/{activity_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["activity_id"] == activity_id
    assert body["max_power_w"] == pytest.approx(400.0)
    assert body["tss"] == pytest.approx(100.0, abs=1.0)
    assert body["load_model"] == "power_tss"


@pytest.mark.integration
async def test_unknown_activity_is_404(seeded: Env) -> None:
    """An id absent for the one athlete is 404 not-found (API-R51), never a fabricated body."""
    client = seeded.client
    resp = await client.get(f"/v1/activities/{uuid.uuid4()}")
    assert resp.status_code == 404


@pytest.mark.integration
async def test_streams_gap_is_explicit_null(seeded: Env) -> None:
    """A stream gap is an explicit ``null`` (never ``0``); channels stay index-aligned (API-R48)."""
    client, activity_id = seeded.client, seeded.activity_id
    resp = await client.get(
        f"/v1/activities/{activity_id}/streams",
        params={"channels": "power_w,hr_bpm", "max_points": 5000},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["base"] == "time"
    power = body["channels"]["power_w"]["values"]
    hr = body["channels"]["hr_bpm"]["values"]
    assert len(power) == len(hr) == len(body["base_values"])
    assert None in hr  # the seeded gap survived as a typed null, never zero-filled
    assert 0 not in hr or hr.count(0) == 0


@pytest.mark.integration
async def test_streams_absent_channel_is_present_with_nulls(
    seeded: Env
) -> None:
    """A requested-but-absent channel is present with ``present=false`` + all-null (API-R48)."""
    client, activity_id = seeded.client, seeded.activity_id
    resp = await client.get(
        f"/v1/activities/{activity_id}/streams", params={"channels": "cadence_rpm"}
    )
    assert resp.status_code == 200
    cadence = resp.json()["channels"]["cadence_rpm"]
    assert cadence["coverage"]["present"] is False
    assert set(cadence["values"]) == {None}


@pytest.mark.integration
async def test_streams_bad_channel_is_422(seeded: Env) -> None:
    """An unknown/excluded channel token (incl. rr_intervals_ms) is rejected 422 (API-R48)."""
    client, activity_id = seeded.client, seeded.activity_id
    resp = await client.get(
        f"/v1/activities/{activity_id}/streams", params={"channels": "rr_intervals_ms"}
    )
    assert resp.status_code == 422


@pytest.mark.integration
async def test_map_track_has_points_and_bounds(seeded: Env) -> None:
    """The GPS map track returns a decimated polyline + a precomputed bounds box (API-R49)."""
    client, activity_id = seeded.client, seeded.activity_id
    resp = await client.get(f"/v1/activities/{activity_id}/map", params={"max_points": 100})
    assert resp.status_code == 200
    body = resp.json()
    assert 0 < len(body["points"]) <= 101
    assert body["bounds"]["min_lat"] < body["bounds"]["max_lat"]
    assert body["coverage"]["present"] is True


@pytest.mark.integration
async def test_map_no_gps_is_typed_empty_not_404(seeded: Env) -> None:
    """A no-GPS activity is a typed empty map (200, points=[], bounds=null), never 404 (API-R49)."""
    no_gps = await _seed_no_gps_activity(seeded.session, seeded.athlete_id)
    resp = await seeded.client.get(f"/v1/activities/{no_gps}/map")
    assert resp.status_code == 200
    body = resp.json()
    assert body["points"] == []
    assert body["bounds"] is None
    assert body["coverage"]["present"] is False


async def _seed_no_gps_activity(session: AsyncSession, athlete_id: str) -> str:
    """Persist a second activity without GPS through the session the app uses."""
    activity = Activity(
        athlete_id=uuid.UUID(athlete_id),
        start_time=_dt.datetime(2026, 6, 2, 8, 0, tzinfo=UTC),
        sport="cycling", elapsed_time_s=600, moving_time_s=600,
        has_power=False, has_gps=False,
    )
    session.add(activity)
    await session.commit()
    return str(activity.activity_id)


@pytest.mark.integration
async def test_laps_are_full_ordered_list(seeded: Env) -> None:
    """Laps are the full ordered table (not paginated); lap scalars are lap-scoped (API-R50)."""
    client, activity_id = seeded.client, seeded.activity_id
    resp = await client.get(f"/v1/activities/{activity_id}/laps")
    assert resp.status_code == 200
    body = resp.json()
    assert body["activity_id"] == activity_id
    assert [lap["lap_index"] for lap in body["laps"]] == [0]
    assert body["laps"][0]["avg_power_w"] == pytest.approx(250.0)


@pytest.mark.integration
async def test_read_scope_required(seeded: Env) -> None:
    """An endpoint whose ``read`` scope is unwired fails closed with 403 (AUTH-R11)."""
    seeded.app.dependency_overrides.pop(perf_router.require_read_scope)
    try:
        resp = await seeded.client.get("/v1/performance/load-fitness", params=_range())
        assert resp.status_code == 403
    finally:
        seeded.app.dependency_overrides[perf_router.require_read_scope] = lambda: None


# --- convergence regressions (PAGE-R5/R6, API-R48 base_values, sort) --------------


@pytest.mark.integration
async def test_tampered_cursor_is_400_invalid_cursor(seeded: Env) -> None:
    """A forged/garbage cursor yields 400 invalid-cursor, not a remapped 422 (PAGE-R5)."""
    resp = await seeded.client.get("/v1/activities", params={"cursor": "not-a-signed-cursor"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["type"].endswith("/invalid-cursor")
    assert body["status"] == 400


@pytest.mark.integration
async def test_cursor_mismatched_filters_is_cursor_parameter_mismatch(seeded: Env) -> None:
    """A cursor replayed against changed filters -> 400 cursor-parameter-mismatch (PAGE-R6)."""
    first = await seeded.client.get("/v1/activities", params={"sport": "cycling", "limit": 1})
    assert first.status_code == 200
    # Force a next page by seeding a second ride so the first page reports a cursor.
    await _seed_no_gps_activity(seeded.session, seeded.athlete_id)
    page = await seeded.client.get("/v1/activities", params={"limit": 1})
    cursor = page.json()["page"]["next_cursor"]
    assert cursor is not None
    # Reuse the cursor but with a different filter fingerprint than it was issued for.
    resp = await seeded.client.get(
        "/v1/activities", params={"limit": 1, "sport": "running", "cursor": cursor}
    )
    assert resp.status_code == 400
    assert resp.json()["type"].endswith("/cursor-parameter-mismatch")


@pytest.mark.integration
async def test_streams_base_values_are_seconds_not_indices(seeded: Env) -> None:
    """``base=time`` base_values are seconds from start (API-R48), index-aligned to channels."""
    resp = await seeded.client.get(
        f"/v1/activities/{seeded.activity_id}/streams",
        params={"channels": "power_w", "base": "time", "max_points": 10},
    )
    assert resp.status_code == 200
    body = resp.json()
    base = body["base_values"]
    assert body["base"] == "time"
    assert base[0] == 0.0  # seconds from start_time, not a bare sample index offset
    assert base == sorted(base)
    assert len(base) == len(body["channels"]["power_w"]["values"])


@pytest.mark.integration
async def test_streams_latlng_channel_is_rejected_422(seeded: Env) -> None:
    """A latlng request on /streams is a 422 (pair shape can't ride a scalar column, API-R48)."""
    resp = await seeded.client.get(
        f"/v1/activities/{seeded.activity_id}/streams", params={"channels": "latlng"}
    )
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/validation-error")


@pytest.mark.integration
async def test_activities_sort_by_duration_is_accepted(seeded: Env) -> None:
    """The PAGE-R2 sort allow-list (start_time|duration|tss) is accepted and applied."""
    resp = await seeded.client.get("/v1/activities", params={"sort": "duration", "order": "asc"})
    assert resp.status_code == 200
    bad = await seeded.client.get("/v1/activities", params={"sort": "bogus"})
    assert bad.status_code == 422  # outside the allow-list


# --- best-efforts (API-R30 §5) + threshold-history (API-R30 exception §12) -------


@pytest.mark.integration
async def test_best_efforts_is_paginated_collection_from_mmp(seeded: Env) -> None:
    """best-efforts is a PAGINATED collection of MMP-derived items, not a chart series (§5).

    Each item carries its OWN coverage + ``duration_s`` and the power equals ``MMP(d)``
    (BEST-R1, single source of truth). The envelope is the PAGE-R4 ``{data, page}`` wrapper
    (SCHEMA-R8), NOT the ``ChartSeries`` ``{items, x_axis, summary, ...}`` shape — pinning the
    page shape so a regression to ``ChartSeries`` (which would drop the per-item coverage
    contract) fails. Each item also carries the BEST-R2 lineage (``local_date`` +
    ``activity_id`` of the originating activity, MMP-R4).
    """
    resp = await seeded.client.get("/v1/performance/best-efforts", params=_range())
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body and "page" in body  # PAGE-R4 envelope, not {items, next_cursor}
    assert body["page"]["has_more"] is False and body["page"]["next_cursor"] is None
    assert "x_axis" not in body and "summary" not in body  # NOT a ChartSeries
    items = body["data"]
    assert items, "the constant-FTP ride yields best efforts for the MMP grid"
    assert body["page"]["limit"] == len(items)
    item = items[0]
    assert set(item) == {
        "duration_s", "label", "power_watts", "local_date", "activity_id", "coverage",
    }
    # The 1-hour 250 W ride: a short best effort equals MMP(d) ≈ 250 W (BEST-R1).
    powered = [i for i in items if i["power_watts"] is not None]
    assert powered and powered[0]["power_watts"] == pytest.approx(250.0, abs=1.0)
    assert powered[0]["coverage"]["present"] is True
    # BEST-R2 lineage: a computed best effort names the originating activity + its local date
    # (MMP-R4), so the agent can cite "your best 5-min power came from <activity on date>".
    assert powered[0]["local_date"] == _START.date().isoformat()
    assert powered[0]["activity_id"] == str(seeded.activity_id)
    _assert_no_source_name(body)


@pytest.mark.integration
async def test_threshold_history_reads_canonical_signature(seeded: Env) -> None:
    """threshold-history is a paginated canonical read of fitness_signature rows (API-R30 exc, §12).

    The seeded signature (effective 2026-01-01, FTP/CP 250, origin=measured) surfaces as ONE
    ``ThresholdPoint`` carrying the canonical threshold fields + the ``origin`` provenance class
    (NOT a source name, AUTH-R15). The envelope is the PAGE-R4 ``{data, page}`` page, not a
    ChartSeries (SCHEMA-R8).
    """
    resp = await seeded.client.get(
        "/v1/performance/threshold-history",
        params={"from": "2026-01-01", "to": "2026-06-07"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body and "page" in body  # PAGE-R4 envelope, not {items, next_cursor}
    assert body["page"]["has_more"] is False and body["page"]["next_cursor"] is None
    assert "x_axis" not in body  # NOT a ChartSeries
    items = body["data"]
    assert len(items) == 1
    assert body["page"]["limit"] == 1
    pt = items[0]
    assert pt["local_date"] == "2026-01-01"
    assert pt["ftp_w"] == pytest.approx(250.0)
    assert pt["cp_w"] == pytest.approx(250.0)
    assert pt["origin"] == "measured"  # canonical provenance class, not a source name
    assert pt["coverage"]["present"] is True
    _assert_no_source_name(body)


@pytest.mark.integration
async def test_threshold_history_empty_out_of_range(seeded: Env) -> None:
    """A range with no signature returns an empty PAGE-R4 page, not a fabricated point (ANL-R4)."""
    resp = await seeded.client.get(
        "/v1/performance/threshold-history",
        params={"from": "2025-01-01", "to": "2025-12-31"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == []  # PAGE-R4 envelope key, not `items`
    assert body["page"]["limit"] == 0 and body["page"]["has_more"] is False


@pytest.mark.integration
async def test_best_efforts_and_threshold_history_require_read_scope(seeded: Env) -> None:
    """Both new routes are gated on the ``read`` scope (AUTH-R11): no-read principal → 403."""
    seeded.app.dependency_overrides[perf_router.require_read_scope] = (
        perf_router.require_read_scope  # the fail-closed default raises insufficient-scope
    )
    try:
        be = await seeded.client.get("/v1/performance/best-efforts", params=_range())
        th = await seeded.client.get("/v1/performance/threshold-history", params=_range())
        assert be.status_code == 403
        assert th.status_code == 403
    finally:
        seeded.app.dependency_overrides[perf_router.require_read_scope] = lambda: None
