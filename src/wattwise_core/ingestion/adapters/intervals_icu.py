"""Intervals.icu ``api_key`` source adapter (CLI-R13, ADP-R*, AUT-R17 probe).

Intervals.icu is the OSS direct-API source. Auth is HTTP Basic with the literal
username ``API_KEY`` and the athlete's API key as the password (CLI-R13). The thin
typed client (:class:`IntervalsIcuClient`) owns the impure I/O — probe/discover/fetch
over ``httpx.AsyncClient`` (CLI-R1/R3/R4); the stale ``intervalsicu`` PyPI package is
deliberately NOT used. A mandatory read-only :meth:`IntervalsIcuClient.probe` (GET
athlete) MUST succeed before a connection reports ``connected`` (AUT-R17).

:meth:`IntervalsIcuAdapter.map` is **pure and deterministic** (MAP-R1): no clock, no
randomness, no network. It turns one source-shaped activity-or-wellness payload (ASBO)
into canonical :class:`~wattwise_core.domain.candidate.GboCandidate` records carrying
ONLY canonical field names (MAP-R2), SI units (MAP-R3), canonical sport codes (MAP-R4),
typed gaps as ``None`` never ``0`` (MAP-R5), and free text tagged untrusted (MAP-R7).
``observed_at`` / ``fetched_at`` come from the :class:`FetchContext`; the map never
reads the wall clock. The adapter depends only on its client + canonical models +
lineage/enums (ADP-R16) and is fully exercisable offline against recorded fixtures
(ADP-R17, TST-R1).
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Final

import httpx

from wattwise_core.config import Settings
from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import (
    AuthArchetype,
    Fidelity,
    SourceKind,
    StreamChannelName,
)
from wattwise_core.ingestion.adapters import _intervals_map as _im
from wattwise_core.ingestion.adapters._intervals_asbo import (
    ActivityWithStreams,
    IntervalsActivityAsbo,
    IntervalsStreamAsbo,
    IntervalsWellnessAsbo,
)
from wattwise_core.ingestion.adapters._resilience import (
    AttemptBudget,
    RetryExhaustedError,
    TokenBucket,
    intervals_icu_bucket,
    intervals_icu_budget,
    resilient_get,
    validate_or_fetch_error,
)

# ``ingestion.base`` is the rankless adapter CONTRACT (the SourceAdapter Protocol,
# SourceDescriptorRef, FetchContext) that every L2 adapter is DEFINED against — the
# one inbound edge an adapter must have. The layer linter ranks the whole
# ``ingestion`` subpackage L3 and does not carve the contract module out the way it
# carves ``domain`` out, so this single contract import is suppressed (ARCH-R21).
from wattwise_core.ingestion.base import (  # noqa: import-direction
    AuthError,
    FetchContext,
    FetchError,
    FetchErrorKind,
    SourceDescriptorRef,
)
from wattwise_core.storage import content_hash

_BASE_URL: Final = "https://intervals.icu"
_BASIC_USERNAME: Final = "API_KEY"  # literal username per CLI-R13 (NOT a secret)


class IntervalsIcuClient:
    """Thin typed ``httpx.AsyncClient`` for Intervals.icu (impure I/O; CLI-R1/R3).

    Construction takes an injectable ``transport`` so tests substitute a fixture
    transport with no live network (CLI-R3, TST-R1). All calls are bounded by a
    connect + total timeout (CLI-R4). This object owns NO mapping logic.
    """

    def __init__(
        self,
        api_key: str,
        athlete_id: str,
        *,
        base_url: str = _BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 30.0,
        budget: AttemptBudget,
        bucket: TokenBucket | None = None,
    ) -> None:
        self._athlete_id = athlete_id
        self._budget = budget
        self._bucket = bucket
        self._client = httpx.AsyncClient(
            base_url=base_url,
            auth=httpx.BasicAuth(_BASIC_USERNAME, api_key),
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
        )

    @classmethod
    def from_settings(
        cls,
        api_key: str,
        athlete_id: str,
        settings: Settings,
        *,
        base_url: str = _BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        bucket: TokenBucket | None = None,
    ) -> IntervalsIcuClient:
        """Build the production client with its resilience read FROM config (CFG-R1a).

        The per-source retry budget (CLI-R6) + the request timeout (CLI-R4) come from the
        ``adapters.intervals_icu`` config section — NO resilience literal is baked into code
        (CFG-R1a: a value absent from every layer fails closed at load, never a code default).
        When no shared ``bucket`` is supplied, the client-side token-bucket limiter (CLI-R10/R11)
        is also built from config, with the adaptive-429 ``reduce_factor`` / ``min_rate`` set so a
        quota signal persistently lowers the issue rate rather than failing closed. The
        budget/bucket assembly lives in :mod:`_resilience` (its primitives' home).
        """
        return cls(
            api_key,
            athlete_id,
            base_url=base_url,
            transport=transport,
            timeout=settings.adapters__intervals_icu__http_timeout_s,
            budget=intervals_icu_budget(settings),
            bucket=bucket if bucket is not None else intervals_icu_bucket(settings),
        )

    async def __aenter__(self) -> IntervalsIcuClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        """Issue a resilient read GET: token-bucket throttle + retry/backoff + typed errors.

        The per-source token bucket (CLI-R10) gates issue rate client-side; the GET is
        retried on a transient 429/5xx/reset/timeout with exponential backoff + full
        jitter, bounded by the per-source budget (CLI-R6), honoring ``Retry-After`` on a
        429 (CLI-R11). A 401/403 becomes a typed :class:`AuthError` (AUT-R4); any other
        non-2xx (incl. a budget-exhausted transient) becomes a typed :class:`FetchError`
        (CLI-R7) — never a raw ``httpx`` exception leaking to the engine.
        """
        resp = await self._raw_get(path, params=params)
        if resp.status_code >= 400:
            raise FetchError(FetchErrorKind.FETCH_FAILED, f"source returned {resp.status_code}")
        return resp

    async def _raw_get(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        """Resilient GET that still converts auth/transient faults but RETURNS a non-auth 4xx.

        Used for the optional streams endpoint, where a ``404`` legitimately means "no
        streams" rather than a failure: the auth (AUT-R4) and budget-exhaustion (CLI-R7)
        conversions still apply, but a non-auth 4xx is returned for the caller to treat as
        empty. The token bucket (CLI-R10) and retry/backoff budget (CLI-R6) gate the call;
        on a 429 the shared bucket's issue rate is adaptively reduced (CLI-R11).
        """
        if self._bucket is not None:
            await self._bucket.acquire()
        bucket = self._bucket
        on_quota: Callable[[], object] | None = (
            bucket.reduce_rate  # CLI-R11: a 429 lowers the shared bucket's issue rate
            if bucket is not None and bucket.adaptive_reduction_enabled
            else None
        )
        try:
            resp = await resilient_get(
                self._client, path, budget=self._budget, params=params, on_quota_signal=on_quota
            )
        except RetryExhaustedError as exc:
            raise FetchError(
                FetchErrorKind.SOURCE_UNAVAILABLE, "source unavailable after retries"
            ) from exc
        if resp.status_code in (401, 403):
            raise AuthError(
                FetchErrorKind.AUTH_REVOKED
                if resp.status_code == 401
                else FetchErrorKind.INSUFFICIENT_SCOPE,
                "source credential rejected",
            )
        return resp

    async def probe(self) -> Mapping[str, Any]:
        """Mandatory read-only credential probe (AUT-R17): GET the athlete profile.

        Returns the profile mapping on success; raises a typed :class:`AuthError` on a
        401/403 (AUT-R4) so the caller never marks the connection ``connected``.
        """
        resp = await self._get(f"/api/v1/athlete/{self._athlete_id}")
        result: Mapping[str, Any] = resp.json()
        return result

    async def discover_activities(self, oldest: str, newest: str) -> list[dict[str, Any]]:
        """List activity summaries in an ISO-date window (ADP-R5; oldest/newest)."""
        resp = await self._get(
            f"/api/v1/athlete/{self._athlete_id}/activities",
            params={"oldest": oldest, "newest": newest},
        )
        payload: list[dict[str, Any]] = resp.json()
        return payload

    async def fetch_activity(self, activity_id: str) -> ActivityWithStreams:
        """Fetch one activity detail + its streams as a validated ASBO (ADP-R8, CLI-R2)."""
        detail = await self._get(f"/api/v1/activity/{activity_id}")
        streams = await self._raw_get(
            f"/api/v1/activity/{activity_id}/streams",
            params={"types": ",".join(_im.STREAM_CHANNELS)},
        )
        stream_rows = streams.json() if streams.status_code == httpx.codes.OK else []
        return ActivityWithStreams(
            activity=validate_or_fetch_error(IntervalsActivityAsbo, detail.json()),
            streams=[validate_or_fetch_error(IntervalsStreamAsbo, s) for s in stream_rows],
        )

    async def fetch_wellness(self, oldest: str, newest: str) -> list[IntervalsWellnessAsbo]:
        """Fetch daily-wellness rows in an ISO-date window (ADP-R8, CLI-R2)."""
        resp = await self._get(
            f"/api/v1/athlete/{self._athlete_id}/wellness",
            params={"oldest": oldest, "newest": newest},
        )
        rows: list[dict[str, Any]] = resp.json()
        return [validate_or_fetch_error(IntervalsWellnessAsbo, r) for r in rows]


def _parse_utc(value: str | None) -> _dt.datetime | None:
    """Parse a source ISO-8601 instant to a tz-aware UTC datetime (IDS-R3, MAP-R3).

    Returns ``None`` for an absent or unparseable value (a typed gap, never a guess).
    A trailing ``Z`` is normalized; a naive value is rejected (returns ``None``).
    """
    if not value:
        return None
    try:
        parsed = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(_dt.UTC)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    """Deterministic sha256 over the canonical payload (MAP-R8; stable across runs)."""
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return content_hash(encoded.encode("utf-8"))


def _activity_payload(
    act: IntervalsActivityAsbo,
    start_time: _dt.datetime,
    streams: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the canonical ``activity`` payload (MAP-R2/R3; SI units, no source keys)."""
    return {
        "start_time": start_time,
        "sport": _im.sport_code(act.type),
        "sub_sport": _im.sub_sport_code(act.sub_type),
        "elapsed_time_s": act.elapsed_time,
        "moving_time_s": act.moving_time,
        "distance_m": act.distance,
        "total_work_j": act.icu_joules,
        "energy_kj": None if act.icu_joules is None else act.icu_joules / 1000.0,
        "avg_power_w": act.icu_average_watts,
        "max_power_w": act.p_max,
        "avg_hr_bpm": act.average_heartrate,
        "max_hr_bpm": act.max_heartrate,
        "avg_cadence_rpm": act.average_cadence,
        "avg_speed_mps": act.average_speed,
        "elevation_gain_m": act.total_elevation_gain,
        "avg_temp_c": act.average_temp,
        "device_class": _im.device_class(act),
        "has_power": act.device_watts is True or act.icu_average_watts is not None,
        "has_hr": bool(act.has_heartrate) or act.average_heartrate is not None,
        "has_gps": StreamChannelName.LATLNG.value in streams,
        "has_cadence": act.average_cadence is not None
        or StreamChannelName.CADENCE_RPM.value in streams,
        "streams": streams,
    }


def _wellness_payload(well: IntervalsWellnessAsbo, local_date: _dt.date) -> dict[str, Any]:
    """Assemble the canonical ``daily_wellness`` payload (doc 20 §3.5; MAP-R2/R5)."""
    return {
        "local_date": local_date,
        "resting_hr_bpm": well.restingHR,
        "hrv_rmssd_ms": well.hrv,
        "hrv_sdnn_ms": well.hrvSDNN,
        "sleep_score": well.sleepScore,
        "sleep_duration_s": well.sleepSecs,
        "steps": well.steps,
        "weight_kg": well.weight,
        "readiness": well.readiness,
        "spo2_pct": well.spO2,
        "respiration_rpm": well.respiration,
        "vo2max": well.vo2max,
    }


class IntervalsIcuAdapter:
    """The Intervals.icu pluggable adapter (ADP-R*; satisfies ``SourceAdapter``).

    Identity metadata is declared as class attributes (ADP-R1). :meth:`map` is pure
    (MAP-R1); the fetch side lives on :class:`IntervalsIcuClient` and is invoked by
    the sync engine OUTSIDE ``map``.
    """

    source_key: ClassVar[str] = "intervals_icu"
    auth_archetype: ClassVar[AuthArchetype] = AuthArchetype.API_KEY
    kind: ClassVar[SourceKind] = SourceKind.OAUTH_API
    adapter_version: ClassVar[str] = "1"
    mapping_version: ClassVar[str] = "1"

    def map(
        self,
        asbo: Any,
        source_descriptor: SourceDescriptorRef,
        fetch_context: FetchContext,
    ) -> list[GboCandidate]:
        """Map one source object into canonical candidates (MAP-R1; pure/deterministic).

        Dispatches on the ASBO type: an :class:`ActivityWithStreams` (or bare
        :class:`IntervalsActivityAsbo`) -> one ``activity`` candidate; an
        :class:`IntervalsWellnessAsbo` -> one ``daily_wellness`` candidate. An
        un-mappable required field (e.g. an absent/naive ``start_date``) yields no
        candidate rather than a fabricated value (ING-R3/MAP-R5).
        """
        if isinstance(asbo, IntervalsWellnessAsbo):
            return self._map_wellness(asbo, source_descriptor, fetch_context)
        if isinstance(asbo, ActivityWithStreams):
            return self._map_activity(asbo.activity, asbo.streams, source_descriptor, fetch_context)
        if isinstance(asbo, IntervalsActivityAsbo):
            return self._map_activity(asbo, [], source_descriptor, fetch_context)
        return []

    def _map_activity(
        self,
        act: IntervalsActivityAsbo,
        streams: Sequence[IntervalsStreamAsbo],
        descriptor: SourceDescriptorRef,
        ctx: FetchContext,
    ) -> list[GboCandidate]:
        start_time = _parse_utc(act.start_date) or _parse_utc(act.start_date_local)
        if start_time is None:
            return []  # required canonical field absent -> no fabricated candidate
        canonical_streams = _im.build_streams(streams)
        payload = _activity_payload(act, start_time, canonical_streams)
        has_real_stream = bool(canonical_streams)
        untrusted = bool(act.name or act.description)
        return [
            GboCandidate(
                gbo_type="activity",
                source_descriptor_id=descriptor.source_descriptor_id,
                source_native_id=str(act.id),
                content_hash=_stable_hash(payload),
                payload=payload,
                observed_at=start_time,
                fetched_at=ctx.fetched_at,
                confidence=1.0,
                trust_tier=(Fidelity.RAW_STREAM if has_real_stream else Fidelity.PLATFORM_COMPUTED),
                untrusted_content=untrusted,
                connection_id=ctx.connection_id,
                adapter_version=self.adapter_version,
                mapping_version=self.mapping_version,
            )
        ]

    def _map_wellness(
        self,
        well: IntervalsWellnessAsbo,
        descriptor: SourceDescriptorRef,
        ctx: FetchContext,
    ) -> list[GboCandidate]:
        try:
            local_date = _dt.date.fromisoformat(well.id)
        except ValueError:
            return []  # the wellness id must be a local ISO date; else no candidate
        payload = _wellness_payload(well, local_date)
        observed = _dt.datetime.combine(local_date, _dt.time(), tzinfo=_dt.UTC)
        return [
            GboCandidate(
                gbo_type="daily_wellness",
                source_descriptor_id=descriptor.source_descriptor_id,
                source_native_id=str(well.id),
                content_hash=_stable_hash(payload),
                payload=payload,
                observed_at=observed,
                fetched_at=ctx.fetched_at,
                confidence=1.0,
                trust_tier=Fidelity.SUMMARY_ONLY,
                untrusted_content=False,
                connection_id=ctx.connection_id,
                adapter_version=self.adapter_version,
                mapping_version=self.mapping_version,
            )
        ]


__all__ = [
    "ActivityWithStreams",
    "IntervalsActivityAsbo",
    "IntervalsIcuAdapter",
    "IntervalsIcuClient",
    "IntervalsStreamAsbo",
    "IntervalsWellnessAsbo",
]
