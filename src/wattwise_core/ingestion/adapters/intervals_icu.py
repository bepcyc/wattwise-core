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
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final

import httpx

from wattwise_core.config import Settings
from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import (
    AuthArchetype,
    Fidelity,
    GboType,
    SourceKind,
    StreamChannelName,
)
from wattwise_core.ingestion.adapters import _intervals_map as _im
from wattwise_core.ingestion.adapters import _intervals_sync as _isync
from wattwise_core.ingestion.adapters._intervals_asbo import (
    ActivityWithStreams,
    IntervalsActivityAsbo,
    IntervalsStreamAsbo,
    IntervalsWellnessAsbo,
)
from wattwise_core.ingestion.adapters._intervals_client import IntervalsIcuClient
from wattwise_core.ingestion.adapters._map_activity import RpeEncoding, feel_value, rpe_value

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
from wattwise_core.ingestion.capability import (  # noqa: import-direction
    AuthContext,
    CapabilityDescriptor,
    DiscoveryOrder,
    DiscoveryPage,
    DiscoveryRef,
    Granularity,
    SyncMode,
)
from wattwise_core.storage import content_hash

# Intervals.icu resolves athlete id "0" to the authenticated key's own athlete (the
# vendor's documented self-id convention); used when a connection carries no native id.
_SELF_ATHLETE_ID: Final = "0"


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
        # intervals.icu ``icu_rpe`` is already on the canonical CR-10 scale (SRPE-R2).
        "perceived_exertion": rpe_value(act.icu_rpe, RpeEncoding.CR10),
        "feel": feel_value(act.feel),
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
        # Canonical column names ONLY (MAP-R2): the source's weight/readiness/spO2/
        # respiration map onto the doc-20 daily_wellness fields, never source spellings.
        "body_mass_kg": well.weight,
        "readiness_external": well.readiness,
        "spo2_avg_pct": well.spO2,
        "respiration_avg_rpm": well.respiration,
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
    mapping_version: ClassVar[str] = "2"
    #: The static machine-readable capability declaration (ADP-R1), validated at
    #: registration (ADP-R2/ONB-R2). The declared GBO types gate the engine's upsert
    #: refusal (ADP-R3); the rate-limit VALUES live in config (CFG-R1a) — the
    #: descriptor carries only the machine-readable pointer to that section.
    capability: ClassVar[CapabilityDescriptor] = CapabilityDescriptor(
        source_key="intervals_icu",
        supported_gbo_types=frozenset({GboType.ACTIVITY, GboType.DAILY_WELLNESS}),
        sync_modes=frozenset({SyncMode.INCREMENTAL, SyncMode.BACKFILL}),
        auth_archetype=AuthArchetype.API_KEY,
        server_side_incremental=True,  # the listing supports an oldest/newest window
        discovery_order=DiscoveryOrder.OLDEST_FIRST,
        granularity={
            GboType.ACTIVITY: Granularity.FULL_STREAMS,
            GboType.DAILY_WELLNESS: Granularity.SUMMARY_ONLY,
        },
        # doc 20 §7.5 / DM-SUB-R1: the source contributes power-based training_load
        # members (per-second power streams) — consumed by the §9A withdrawal lifecycle.
        equivalence_classes=("training_load",),
        default_trust_profile=Fidelity.PLATFORM_COMPUTED,
        rate_limit_config_section="adapters.intervals_icu",
    )

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # Both seams are injectable for offline tests (CLI-R3/TST-R1); the entry-point
        # factory constructs no-args and resolves config lazily at first use (CFG-R1a).
        self._settings = settings
        self._transport = transport

    def _resolved_settings(self) -> Settings:
        """The adapter's config, resolved lazily once (CFG-R1a; injectable in tests)."""
        if self._settings is None:
            self._settings = Settings()
        return self._settings

    def _client(self, ctx: AuthContext) -> IntervalsIcuClient:
        """A per-call typed client bound to the run's validated credentials (stateless)."""
        if ctx.api_key is None:
            raise AuthError(FetchErrorKind.AUTH_REVOKED, "no usable credential")
        return IntervalsIcuClient.from_settings(
            ctx.api_key,
            ctx.athlete_native_id or _SELF_ATHLETE_ID,
            self._resolved_settings(),
            transport=self._transport,
        )

    async def ensure_authorized(
        self, *, api_key: str | None, athlete_native_id: str | None
    ) -> AuthContext:
        """Validate the credential with a read-only probe and return its context (ADP-R4).

        Returns a valid :class:`AuthContext` (never the raw secret in repr/logs), or
        raises the typed :class:`AuthError` taxonomy — a missing/rejected key is a
        terminal reauth condition (AUT-R4), never a silent degrade. No prompting, no
        secret logging, no global-state mutation.
        """
        if not api_key:
            raise AuthError(FetchErrorKind.AUTH_REVOKED, "no usable credential")
        ctx = AuthContext(athlete_native_id=athlete_native_id or _SELF_ATHLETE_ID, api_key=api_key)
        async with self._client(ctx) as client:
            await client.probe()  # AUT-R17: read-only; AuthError on 401/403
        return ctx

    async def discover(
        self,
        ctx: AuthContext,
        window: Any,
        *,
        cursor: str | None = None,
        since_watermark: _dt.datetime | None = None,
    ) -> DiscoveryPage:
        """One cursor page of lightweight refs for the window (ADP-R5/R6/R7).

        Pages activity refs first (watermark-filtered via the listing's last-modified
        hint, ADP-R6), then wellness refs; surfaces ``next_cursor`` so a partial
        discovery is reportable as a typed gap from exactly the broken cursor (ADP-R7).
        """
        stage, offset = _isync.parse_cursor(cursor)
        page_size = self._resolved_settings().adapters__intervals_icu__discover_page_size
        async with self._client(ctx) as client:
            if stage == "act":
                raw = await client.discover_activities(window.oldest, window.newest)
                refs = _isync.activity_refs(raw, since_watermark)
                return _isync.page_of(refs, offset, page_size, stage="act", last_stage=False)
            rows = await client.fetch_wellness(window.oldest, window.newest)
            refs = _isync.wellness_refs([str(w.id) for w in rows])
            return _isync.page_of(refs, offset, page_size, stage="well", last_stage=True)

    async def fetch_ref(self, ctx: AuthContext, ref: DiscoveryRef) -> Any:
        """Fetch ONE discovered record as a validated typed ASBO (ADP-R8/CLI-R2)."""
        async with self._client(ctx) as client:
            if ref.gbo_type is GboType.DAILY_WELLNESS:
                rows = await client.fetch_wellness(ref.source_native_id, ref.source_native_id)
                if not rows:
                    raise FetchError(FetchErrorKind.FETCH_FAILED, "wellness day not found")
                return rows[0]
            return await client.fetch_activity(ref.source_native_id)

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
