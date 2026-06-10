"""Capability-descriptor validation + declared-type refusal units (ADP-R1/R2/R3, ONB-R2).

The descriptor is load-bearing: registration REJECTS an adapter whose descriptor is
absent, declares unsupported GBO types, contradicts the adapter's identity, or omits
a phase its sync modes require (ONB-R2) — and the engine-side
:func:`require_declared_types` REFUSES (typed error, fail-closed) any candidate whose
GBO type is undeclared or engine-unwritable, the exact spot the audited ADP-R3
silent-drop data-loss bug lived. The redaction allowlist test pins that the
ING-OBS-R1 trace's numeric fields actually flow through the central redactor.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, ClassVar

import pytest

from tests.integration._fake_capability import fake_capability
from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import AuthArchetype, Fidelity, GboType
from wattwise_core.ingestion.capability import (
    CapabilityDescriptor,
    CapabilityError,
    DiscoveryOrder,
    Granularity,
    SyncMode,
    UndeclaredGboTypeError,
    require_declared_types,
    validate_capability,
)
from wattwise_core.ingestion.registry import AdapterRegistryError, registry_from_adapters
from wattwise_core.observability.logging import redact_processor

pytestmark = pytest.mark.unit


class _Fake:
    """A minimal window-fetch adapter shape carrying a configurable capability."""

    source_key: ClassVar[str] = "fake_src"
    auth_archetype: ClassVar[AuthArchetype] = AuthArchetype.API_KEY
    kind: ClassVar[Any] = "oauth_api"
    adapter_version: ClassVar[str] = "1"
    mapping_version: ClassVar[str] = "1"

    def __init__(self, capability: Any) -> None:
        self.capability = capability

    async def fetch(self, **_: Any) -> list[Any]:
        return []

    def map(self, asbo: Any, source_descriptor: Any, fetch_context: Any) -> list[Any]:
        return []


def _cap(**overrides: Any) -> CapabilityDescriptor:
    base: dict[str, Any] = dict(  # noqa: C408  (kwargs-style keeps overrides uniform)
        source_key="fake_src",
        supported_gbo_types=frozenset({GboType.ACTIVITY}),
        sync_modes=frozenset({SyncMode.INCREMENTAL}),
        auth_archetype=AuthArchetype.API_KEY,
        server_side_incremental=False,
        discovery_order=DiscoveryOrder.OLDEST_FIRST,
        granularity={GboType.ACTIVITY: Granularity.SUMMARY_ONLY},
        equivalence_classes=(),
        default_trust_profile=Fidelity.PLATFORM_COMPUTED,
        rate_limit_config_section=None,
    )
    base.update(overrides)
    return CapabilityDescriptor(**base)


def test_valid_capability_passes() -> None:
    """A complete, consistent descriptor validates and is returned (ADP-R2)."""
    adapter = _Fake(_cap())
    assert validate_capability(adapter) is adapter.capability


def test_missing_descriptor_is_rejected() -> None:
    """An adapter exposing no machine-readable descriptor is rejected (ADP-R1/ONB-R2)."""
    with pytest.raises(CapabilityError, match="no CapabilityDescriptor"):
        validate_capability(_Fake(capability=None))


def test_unsupported_gbo_type_is_rejected() -> None:
    """Declaring a GBO type the engine cannot write is an ONB-R2 registration rejection."""
    bad = _cap(
        supported_gbo_types=frozenset({GboType.ACTIVITY, GboType.FITNESS_SIGNATURE}),
        granularity={},
    )
    with pytest.raises(CapabilityError, match="unsupported GBO types"):
        validate_capability(_Fake(bad))


def test_empty_declarations_are_rejected() -> None:
    """Zero declared GBO types or zero sync modes is rejected (ONB-R2)."""
    with pytest.raises(CapabilityError, match="no GBO types"):
        validate_capability(_Fake(_cap(supported_gbo_types=frozenset(), granularity={})))
    with pytest.raises(CapabilityError, match="no sync modes"):
        validate_capability(_Fake(_cap(sync_modes=frozenset())))


def test_identity_mismatch_is_rejected() -> None:
    """A descriptor contradicting the adapter's own identity attrs is rejected (ADP-R2)."""
    with pytest.raises(CapabilityError, match="source_key mismatch"):
        validate_capability(_Fake(_cap(source_key="other")))
    with pytest.raises(CapabilityError, match="auth_archetype mismatch"):
        validate_capability(_Fake(_cap(auth_archetype=AuthArchetype.FILE_UPLOAD)))


def test_missing_required_phase_is_rejected() -> None:
    """Declaring incremental without ANY fetch phase is rejected (ONB-R2 'omits phases')."""

    class _NoFetch(_Fake):
        fetch = None  # type: ignore[assignment]

    with pytest.raises(CapabilityError, match="omits the discover/fetch phases"):
        validate_capability(_NoFetch(_cap()))


def test_granularity_for_undeclared_type_is_rejected() -> None:
    """Granularity entries must stay within the declared GBO-type set (ADP-R1)."""
    bad = _cap(granularity={GboType.DAILY_WELLNESS: Granularity.SUMMARY_ONLY})
    with pytest.raises(CapabilityError, match="granularity"):
        validate_capability(_Fake(bad))


def test_registry_rejects_invalid_capability() -> None:
    """Registration fails CLOSED on an invalid descriptor (ONB-R2 via the registry)."""
    with pytest.raises(AdapterRegistryError, match="capability"):
        registry_from_adapters([_Fake(_cap(sync_modes=frozenset()))])


def _candidate(gbo_type: str) -> GboCandidate:
    return GboCandidate(
        gbo_type=gbo_type,
        source_descriptor_id="d",
        source_native_id="n-1",
        content_hash="h",
        payload={},
        fetched_at=_dt.datetime(2026, 6, 1, tzinfo=_dt.UTC),
        trust_tier=Fidelity.SUMMARY_ONLY,
    )


def test_undeclared_type_is_refused() -> None:
    """ADP-R3: a candidate outside the declared set raises the typed refusal."""
    with pytest.raises(UndeclaredGboTypeError):
        require_declared_types(
            [_candidate("daily_wellness")], frozenset({GboType.ACTIVITY}), source_key="s"
        )


def test_unknown_type_is_refused_even_without_declaration() -> None:
    """A caller with no descriptor still cannot silently land an unknown type (fail-closed)."""
    with pytest.raises(UndeclaredGboTypeError):
        require_declared_types([_candidate("not_a_real_type")], None, source_key="s")
    # Engine-unwritable canonical types are refused too, never silently ignored:
    with pytest.raises(UndeclaredGboTypeError):
        require_declared_types([_candidate("fitness_signature")], None, source_key="s")


def test_declared_types_pass() -> None:
    """Candidates within the declared, engine-writable set are accepted (ADP-R3)."""
    require_declared_types(
        [_candidate("activity"), _candidate("daily_wellness")],
        frozenset({GboType.ACTIVITY, GboType.DAILY_WELLNESS}),
        source_key="s",
    )


def test_fake_capability_helper_is_itself_valid() -> None:
    """The shared test helper produces a descriptor real validation accepts."""
    adapter = _Fake(fake_capability("fake_src"))
    assert validate_capability(adapter).source_key == "fake_src"


def test_run_trace_numeric_fields_survive_redaction() -> None:
    """ING-OBS-R1/ING-OBS-R3: the sync trace's per-phase fields flow; PII still masked."""
    event = {
        "event": "sync.run_trace",
        "source_key": "intervals_icu",
        "outcome": "ok",
        "discover_ms": 12.5,
        "refs_discovered": 4,
        "gaps_opened": 1,
        "watermarks_advanced": 2,
        "athlete_name": "Jane Doe",  # never allowlisted -> must be masked
    }
    redacted = dict(redact_processor(None, "info", dict(event)))
    assert redacted["discover_ms"] == 12.5
    assert redacted["refs_discovered"] == 4
    assert redacted["gaps_opened"] == 1
    assert redacted["watermarks_advanced"] == 2
    assert redacted["athlete_name"] != "Jane Doe"
