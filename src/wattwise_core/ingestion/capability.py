"""Adapter capability descriptor + engine-side declared-type enforcement (ADP-R1..R3).

Every adapter exposes a static, machine-readable :class:`CapabilityDescriptor`
(ADP-R1) declaring what it can produce and how it syncs. Registration validates the
descriptor and REJECTS an adapter that declares unsupported GBO types or omits a
required phase (ADP-R2 / ONB-R2) — fail-closed, never a silently-degraded source.
The engine plans runs from the descriptor, never from hard-coded source identity
(ADP-R2), and REFUSES an upsert of any GBO type the adapter did not declare
(ADP-R3): an undeclared candidate raises :class:`UndeclaredGboTypeError` before any
write — it is NEVER silently dropped from the canonical store (fail-closed against
data loss).

The discovery shapes (:class:`DiscoveryRef` / :class:`DiscoveryPage`) and the
authorize result (:class:`AuthContext`) are the typed phase contracts of §3.2
(ADP-R4/R5/R7). This module is rankless contract vocabulary: it imports only the
domain enums and the standard library.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import AuthArchetype, Fidelity, GboType

#: The GBO types the OSS engine can canonically WRITE today. A descriptor declaring
#: anything outside this set is rejected at registration (ONB-R2 "unsupported GBO
#: types"), so a declared type always has a real canonical write path (no silent drop).
ENGINE_WRITABLE_GBO_TYPES: frozenset[GboType] = frozenset(
    {GboType.ACTIVITY, GboType.DAILY_WELLNESS}
)


class SyncMode(StrEnum):
    """The sync modes an adapter may declare support for (ADP-R1 / SYN-R1)."""

    INCREMENTAL = "incremental"
    BACKFILL = "backfill"
    WEBHOOK = "webhook"
    FILE_IMPORT = "file_import"


class DiscoveryOrder(StrEnum):
    """The deterministic discover yield order a descriptor declares (ADP-R5)."""

    NEWEST_FIRST = "newest_first"
    OLDEST_FIRST = "oldest_first"


class Granularity(StrEnum):
    """Per-GBO-type data granularity a source can return (ADP-R1)."""

    SUMMARY_ONLY = "summary_only"
    FULL_STREAMS = "full_streams"


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """The static, machine-readable adapter capability declaration (ADP-R1).

    Declares: the stable ``source_key`` slug (IDS-R2 — the ``source_descriptor_id``
    uuid is registration data bound at runtime, so the STATIC descriptor carries the
    slug identity); every GBO type the adapter will write (ADP-R3); supported sync
    modes; the auth scheme (§5); whether the source supports server-side incremental
    filtering; the deterministic discover order (ADP-R5); per-GBO-type granularity;
    the metric-equivalence classes it satisfies (doc 20 §7.5 / DM-SUB-R1, consumed by
    the §9A withdrawal lifecycle); the default ``trust_profile`` (§6.3); and the
    native rate-limit profile as a pointer to its config section (§4.4 — the VALUES
    live in config per CFG-R1a, never code literals).
    """

    source_key: str
    supported_gbo_types: frozenset[GboType]
    sync_modes: frozenset[SyncMode]
    auth_archetype: AuthArchetype
    server_side_incremental: bool
    discovery_order: DiscoveryOrder
    granularity: Mapping[GboType, Granularity]
    equivalence_classes: tuple[str, ...]
    default_trust_profile: Fidelity
    rate_limit_config_section: str | None = None


@dataclass(frozen=True, slots=True)
class DiscoveryRef:
    """One lightweight discover reference (ADP-R5): native id + hint + GBO type."""

    source_native_id: str
    gbo_type: GboType
    last_modified: _dt.datetime | None = None


@dataclass(frozen=True, slots=True)
class DiscoveryPage:
    """One discover page surfacing its continuation cursor (ADP-R7).

    ``next_cursor`` is ``None`` exactly when discovery is complete; a non-``None``
    cursor names the next page so a mid-pagination failure is reportable as a typed
    gap covering exactly the un-discovered remainder (ADP-R7 / ING-GAP-R5).
    """

    refs: tuple[DiscoveryRef, ...]
    next_cursor: str | None = None


@dataclass(frozen=True, slots=True)
class AuthContext:
    """The validated, non-expired credential context ``ensure_authorized`` returns (ADP-R4).

    Carries the live credential for the run's fetch calls (``repr=False`` so it can
    never leak through repr/str into a log or error — AUT-R2) plus the resolved
    athlete-native identity. A failed authorize raises the typed
    :class:`~wattwise_core.ingestion.base.AuthError` taxonomy instead.
    """

    athlete_native_id: str | None = None
    api_key: str | None = field(default=None, repr=False)


class CapabilityError(ValueError):
    """An adapter's capability descriptor is absent or invalid (ADP-R2 / ONB-R2).

    Raised at REGISTRATION so a non-conforming adapter is rejected fail-closed —
    it never reaches the engine where its declarations would be load-bearing.
    """


class UndeclaredGboTypeError(RuntimeError):
    """The engine refused an upsert of a GBO type the adapter did not declare (ADP-R3).

    Fail-closed: raised BEFORE any canonical write, so an undeclared (or
    engine-unwritable) candidate can never be silently dropped — the refusal is
    typed, surfaced, and recorded as a gap by the sync flow.
    """

    def __init__(self, gbo_type: str, source_key: str) -> None:
        self.gbo_type = gbo_type
        self.source_key = source_key
        super().__init__(
            f"adapter {source_key!r} emitted undeclared/unwritable gbo_type {gbo_type!r}"
        )


def require_declared_types(
    candidates: Iterable[GboCandidate],
    declared: frozenset[GboType] | None,
    *,
    source_key: str,
) -> None:
    """REFUSE any candidate whose GBO type is undeclared or engine-unwritable (ADP-R3).

    Runs BEFORE the canonical write. ``declared=None`` (a caller with no descriptor
    in hand) still enforces the engine-writable set, so an unknown type is rejected,
    never silently ignored (ING-R3 fail-closed). Raises
    :class:`UndeclaredGboTypeError` on the first violation.
    """
    allowed = ENGINE_WRITABLE_GBO_TYPES if declared is None else declared
    for cand in candidates:
        try:
            gbo_type = GboType(cand.gbo_type)
        except ValueError:
            raise UndeclaredGboTypeError(str(cand.gbo_type), source_key) from None
        if gbo_type not in allowed or gbo_type not in ENGINE_WRITABLE_GBO_TYPES:
            raise UndeclaredGboTypeError(gbo_type.value, source_key)


def validate_capability(adapter: Any) -> CapabilityDescriptor:
    """Validate an adapter's capability descriptor at registration (ADP-R2 / ONB-R2).

    Rejects (raises :class:`CapabilityError`) when the descriptor is absent or not
    machine-readable, declares zero/unsupported GBO types, declares zero sync modes,
    contradicts the adapter's own identity attributes, or omits a phase its declared
    sync modes require: ``file_import`` needs ``decode_upload``; ``incremental`` /
    ``backfill`` need the authorize→discover→fetch trio (``ensure_authorized``,
    ``discover``, ``fetch_ref``) or the window-fetch seam (``fetch``). Returns the
    validated descriptor.
    """
    cap = getattr(adapter, "capability", None)
    if not isinstance(cap, CapabilityDescriptor):
        raise CapabilityError(
            f"adapter {getattr(adapter, 'source_key', '?')!r} exposes no CapabilityDescriptor"
        )
    problems = _capability_problems(adapter, cap)
    if problems:
        raise CapabilityError(
            f"adapter {cap.source_key!r} capability invalid: " + "; ".join(problems)
        )
    return cap


def _capability_problems(adapter: Any, cap: CapabilityDescriptor) -> list[str]:
    """Every ONB-R2 rejection reason the descriptor triggers (empty = valid)."""
    problems: list[str] = []
    if cap.source_key != getattr(adapter, "source_key", None):
        problems.append("source_key mismatch with the adapter")
    if not cap.supported_gbo_types:
        problems.append("declares no GBO types")
    unsupported = cap.supported_gbo_types - ENGINE_WRITABLE_GBO_TYPES
    if unsupported:
        problems.append(f"declares unsupported GBO types {sorted(t.value for t in unsupported)}")
    if not cap.sync_modes:
        problems.append("declares no sync modes")
    if cap.auth_archetype is not getattr(adapter, "auth_archetype", None):
        problems.append("auth_archetype mismatch with the adapter")
    extra_granularity = set(cap.granularity) - cap.supported_gbo_types
    if extra_granularity:
        problems.append("granularity declared for an undeclared GBO type")
    problems.extend(_phase_problems(adapter, cap))
    return problems


def _phase_problems(adapter: Any, cap: CapabilityDescriptor) -> list[str]:
    """Required-phase checks per declared sync mode (ONB-R2 'omits required phases')."""
    problems: list[str] = []
    if SyncMode.FILE_IMPORT in cap.sync_modes and not callable(
        getattr(adapter, "decode_upload", None)
    ):
        problems.append("declares file_import but omits decode_upload")
    pull_modes = {SyncMode.INCREMENTAL, SyncMode.BACKFILL}
    if cap.sync_modes & pull_modes:
        trio = all(
            callable(getattr(adapter, phase, None))
            for phase in ("ensure_authorized", "discover", "fetch_ref")
        )
        window_fetch = callable(getattr(adapter, "fetch", None))
        if not (trio or window_fetch):
            problems.append("declares incremental/backfill but omits the discover/fetch phases")
    return problems


__all__ = [
    "ENGINE_WRITABLE_GBO_TYPES",
    "AuthContext",
    "CapabilityDescriptor",
    "CapabilityError",
    "DiscoveryOrder",
    "DiscoveryPage",
    "DiscoveryRef",
    "Granularity",
    "SyncMode",
    "UndeclaredGboTypeError",
    "require_declared_types",
    "validate_capability",
]
