"""Source-adapter contract (ADP-R*, MAP-R1, Principle A/B).

Each source is a pluggable adapter — the ONLY code that knows a source's shape,
units, quirks, and ids (GBO-R3). An adapter's :meth:`SourceAdapter.map` is a **pure**
function (no clocks, no randomness, no network; MAP-R1) turning a source-shaped object
(ASBO) into canonical :class:`~wattwise_core.domain.candidate.GboCandidate` list.

Fetching (the impure part — network/file I/O) is separate from mapping, so mapping
stays unit- and golden-testable. Ingestion uses direct typed clients, never MCP
(Principle B); MCP exists only as the agent's runtime tool interface.

Adding a source is one adapter + one descriptor registration (ROAD-R6): no consumer,
analytics, or agent change. Adapters are discovered via the
``wattwise_core.adapters`` entry-point group.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import (
    ActivityFileFormat,
    AuthArchetype,
    GapReason,
    SourceKind,
)
from wattwise_core.ingestion.capability import CapabilityDescriptor


@dataclass(frozen=True, slots=True)
class SourceDescriptorRef:
    """The registered-source identity an adapter maps under (LIN-R1).

    This is lineage metadata, opaque to consumers. ``source_key`` and ``kind`` come
    from the registry, not from a hardcoded source name in consumer code.
    """

    source_descriptor_id: str
    source_key: str
    kind: SourceKind


@dataclass(frozen=True, slots=True)
class FetchContext:
    """Deterministic inputs a mapping may need that are NOT wall-clock/network.

    ``ingest_run_id`` and ``fetched_at`` are supplied by the caller (the sync
    orchestrator) so :meth:`SourceAdapter.map` stays pure — it never reads the clock.
    """

    ingest_run_id: str
    fetched_at: Any  # datetime, passed in (never read from the clock inside map)
    connection_id: str | None = None


@runtime_checkable
class SourceAdapter(Protocol):
    """The pluggable ingestion adapter contract (ADP-R*).

    Implementations declare their identity metadata as class attributes and provide a
    pure :meth:`map`. The fetch side (a direct typed client) lives on the concrete
    adapter but is invoked outside :meth:`map`.
    """

    source_key: str
    auth_archetype: AuthArchetype
    kind: SourceKind
    adapter_version: str
    mapping_version: str
    #: The static machine-readable capability declaration (ADP-R1); validated at
    #: registration (ADP-R2/ONB-R2) and the ONLY input the engine plans runs from.
    capability: CapabilityDescriptor

    def map(
        self,
        asbo: Any,
        source_descriptor: SourceDescriptorRef,
        fetch_context: FetchContext,
    ) -> list[GboCandidate]:
        """Map one source-shaped object into canonical candidates (MAP-R1).

        MUST be pure and deterministic: no I/O, no clock, no randomness. MUST emit
        only canonical fields + lineage (MAP-R2); convert to canonical units (MAP-R3);
        map source vocab to canonical enums/registries (MAP-R4); preserve real gaps as
        typed missing (MAP-R5); tag free text untrusted (MAP-R7).
        """
        ...


class FetchErrorKind(StrEnum):
    """The typed cause a client converts a fetch failure into (CLI-R2/CLI-R7).

    A client NEVER lets a raw ``httpx``/``pydantic`` exception leak to the engine; it
    converts the failure to a :class:`FetchError` carrying one of these kinds so the
    engine can branch on the CAUSE (auth vs schema vs transport) without parsing a
    stringly error. ``SCHEMA_MISMATCH`` is the CLI-R2 validation failure; the auth
    kinds carry the AUT-R4 revocation/expiry distinction.
    """

    SCHEMA_MISMATCH = "schema_mismatch"
    AUTH_REVOKED = "auth_revoked"
    AUTH_EXPIRED = "auth_expired"
    INSUFFICIENT_SCOPE = "insufficient_scope"
    RATE_LIMITED = "rate_limited"
    SOURCE_UNAVAILABLE = "source_unavailable"
    FETCH_FAILED = "fetch_failed"


class FetchError(Exception):
    """The single typed error a typed client raises to the engine (CLI-R2/CLI-R7).

    Non-transient failures (a 4xx other than 429, a schema mismatch, an auth error)
    MUST be converted to this typed shape and surfaced to the engine (CLI-R7) — never a
    raw ``httpx.HTTPStatusError`` or ``pydantic.ValidationError``. Carries ONLY the
    typed ``kind`` plus a short, non-sensitive ``detail``; never response bytes,
    credentials, or PII (AUT-R2/ING-SEC-R3).
    """

    def __init__(self, kind: FetchErrorKind, detail: str = "") -> None:
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind.value}: {detail}" if detail else kind.value)


class AuthError(FetchError):
    """A :class:`FetchError` whose cause is a credential break (AUT-R4).

    A revoked / expired / insufficient-scope credential (the OSS api_key 401/403 path)
    is surfaced as this typed error so the engine flips the Connection to
    ``reauth_required`` and stops the source rather than degrading silently. ``kind``
    defaults to ``AUTH_REVOKED`` and MUST be one of the auth kinds.
    """

    def __init__(
        self, kind: FetchErrorKind = FetchErrorKind.AUTH_REVOKED, detail: str = ""
    ) -> None:
        super().__init__(kind, detail)


@dataclass(frozen=True, slots=True)
class AuthGapSignal:
    """An in-memory auth-gap SIGNAL the fetch boundary hands the sync flow (§7, AUT-R4).

    NOT the persisted gap (that is the ORM ``persistence.models.source.IngestionGap``,
    opened via the watermark module's ``open_gap``). This is the lightweight, transport-
    level envelope a client/orchestrator passes around BEFORE the gap is written: it
    carries the canonical :class:`~wattwise_core.domain.enums.GapReason`, whether the
    failure is ``transient`` (auto-retryable) or terminal (needs user/operator action),
    and a short non-sensitive ``detail`` (never secrets/PII/response bytes). A terminal
    auth signal (``needs_reauth`` / ``auth_revoked``) is the AUT-R4 reauth indication the
    sync flow records as a persisted typed gap and the data-health surface (§9) renders.
    """

    reason: GapReason
    transient: bool
    detail: str = ""

    @classmethod
    def needs_reauth(cls, detail: str = "") -> AuthGapSignal:
        """A terminal reauth signal for a revoked/expired credential (AUT-R4)."""
        return cls(reason=GapReason.NEEDS_REAUTH, transient=False, detail=detail)


class FileImportError(Exception):
    """An uploaded recording file could not be decoded into canonical candidates (FIL-R*).

    The NEUTRAL, source-agnostic failure a file-import consumer catches: a concrete decoder's
    typed error is wrapped here so a consumer (the import seam) never imports a source-specific
    exception (ARCH-R22). Carries only a short, non-sensitive reason — never the raw bytes.
    """


@dataclass(frozen=True, slots=True)
class UploadDecode:
    """The result of decoding one uploaded recording file (FIL-R1, source-agnostic).

    ``candidates`` are the canonical activity candidate(s) the pure map produced; ``file_format``
    is the verbatim original's format for tier-1 capture. A consumer reads these without knowing
    which file type or adapter produced them (ARCH-R22).
    """

    candidates: list[GboCandidate]
    file_format: ActivityFileFormat


@runtime_checkable
class FileImportAdapter(Protocol):
    """A file-upload :class:`SourceAdapter` that decodes a verbatim uploaded file (FIL-R1).

    The file-import archetype seam: given raw bytes + the resolved descriptor/context it decodes
    (impure) and pure-maps to canonical candidates, reporting the original file format for tier-1
    capture. A consumer selects it through the registry by the built-in ``file_import`` key and
    drives THIS method — never importing a named adapter (ARCH-R22). A file that cannot be parsed
    raises :class:`FileImportError` (a neutral, source-agnostic error).
    """

    source_key: str

    def decode_upload(
        self,
        raw_bytes: bytes,
        *,
        filename: str | None,
        source_descriptor: SourceDescriptorRef,
        fetch_context: FetchContext,
    ) -> UploadDecode:
        """Decode + pure-map one uploaded file into canonical candidates (FIL-R1/MAP-R1)."""
        ...


__all__ = [
    "AuthError",
    "AuthGapSignal",
    "FetchContext",
    "FetchError",
    "FetchErrorKind",
    "FileImportAdapter",
    "FileImportError",
    "GapReason",
    "SourceAdapter",
    "SourceDescriptorRef",
    "UploadDecode",
]
