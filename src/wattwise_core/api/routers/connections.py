"""Connections router — the OSS connectable-source surface (``/v1/connections/*``).

This is one of the three surfaces where a source name is a legitimate part of the
consumer contract (AUTH-R15): the athlete is choosing/managing a data source, so the
source key and display name appear here (and only here, on Sync and Data-health). No
analytics/agent/dashboard surface ever names a source.

The OSS catalog is fixed by the spec (API-R42): exactly two connectable archetypes —
direct activity-file upload (``file_upload``) and one ``api_key`` source (Intervals.icu).
OAuth-redirect connectors and the ``/v1/connections`` OAuth start/callback are a
commercial overlay (COMM-R18) and are deliberately NOT mounted here.

Endpoints:

- ``GET /v1/connections/available`` (``read``) — the connectable-source catalog
  (API-R42); each entry is ``{source, display_name, auth_archetype, connect_hint}``.
- ``POST /v1/connections/{source}/initiate`` (``write``) — the archetype-discriminated
  :class:`ConnectionNextStep` union (API-R43/SCHEMA-R10): an ``api_key`` source returns
  ``{label, hint_url}``; a ``file_upload`` source returns ``{accepted_formats}`` and
  routes the client to ``POST /v1/imports``.
- ``POST /v1/connections/{source}/complete`` (``write``) — completes an ``api_key``
  connection (API-R44): the raw secret arrives over TLS, is handed to the credential
  store for envelope encryption, and is discarded immediately (AUTH-R16); a MANDATORY
  read-only probe runs BEFORE the connection may report ``connected`` (AUTH-R17); a
  failed probe yields ``422 credential-invalid`` with NO half-connected row.

Identity is server-derived from the bearer token (AUTH-R3) via the auth gate; the
client never supplies an athlete id. The credential-probe is an injectable seam the
app factory overrides with the registered adapter's read-only probe, so this router
never imports a concrete source adapter (ARCH-R22 / ONB-R4 — consumers select adapters
through the seam, never by importing a named adapter).

Requirement IDs: API-R27, API-R42, API-R43, API-R44, API-R46a, AUTH-R3, AUTH-R11,
AUTH-R15, AUTH-R16, AUTH-R17, SCHEMA-R4, SCHEMA-R10, ERR-R6, ERR-R8.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.api.auth import Scope, require_scopes
from wattwise_core.api.connection_catalog import (
    ACCEPTED_FILE_FORMATS,
    CATALOG_BY_SOURCE,
    FILE_IMPORT_SOURCE_KEY,
    INTERVALS_SOURCE_KEY,
    OSS_CATALOG,
    CatalogEntry,
)
from wattwise_core.api.deps import CurrentPrincipal, DbSession, RateLimit
from wattwise_core.api.errors import FieldError, ProblemError
from wattwise_core.domain.enums import AuthArchetype, ConnectionStatus
from wattwise_core.persistence.models import Connection, SourceDescriptor

router = APIRouter(prefix="/v1/connections", tags=["connections"], dependencies=[RateLimit])


# --------------------------------------------------------- credential-probe seam


class CredentialProbeError(Exception):
    """A read-only credential probe rejected the supplied secret (AUTH-R17).

    Raised by the probe seam when the adapter's read-only check fails (bad key /
    revoked / unreachable-with-this-credential). Carries no secret material and no
    source-specific detail; the router maps it to ``422 credential-invalid``.
    """


#: The probe seam: given a source key + the raw secret, run the adapter's MANDATORY
#: read-only check (AUTH-R17). Returns nothing on success; raises
#: :class:`CredentialProbeError` on a bad credential. The app factory overrides this
#: with the registered adapter's probe so this router never imports a named adapter
#: (ARCH-R22 / ONB-R4); tests inject a mock probe.
CredentialProbe = Callable[[str, str], Awaitable[None]]


async def _unconfigured_probe(source: str, secret: str) -> None:
    """Fail-closed default probe: refuse every credential until the factory wires one.

    The real probe is the registered adapter's read-only check, injected by the app
    factory. Until then no credential can pass — a connection is NEVER marked
    ``connected`` without a successful probe (AUTH-R17, fail-closed).
    """
    raise CredentialProbeError(source)


def credential_probe() -> CredentialProbe:
    """Provide the credential-probe seam; the app factory overrides it (AUTH-R17)."""
    return _unconfigured_probe


# --------------------------------------------------------- credential-store seam


class CredentialSink(BaseModel):
    """The minimal credential-store surface this router needs (AUTH-R16).

    A structural seam (``store`` only) so the router depends on a capability, not on
    the security package's concrete store. The app factory binds the process
    :class:`~wattwise_core.security.credentials.CredentialStore`; tests inject a fake.
    Envelope encryption + opaque-ref issuance live behind it; the raw secret is never
    persisted here (AUTH-R16).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    store: Callable[[str], str]


def credential_sink() -> CredentialSink:
    """Provide the credential-store seam; the app factory overrides it (AUTH-R16)."""
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


ProbeDep = Annotated[CredentialProbe, Depends(credential_probe)]
SinkDep = Annotated[CredentialSink, Depends(credential_sink)]
SourcePath = Annotated[str, Path(description="The connectable source key (catalog).")]


# --------------------------------------------------------------------------- wire shapes


class AvailableConnection(BaseModel):
    """One catalog entry (API-R42): a connectable source the athlete may connect."""

    source: str
    display_name: str
    auth_archetype: AuthArchetype
    connect_hint: str


class ConnectionCatalog(BaseModel):
    """The connectable-source catalog response (``GET /available``, API-R42)."""

    sources: list[AvailableConnection]


class ApiKeyNextStep(BaseModel):
    """``ConnectionNextStep`` for an ``api_key`` source (API-R43/SCHEMA-R10).

    ``kind`` discriminates the union on the wire; clients branch on it, never on the
    source name (SCHEMA-R10). ``hint_url`` points the athlete at where to find their
    key (distinct from the catalog ``connect_hint``).
    """

    kind: Literal[AuthArchetype.API_KEY] = AuthArchetype.API_KEY
    label: str
    hint_url: str


class FileUploadNextStep(BaseModel):
    """``ConnectionNextStep`` for a ``file_upload`` source (API-R43/SCHEMA-R10).

    Routes the client to ``POST /v1/imports``; carries the accepted formats so the
    file picker can filter (API-R33).
    """

    kind: Literal[AuthArchetype.FILE_UPLOAD] = AuthArchetype.FILE_UPLOAD
    accepted_formats: list[str]


#: The archetype-discriminated next-step union (SCHEMA-R10): a typed client branches on
#: ``kind`` (the OpenAPI ``discriminator``) without guessing the union member.
ConnectionNextStep = Annotated[ApiKeyNextStep | FileUploadNextStep, Field(discriminator="kind")]


class ApiKeyCompleteRequest(BaseModel):
    """Body for completing an ``api_key`` connection (API-R44).

    Carries ONLY the raw key (handed to the store over TLS, then discarded — AUTH-R16)
    and no caller-identity field (AUTH-R3). ``additionalProperties:false`` (SCHEMA-R4)
    rejects any unknown property, including a forged ``athlete_id``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: str = Field(min_length=1, max_length=512)


class ConnectionResult(BaseModel):
    """``ConnectionCompletionResult`` (connected branch) for an ``api_key`` source.

    The persisted connection summary after a successful probe (API-R44/API-R47). The
    source key/display name appear here per the AUTH-R15 exception. ``status`` is the
    canonical connection status (``connected``).
    """

    connection_id: str
    source: str
    display_name: str
    status: ConnectionStatus
    auth_archetype: AuthArchetype
    connected_at: datetime


# --------------------------------------------------------------------------- routes


@router.get(
    "/available",
    response_model=ConnectionCatalog,
    operation_id="listAvailableConnections",
    dependencies=[Depends(require_scopes(Scope.READ))],
)
async def list_available() -> ConnectionCatalog:
    """Return the fixed OSS connectable-source catalog (API-R42).

    Two archetypes only: a file-upload importer and one ``api_key`` source
    (Intervals.icu). OAuth-redirect connectors are a commercial overlay and never
    appear in the OSS catalog (COMM-R18).
    """
    return ConnectionCatalog(
        sources=[
            AvailableConnection(
                source=e.source,
                display_name=e.display_name,
                auth_archetype=e.auth_archetype,
                connect_hint=e.connect_hint,
            )
            for e in OSS_CATALOG
        ]
    )


@router.post(
    "/{source}/initiate",
    response_model=ConnectionNextStep,
    operation_id="initiateConnection",
    dependencies=[Depends(require_scopes(Scope.WRITE))],
)
async def initiate(source: SourcePath) -> ApiKeyNextStep | FileUploadNextStep:
    """Return the archetype-discriminated next step for connecting ``source`` (API-R43).

    An ``api_key`` source returns where to paste the key; a ``file_upload`` source
    returns the accepted formats and is completed by uploading to ``POST /v1/imports``
    (no ``complete`` call). An unknown source key → ``404 not-found`` (API-R51).
    """
    entry = _require_catalog_entry(source)
    if entry.auth_archetype is AuthArchetype.API_KEY:
        return ApiKeyNextStep(
            label="Your Intervals.icu API key",
            hint_url="https://intervals.icu/settings",
        )
    return FileUploadNextStep(accepted_formats=list(ACCEPTED_FILE_FORMATS))


@router.post(
    "/{source}/complete",
    response_model=ConnectionResult,
    operation_id="completeConnection",
    dependencies=[Depends(require_scopes(Scope.WRITE))],
)
async def complete(
    source: SourcePath,
    body: ApiKeyCompleteRequest,
    principal: CurrentPrincipal,
    session: DbSession,
    probe: ProbeDep,
    sink: SinkDep,
) -> ConnectionResult:
    """Complete an ``api_key`` connection: probe, then store, then persist (API-R44).

    Order is load-bearing (AUTH-R17): the MANDATORY read-only probe runs FIRST against
    the raw key; only on success is the key envelope-encrypted into an opaque
    ``credential_ref`` (the raw value is discarded, AUTH-R16) and a ``connected`` row
    written. A failed probe → ``422 credential-invalid`` and NO half-connected row is
    created. A non-``api_key`` source (e.g. ``file_upload``) → ``422`` (it has no
    ``complete`` step). An unknown source → ``404``.
    """
    entry = _require_catalog_entry(source)
    if entry.auth_archetype is not AuthArchetype.API_KEY:
        raise _wrong_archetype(entry.auth_archetype)
    await _run_probe(probe, source, body.api_key)
    credential_ref = sink.store(body.api_key)
    connection = await _persist_connected(session, principal.athlete_id, entry, credential_ref)
    return _to_result(connection, entry)


@router.post(
    "/{connection_id}/reconnect",
    response_model=ConnectionResult,
    operation_id="reconnectConnection",
    dependencies=[Depends(require_scopes(Scope.WRITE))],
)
async def reconnect(
    connection_id: Annotated[str, Path(description="An existing api_key connection id.")],
    body: ApiKeyCompleteRequest,
    principal: CurrentPrincipal,
    session: DbSession,
    probe: ProbeDep,
    sink: SinkDep,
) -> ConnectionResult:
    """Re-authenticate a stuck/errored ``api_key`` connection (API-R45).

    The recovery path for a ``reauth_required``/``error`` connection: the MANDATORY
    read-only probe runs FIRST against the new raw key; only on success is the key
    envelope-encrypted into a fresh ``credential_ref`` (the raw value discarded,
    AUTH-R16), the ref atomically replaced, and the status flipped back to ``connected``
    (AUTH-R17) while the row's history is preserved. A failed probe → ``422
    credential-invalid`` leaving the existing row untouched. An unknown/foreign
    connection → ``404`` (API-R51).
    """
    athlete_uuid = uuid.UUID(principal.athlete_id)
    connection = await _owned_connection(session, athlete_uuid, connection_id)
    entry = CATALOG_BY_SOURCE.get(await _source_key_for(session, connection.source_descriptor_id))
    if entry is None or entry.auth_archetype is not AuthArchetype.API_KEY:
        raise _wrong_archetype(connection.auth_archetype)
    await _run_probe(probe, entry.source, body.api_key)
    connection.credential_ref = sink.store(body.api_key)
    connection.status = ConnectionStatus.CONNECTED
    connection.connected_at = datetime.now(UTC)
    await session.flush()
    return _to_result(connection, entry)


# --------------------------------------------------------------------------- helpers


def _require_catalog_entry(source: str) -> CatalogEntry:
    """Look up a catalog entry by source key; unknown → ``404`` (API-R51)."""
    entry = CATALOG_BY_SOURCE.get(source)
    if entry is None:
        raise ProblemError("not-found")
    return entry


async def _run_probe(probe: CredentialProbe, source: str, secret: str) -> None:
    """Run the mandatory read-only probe; a failure → ``422 credential-invalid`` (AUTH-R17).

    The raw secret never reaches the problem document or any log line (AUTH-R16): a
    rejected probe surfaces only the catalog copy with a machine ``invalid_credential``
    code. No half-connected row exists — nothing was persisted before this point.
    """
    try:
        await probe(source, secret)
    except CredentialProbeError as exc:
        raise ProblemError(
            "credential-invalid",
            errors=[
                FieldError(
                    code="invalid_credential",
                    message="That key didn't work — double-check it and try again.",
                    pointer="/api_key",
                )
            ],
        ) from exc


async def _persist_connected(
    session: AsyncSession,
    athlete_id: str,
    entry: CatalogEntry,
    credential_ref: str,
) -> Connection:
    """Create-or-update the ``connected`` connection for this athlete+source (API-R44).

    Resolves the registered :class:`SourceDescriptor` for the source key and writes a
    row holding ONLY the opaque ``credential_ref`` (never the secret, AUTH-R16). A
    re-complete (reconnect within the same archetype) atomically replaces the ref and
    flips the status back to ``connected`` rather than minting a duplicate row.
    """
    athlete_uuid = uuid.UUID(athlete_id)
    descriptor = await _descriptor_for(session, entry.source)
    existing = await _existing_connection(session, athlete_uuid, descriptor.source_descriptor_id)
    now = datetime.now(UTC)
    if existing is None:
        connection = Connection(
            athlete_id=athlete_uuid,
            source_descriptor_id=descriptor.source_descriptor_id,
            status=ConnectionStatus.CONNECTED,
            credential_ref=credential_ref,
            scopes=[],
            connected_at=now,
            auth_archetype=entry.auth_archetype,
        )
        session.add(connection)
        await session.flush()
        return connection
    existing.credential_ref = credential_ref
    existing.status = ConnectionStatus.CONNECTED
    existing.connected_at = now
    await session.flush()
    return existing


async def _descriptor_for(session: AsyncSession, source_key: str) -> SourceDescriptor:
    """Resolve the registered source descriptor for ``source_key`` (LIN-R1).

    The descriptor is registration *data* seeded by migration; its absence is an
    operator-configuration fault (a catalog source with no registered descriptor),
    surfaced fail-closed as a generic internal error — never as a hint (ERR-R5).
    """
    stmt = select(SourceDescriptor).where(SourceDescriptor.source_key == source_key)
    descriptor = (await session.execute(stmt)).scalar_one_or_none()
    if descriptor is None:
        raise ProblemError("internal-error")
    return descriptor


async def _existing_connection(
    session: AsyncSession, athlete_id: uuid.UUID, descriptor_id: uuid.UUID
) -> Connection | None:
    """Return the athlete's existing connection for this source, if any (UNIQUE pair)."""
    stmt = select(Connection).where(
        Connection.athlete_id == athlete_id,
        Connection.source_descriptor_id == descriptor_id,
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def _owned_connection(
    session: AsyncSession, athlete_id: uuid.UUID, connection_id: str
) -> Connection:
    """Fetch the owner's connection by id; unknown/malformed/foreign → ``404`` (API-R51)."""
    try:
        target_id = uuid.UUID(connection_id)
    except ValueError as exc:
        raise ProblemError("not-found") from exc
    stmt = select(Connection).where(
        Connection.connection_id == target_id,
        Connection.athlete_id == athlete_id,
    )
    connection = (await session.execute(stmt)).scalar_one_or_none()
    if connection is None:
        raise ProblemError("not-found")
    return connection


async def _source_key_for(session: AsyncSession, descriptor_id: uuid.UUID) -> str:
    """Resolve the catalog source key for a connection's descriptor (reconnect lookup)."""
    descriptor = await session.get(SourceDescriptor, descriptor_id)
    if descriptor is None:
        raise ProblemError("internal-error")
    return descriptor.source_key


def _to_result(connection: Connection, entry: CatalogEntry) -> ConnectionResult:
    """Render a persisted connection to the completion result (API-R44/R47)."""
    return ConnectionResult(
        connection_id=str(connection.connection_id),
        source=entry.source,
        display_name=entry.display_name,
        status=connection.status,
        auth_archetype=connection.auth_archetype,
        connected_at=connection.connected_at or datetime.now(UTC),
    )


def _wrong_archetype(archetype: AuthArchetype) -> ProblemError:
    """A ``422`` for completing a source that has no ``api_key`` complete step (API-R44)."""
    return ProblemError(
        "validation-error",
        errors=[
            FieldError(
                code="unsupported_archetype",
                message="This source isn't connected with a key.",
                parameter="source",
            )
        ],
    )


__all__ = [
    "ACCEPTED_FILE_FORMATS",
    "FILE_IMPORT_SOURCE_KEY",
    "INTERVALS_SOURCE_KEY",
    "ApiKeyCompleteRequest",
    "ApiKeyNextStep",
    "AvailableConnection",
    "ConnectionCatalog",
    "ConnectionResult",
    "CredentialProbe",
    "CredentialProbeError",
    "CredentialSink",
    "FileUploadNextStep",
    "credential_probe",
    "credential_sink",
    "router",
]
