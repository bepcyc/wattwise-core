"""Production composition root for the ingestion/credential router seams (ARCH-R22).

:func:`wattwise_core.api.app.create_app` binds the connect/import/sync routers'
injectable seams to the REAL OSS services here, so the routers themselves never import a
named adapter (ARCH-R22 / ONB-R4): they depend only on the seam Protocols, and this one
module — the composition root — is the single place allowed to know the file-upload
adapter and the on-demand sync orchestrator.

What is wired (so the built stack can actually connect → sync → land canonical data):

- **import processor** (``POST /v1/imports``) — decode (impure) → pure ``map`` → land one
  connectionless ``file_import`` candidate through :class:`IngestService` in one
  transaction (LIN-R1.1, UPS-R6), capturing the verbatim original tier-1 (FIL-R1).
- **sync orchestrator** (``POST /v1/sync/run``) — the real :class:`SyncOrchestrator` over
  the entry-point adapter registry + credential store, returning the started-run handle.
- **credential sink** (``POST /v1/connections/{source}/complete``) — the in-memory
  envelope-encrypting credential store (AUTH-R16), shared with the sync orchestrator so a
  stored ``credential_ref`` resolves at sync time.

The api_key **credential probe** is deliberately left at its fail-closed default: a
read-only probe is a live external call (AUTH-R17) AND OSS captures no source-athlete id
at connect time, so the api_key connect path degrades closed (``422``) rather than
fabricating a ``connected`` row — the file-upload path is the offline-complete ingestion
journey. Configuring a probe upgrades the api_key path in place through the same seam.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.api.connection_catalog import FILE_IMPORT_SOURCE_KEY
from wattwise_core.api.errors import ProblemError
from wattwise_core.api.routers.connections import CredentialSink
from wattwise_core.api.routers.imports import ImportJob, ImportProcessor, ImportRejected, queued_job
from wattwise_core.api.routers.sync import SyncRun as RouterSyncRun
from wattwise_core.api.routers.sync import SyncTarget, started_run
from wattwise_core.config import Settings
from wattwise_core.domain.enums import SourceKind
from wattwise_core.ingestion.base import (
    FetchContext,
    FileImportAdapter,
    FileImportError,
    SourceDescriptorRef,
)
from wattwise_core.ingestion.ingest import IngestService, OriginalFile
from wattwise_core.ingestion.registry import AdapterRegistry, load_registry
from wattwise_core.ingestion.sync import SyncOrchestrator
from wattwise_core.persistence import Database
from wattwise_core.persistence.models import SourceDescriptor
from wattwise_core.persistence.types import utcnow, uuid7
from wattwise_core.seams import EngineSessionProvider
from wattwise_core.security.credentials import CredentialStore, InMemoryCredentialStore
from wattwise_core.security.crypto import EnvelopeCipher
from wattwise_core.storage import ObjectStore, create_object_store

#: The sync router's seam shape: a server-resolved target → a started-run handle (API-R46a).
RouterSyncOrchestrator = Callable[[SyncTarget], Awaitable[RouterSyncRun]]


@dataclass(frozen=True, slots=True)
class IngestionSeams:
    """The concrete provider for each ingestion/credential router seam (API-R3).

    ``credential_sink`` is ``None`` when no ``encryption_root_key`` is configured (the OSS
    development default): without an envelope key no secret may be sealed (AUTH-R16), so
    the connect-``complete`` sink stays at its fail-closed default rather than storing a
    secret in the clear — the api_key probe already fails closed first, so the path is
    ``422``, never a half-connected row.
    """

    import_processor: ImportProcessor
    sync_orchestrator: RouterSyncOrchestrator
    credential_sink: CredentialSink | None


def build_ingestion_seams(database: Database, settings: Settings) -> IngestionSeams:
    """Assemble the real connect/import/sync seam providers for the app factory.

    One credential store backs BOTH the connect sink and the sync orchestrator so a
    ``credential_ref`` stored at connect time resolves at sync time (SEC-R7). The adapter
    registry is the entry-point set (ROAD-R6); no named adapter is imported by a router. A
    deployment with no envelope key configured gets no credential store (and so no sink):
    the file-upload import + sync trigger remain fully functional without one.
    """
    registry = load_registry()
    store = _build_credential_store(settings)
    return IngestionSeams(
        import_processor=_make_import_processor(database, settings, registry),
        sync_orchestrator=_make_sync_orchestrator(database, registry, store),
        credential_sink=CredentialSink(store=store.store) if store is not None else None,
    )


def _build_credential_store(settings: Settings) -> CredentialStore | None:
    """The envelope-encrypting credential store when a root key is configured (AUTH-R16).

    Returns ``None`` when no ``encryption_root_key`` is set: the crypto layer refuses to
    seal without a real key (it never falls back to an ephemeral default), and OSS dev may
    boot keyless, so credentials are simply unavailable until a key is supplied (BOOT-R4).
    """
    root = settings.encryption_root_key
    if root is None:
        return None
    return InMemoryCredentialStore(EnvelopeCipher(root.get_secret_value()))


def _make_import_processor(
    database: Database, settings: Settings, registry: AdapterRegistry
) -> ImportProcessor:
    """Build the file-upload import processor: decode → map → land (API-R33, FIL-R1).

    Source-blind (ARCH-R22): the file-import adapter is selected from the registry by the
    built-in ``file_import`` key and driven through the
    :class:`~wattwise_core.ingestion.base.FileImportAdapter` seam — this module imports no
    named adapter. The object store is constructed LAZILY on the first upload (it provisions
    its backing directory on construction), so the app factory never touches the filesystem
    at boot — a read-only/unprovisioned environment must not require it.
    """
    store_cache: list[ObjectStore] = []
    # The upload's canonical write flows through the ONE engine-owned session provider seam
    # (SEAM-R11 / ARCH-R31), keyed on the server-derived ``athlete_id``, never around it.
    sessions = EngineSessionProvider(database)

    def _object_store() -> ObjectStore:
        if not store_cache:
            store_cache.append(create_object_store(settings))
        return store_cache[0]

    async def process(athlete_id: str, data: bytes, filename: str | None) -> ImportJob:
        adapter = _file_import_adapter(registry)
        ctx = FetchContext(ingest_run_id=str(uuid7()), fetched_at=utcnow(), connection_id=None)
        async with sessions.session(subject=athlete_id) as session:
            descriptor = await _file_import_descriptor(session)
            ref = SourceDescriptorRef(
                source_descriptor_id=str(descriptor.source_descriptor_id),
                source_key=FILE_IMPORT_SOURCE_KEY,
                kind=SourceKind(descriptor.kind),
            )
            try:
                decoded = adapter.decode_upload(
                    data, filename=filename, source_descriptor=ref, fetch_context=ctx
                )
            except FileImportError as exc:
                raise ImportRejected(
                    code="unreadable_file", reason="We couldn't read that file."
                ) from exc
            if not decoded.candidates:
                raise ImportRejected(
                    code="no_activity", reason="That file had no activity we could read."
                )
            original = OriginalFile(
                data=data,
                file_format=decoded.file_format,
                source_native_id=decoded.candidates[0].source_native_id,
            )
            result = await IngestService(
                session,
                object_store=_object_store(),
                batch_size=settings.ingestion__batch_size,
            ).ingest(
                athlete_id,
                descriptor.source_descriptor_id,
                decoded.candidates,
                ingest_run_id=uuid.UUID(ctx.ingest_run_id),
                original_files=[original],
                # ADP-R3: the import path enforces the same declared-type contract —
                # the engine refuses any candidate type the adapter did not declare.
                declared_gbo_types=getattr(adapter, "capability", None)
                and adapter.capability.supported_gbo_types,  # type: ignore[attr-defined]
                source_key=FILE_IMPORT_SOURCE_KEY,
            )
        job_id = next(iter(result.activities_written), ctx.ingest_run_id)
        return queued_job(job_id, filename)

    return process


def _file_import_adapter(registry: AdapterRegistry) -> FileImportAdapter:
    """Resolve the built-in file-import adapter by its registered key (ARCH-R22).

    Looked up by the canonical ``file_import`` source key (LIN-R1.1), never by importing a
    named adapter. An installed adapter that does not satisfy the file-import seam is an
    operator-configuration fault surfaced fail-closed as a generic internal error (ERR-R5).
    """
    adapter = registry.get(FILE_IMPORT_SOURCE_KEY)
    if not isinstance(adapter, FileImportAdapter):
        raise ProblemError("internal-error")
    return adapter


def _make_sync_orchestrator(
    database: Database, registry: AdapterRegistry, store: CredentialStore | None
) -> RouterSyncOrchestrator:
    """Adapt the real :class:`SyncOrchestrator` to the router's started-run seam (API-R46a).

    The orchestrator's canonical opens flow through the ONE engine-owned ``SessionProvider``
    seam (SEAM-R11 / ARCH-R31) keyed on the server-derived athlete subject — never around it
    via a raw bound ``database.session`` method.
    """
    orchestrator = SyncOrchestrator(
        EngineSessionProvider(database), registry=registry, credential_store=store
    )

    async def run(target: SyncTarget) -> RouterSyncRun:
        result = await orchestrator.run(target.athlete_id, connection_id=target.connection_id)
        return started_run(result.sync_run_id)

    return run


async def _file_import_descriptor(session: AsyncSession) -> SourceDescriptor:
    """Resolve the seeded single ``file_import`` descriptor (LIN-R1.1); absence is internal.

    The descriptor is registration *data* seeded by the initial migration; its absence is
    an operator-configuration fault, surfaced fail-closed as a generic internal error
    (ERR-R5), never echoed to the client.
    """
    stmt = select(SourceDescriptor).where(SourceDescriptor.source_key == FILE_IMPORT_SOURCE_KEY)
    descriptor = (await session.execute(stmt)).scalar_one_or_none()
    if descriptor is None:
        raise ProblemError("internal-error")
    return descriptor


__all__ = ["IngestionSeams", "RouterSyncOrchestrator", "build_ingestion_seams"]
