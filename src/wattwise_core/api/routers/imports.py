"""Imports router — direct activity-file upload jobs (``POST /v1/imports``, API-R33).

The OSS direct-upload surface: an athlete uploads a recording file from a watch or
another app and it lands as a connectionless ``file_import`` candidate (LIN-R1.1) —
the same path the compliant Strava export uses (CLI-R14). A successful upload returns
``202 ImportJob`` and is processed into a canonical activity by the ingestion write
path.

Validation is layered and fail-closed (API-R33):

- ``multipart/form-data`` with a single ``file`` part; the body cap is the upload cap
  (32 MiB default), enforced as the bytes stream in (LIMIT-R5) → ``413
  payload-too-large`` when exceeded.
- An unsupported extension/format → ``415 unsupported-media-type`` BEFORE any parse.
- A structurally invalid (corrupt/empty/unparseable) file → ``422 import-rejected``
  with a machine ``errors[].code`` and jargon-free ``detail`` (ERR-R6/API-R33); the
  raw bytes are never echoed.

The decode→map→ingest pipeline is an injectable seam (:data:`import_processor`) the
app factory overrides with the registered file-upload adapter + ingest service, so
this router never imports a concrete source adapter (ARCH-R22 / ONB-R4). Acting
identity is server-derived (AUTH-R3); the upload carries no caller-identity field.

Requirement IDs: API-R33, AUTH-R3, AUTH-R11, ERR-R6, ERR-R8, LIMIT-R5, LIN-R1.1.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Final, Literal

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.agent.ops_jobs import ImportJobRecord
from wattwise_core.api.auth import Scope, require_scopes
from wattwise_core.api.deps import (
    AppSettings,
    CurrentPrincipal,
    RateLimit,
    get_agent_state_session,
)
from wattwise_core.api.errors import FieldError, ProblemError
from wattwise_core.api.pagination import clamp_limit, decode_cursor, encode_cursor
from wattwise_core.api.problems import not_found

router = APIRouter(prefix="/v1/imports", tags=["imports"], dependencies=[RateLimit])

#: A session on the agent-state store where the job bookkeeping rows live (ARCH-R13).
StateSession = Annotated[AsyncSession, Depends(get_agent_state_session)]


#: The activity-file extensions the OSS importer accepts (API-R33). A double
#: extension (``.fit.gz``) is matched as a whole suffix, not just the last segment.
ACCEPTED_EXTENSIONS: Final[tuple[str, ...]] = (".fit", ".fit.gz", ".gpx", ".tcx")

#: Read chunk size while streaming the upload to enforce the cap (LIMIT-R5).
_CHUNK_BYTES: Final = 1 << 20  # 1 MiB


# --------------------------------------------------------- import-processor seam


class ImportRejected(Exception):
    """The uploaded file could not be parsed into a canonical activity (API-R33).

    Raised by the processor seam for a corrupt/empty/structurally-invalid file. Carries
    a stable machine ``code`` (e.g. ``unreadable_file``) and a short jargon-free reason;
    NEVER the raw bytes. The router maps it to ``422 import-rejected``.
    """

    def __init__(self, *, code: str = "import_rejected", reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(reason)


#: The processor seam: given the acting athlete id + the verbatim upload bytes +
#: the (optional) filename, decode → map → ingest as a connectionless ``file_import``
#: candidate (connection_id NULL, LIN-R1.1) and return the resulting :class:`ImportJob`.
#: Raises :class:`ImportRejected` for an unparseable file. The app factory overrides
#: this with the registered file-upload adapter + ingest service so this router never
#: imports a named adapter (ARCH-R22 / ONB-R4); tests inject a fake.
ImportProcessor = Callable[[str, bytes, str | None], Awaitable["ImportJob"]]


async def _unconfigured_processor(athlete_id: str, data: bytes, filename: str | None) -> ImportJob:
    """Fail-closed default: reject every upload until the factory wires the processor."""
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


def import_processor() -> ImportProcessor:
    """Provide the import-processing seam; the app factory overrides it (API-R33)."""
    return _unconfigured_processor


ProcessorDep = Annotated[ImportProcessor, Depends(import_processor)]


# --------------------------------------------------------------------------- wire shapes


class ImportJob(BaseModel):
    """An accepted upload job (``202``, API-R33).

    ``status`` is the canonical import-job status; ``status_text`` is jargon-free
    athlete copy (API-R21). No source/provider name and no object-store handle appear
    (Principle A) — a file import is a connectionless ``file_import`` candidate.
    """

    import_job_id: str
    status: Literal["queued", "processing", "done", "failed"]
    filename: str | None
    received_at: datetime
    status_text: str


class ImportPage(BaseModel):
    """The PAGE-R4 page block of the import-job list."""

    limit: int
    next_cursor: str | None = None
    has_more: bool


class ImportJobList(BaseModel):
    """``GET /v1/imports``: the cursor-paginated upload-job list (API-R33, PAGE-R4)."""

    data: list[ImportJob]
    page: ImportPage


@dataclass(frozen=True, slots=True)
class _Upload:
    """A validated, fully-read upload: its bytes + (sanitized) original filename."""

    data: bytes
    filename: str | None


# --------------------------------------------------------------------------- route


@router.post(
    "",
    response_model=ImportJob,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="createImport",
    dependencies=[Depends(require_scopes(Scope.WRITE, Scope.SYNC))],
)
async def create_import(
    file: Annotated[UploadFile, File(description="The activity file to import.")],
    principal: CurrentPrincipal,
    settings: AppSettings,
    processor: ProcessorDep,
    session: StateSession,
) -> ImportJob:
    """Accept one activity-file upload and queue it for ingest (API-R33).

    Validates the extension (``415`` unsupported), streams the body under the upload
    cap (``413`` when exceeded), then hands the verbatim bytes to the processor seam.
    A structurally invalid file → ``422 import-rejected``. On accept the file lands as
    a connectionless ``file_import`` candidate (LIN-R1.1) and a ``202 ImportJob`` is
    returned. Identity is server-derived (AUTH-R3).
    """
    _require_accepted_extension(file.filename)
    upload = await _read_capped(file, _max_bytes(settings))
    job = await _process(processor, principal.athlete_id, upload)
    await _record_job(session, principal.athlete_id, job)
    return job


@router.get(
    "",
    response_model=ImportJobList,
    operation_id="listImports",
    dependencies=[Depends(require_scopes(Scope.READ))],
)
async def list_imports(
    principal: CurrentPrincipal,
    session: StateSession,
    settings: AppSettings,
    limit: Annotated[int, Query(json_schema_extra={"maximum": 200})] = 50,
    cursor: Annotated[str | None, Query()] = None,
) -> ImportJobList:
    """List the owner's upload jobs, newest first, cursor-paginated (API-R33, PAGE-R1).

    Reads the operational job rows (agent-state store, ARCH-R13) scoped to the
    server-derived athlete (AUTH-R3); ``limit`` is clamped/rejected per PAGE-R3.
    """
    bounded = clamp_limit(int(limit))
    stmt = (
        select(ImportJobRecord)
        .where(ImportJobRecord.athlete_id == uuid.UUID(principal.athlete_id))
        .order_by(ImportJobRecord.received_at.desc(), ImportJobRecord.import_job_id.desc())
        .limit(bounded + 1)
    )
    if cursor is not None:
        anchor, _item = decode_cursor(cursor, params={}, key=_cursor_key(settings))
        stmt = stmt.where(ImportJobRecord.received_at < anchor)
    rows = (await session.execute(stmt)).scalars().all()
    has_more = len(rows) > bounded
    page_rows = rows[:bounded]
    nxt = None
    if has_more and page_rows:
        last = page_rows[-1]
        nxt = encode_cursor(
            last.received_at, str(last.import_job_id), params={}, key=_cursor_key(settings)
        )
    return ImportJobList(
        data=[_job_of(row) for row in page_rows],
        page=ImportPage(limit=bounded, next_cursor=nxt, has_more=has_more),
    )


@router.get(
    "/{import_job_id}",
    response_model=ImportJob,
    operation_id="getImport",
    dependencies=[Depends(require_scopes(Scope.READ))],
)
async def get_import(
    import_job_id: str,
    principal: CurrentPrincipal,
    session: StateSession,
) -> ImportJob:
    """One upload job: typed ``status`` + athlete-native ``status_text`` (API-R33).

    Objects belong to the one athlete — an unknown, foreign, or malformed id reads as
    absent and returns ``404 not-found`` (API-R51), never another athlete's job.
    """
    row = (
        await session.execute(
            select(ImportJobRecord).where(
                ImportJobRecord.import_job_id == import_job_id,
                ImportJobRecord.athlete_id == uuid.UUID(principal.athlete_id),
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise not_found()
    return _job_of(row)


# --------------------------------------------------------------------------- helpers


def _max_bytes(settings: AppSettings) -> int:
    """The configured upload cap in bytes (LIMIT-R5; default 32 MiB)."""
    return int(settings.api__request_max_bytes)


def _require_accepted_extension(filename: str | None) -> None:
    """Reject an unsupported extension up front → ``415`` (API-R33).

    Matches a whole suffix so ``.fit.gz`` is accepted as one unit. A missing or
    unrecognized extension fails closed BEFORE any bytes are parsed.
    """
    lowered = (filename or "").lower()
    if not any(lowered.endswith(ext) for ext in ACCEPTED_EXTENSIONS):
        # The catalog ``title`` ("We can't read that kind of file") is the athlete copy
        # (QUAL-R13): user-facing text resolves through the externalized catalog, never
        # an inline literal here. The accepted formats are advertised by ``initiate``.
        raise ProblemError("unsupported-media-type")


async def _read_capped(file: UploadFile, max_bytes: int) -> _Upload:
    """Stream the upload into memory, failing closed at the cap → ``413`` (LIMIT-R5).

    The cap is enforced as bytes arrive (never by trusting a client-sent length), so a
    file larger than the cap is rejected without buffering it whole.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            # Catalog ``title`` ("That file is a little too big") is the athlete copy
            # (QUAL-R13); no inline user-facing literal at the raise site.
            raise ProblemError("payload-too-large")
        chunks.append(chunk)
    return _Upload(data=b"".join(chunks), filename=file.filename)


async def _process(processor: ImportProcessor, athlete_id: str, upload: _Upload) -> ImportJob:
    """Hand the verbatim bytes to the processor; map a rejection → ``422`` (API-R33).

    An empty upload is rejected before the processor (nothing to parse). A processor
    :class:`ImportRejected` becomes ``422 import-rejected`` with its machine code and a
    jargon-free reason; the raw bytes never leak into the problem document (ERR-R5).
    """
    if not upload.data:
        raise _rejected(code="empty_file", reason="That file was empty.")
    try:
        return await processor(athlete_id, upload.data, upload.filename)
    except ImportRejected as exc:
        raise _rejected(code=exc.code, reason=exc.reason) from exc


def _rejected(*, code: str, reason: str) -> ProblemError:
    """A ``422 import-rejected`` with a machine code + jargon-free reason (API-R33)."""
    return ProblemError(
        "import-rejected",
        detail=reason,
        errors=[FieldError(code=code, message=reason, parameter="file")],
    )


def _cursor_key(settings: AppSettings) -> str:
    """The engine signing key the list cursor is signed with (PAGE-R5, fail-closed)."""
    key = settings.token_signing_key
    if key is None:
        raise ProblemError("internal-error")
    return str(key.get_secret_value())


def _job_of(row: ImportJobRecord) -> ImportJob:
    """Project one operational job row onto the wire ``ImportJob`` shape (API-R33)."""
    return ImportJob(
        import_job_id=str(row.import_job_id),
        status=row.status,
        filename=row.filename,
        received_at=row.received_at,
        status_text=row.status_text,
    )


async def _record_job(session: AsyncSession, athlete_id: str, job: ImportJob) -> None:
    """Persist the accepted job's bookkeeping row on the agent-state store (ARCH-R13)."""
    session.add(
        ImportJobRecord(
            import_job_id=job.import_job_id,
            athlete_id=uuid.UUID(athlete_id),
            status=job.status,
            filename=job.filename,
            status_text=job.status_text,
            received_at=job.received_at,
        )
    )
    await session.flush()


def queued_job(import_job_id: str, filename: str | None) -> ImportJob:
    """Build a freshly-queued :class:`ImportJob` (the processor's accept return).

    A small constructor the processor seam reuses so the queued status + athlete copy
    are defined once here (API-R21), not at the wiring site.
    """
    return ImportJob(
        import_job_id=import_job_id,
        status="queued",
        filename=filename,
        received_at=datetime.now(UTC),
        status_text="We've got your file and we're bringing it in.",
    )


__all__ = [
    "ACCEPTED_EXTENSIONS",
    "ImportJob",
    "ImportJobList",
    "ImportProcessor",
    "ImportRejected",
    "import_processor",
    "queued_job",
    "router",
]
