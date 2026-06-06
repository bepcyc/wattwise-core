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

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Final, Literal

from fastapi import APIRouter, Depends, File, UploadFile, status
from pydantic import BaseModel

from wattwise_core.api.auth import Scope, require_scopes
from wattwise_core.api.deps import AppSettings, CurrentPrincipal
from wattwise_core.api.errors import FieldError, ProblemError

router = APIRouter(prefix="/v1/imports", tags=["imports"])


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
    return await _process(processor, principal.athlete_id, upload)


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
    "ImportProcessor",
    "ImportRejected",
    "import_processor",
    "queued_job",
    "router",
]
