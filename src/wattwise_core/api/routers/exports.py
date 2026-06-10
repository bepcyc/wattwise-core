"""Exports router — data-export jobs + the signed download (§8.15, API-R34 / API-R10).

The canonical ``/v1/exports`` group:

- ``POST /v1/exports`` (``export``) — create an export job -> ``202 ExportJob``. The OSS
  artifact is generated deterministically on demand from the stored parameters, so a
  created job is immediately ``ready``.
- ``GET /v1/exports`` (``read``) — cursor-paginated job list (PAGE-R1/R5).
- ``GET /v1/exports/{job_id}`` (``read``) — one job; when ``ready`` it carries the
  short-lived, SINGLE-USE, owner-bound signed ``download`` object (API-R34).
- ``GET /v1/exports/{job_id}/download`` — the artifact. Two authorization paths resolve
  to the one athlete: the default bearer GET, and the bearer-FREE signed-URL path whose
  signature encodes athlete + job + expiry + one-time nonce; an expired / reused /
  tampered / replayed URL -> ``403 invalid-signed-url`` (capability-URL hygiene). Not
  ready -> ``409 conflict``.

Job rows + nonces are OPERATIONAL state on the agent-state store (amended ARCH-R13).

Requirement IDs: API-R34, API-R10, API-R19, AUTH-R2 (documented exception), ARCH-R13,
PAGE-R1/R3/R5, ERR-R8 (``invalid-signed-url``), CFG-R1a (TTL is config-loaded).
"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter, Depends, Query, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security.utils import get_authorization_scheme_param
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.agent.ops_jobs import ExportJobRecord
from wattwise_core.api.auth import Principal, Scope, authenticate, require_scopes
from wattwise_core.api.deps import (
    AppSettings,
    CurrentPrincipal,
    PublicRateLimit,
    RateLimit,
    get_agent_state_session,
    get_database,
)
from wattwise_core.api.errors import ProblemError
from wattwise_core.api.exports_artifacts import (
    CONTENT_TYPES,
    build_artifact,
    sign_download,
    verify_download,
)
from wattwise_core.api.pagination import clamp_limit, decode_cursor, encode_cursor
from wattwise_core.api.problems import not_found
from wattwise_core.config import Settings
from wattwise_core.persistence import Database
from wattwise_core.seams import EngineSessionProvider

# NO router-level RateLimit: the download route must stay bearer-FREE on its signed
# path (API-R34), and the per-athlete limiter depends on the bearer principal. The
# bearer routes attach RateLimit individually; the download debits the public bucket.
router = APIRouter(prefix="/v1/exports", tags=["exports"])

StateSession = Annotated[AsyncSession, Depends(get_agent_state_session)]

ExportScope = Literal["activities", "analytics", "all"]
ExportFormat = Literal["csv", "json", "zip"]


class ExportJobRequest(BaseModel):
    """``POST /v1/exports`` body (API-R34): scope + format + optional date window."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    scope: ExportScope
    format: ExportFormat
    frm: _dt.date | None = None
    to: _dt.date | None = None

    def __init__(self, **data: object) -> None:  # accept the spec's "from" key
        if "from" in data:
            data["frm"] = data.pop("from")
        super().__init__(**data)


class DownloadOut(BaseModel):
    """The short-lived signed download handle on a ready job (API-R34)."""

    url: str
    expires_at: str


class ExportJobOut(BaseModel):
    """One export job on the wire (API-R34): parameters + status + signed download."""

    model_config = ConfigDict(extra="forbid")

    export_job_id: str = Field(json_schema_extra={"format": "uuid"})
    status: Literal["queued", "processing", "ready", "failed"]
    scope: ExportScope
    format: ExportFormat
    from_date: str | None = None
    to_date: str | None = None
    created_at: str
    download: DownloadOut | None = None


class ExportPage(BaseModel):
    """The PAGE-R4 page block of the export-job list."""

    limit: int
    next_cursor: str | None = None
    has_more: bool


class ExportJobList(BaseModel):
    """``GET /v1/exports``: the cursor-paginated export-job list (PAGE-R1/R4)."""

    data: list[ExportJobOut]
    page: ExportPage


def _signing_key(settings: Settings) -> str:
    """The engine signing key the signed download URL is minted with (fail-closed)."""
    key = settings.token_signing_key
    if key is None:
        raise ProblemError("internal-error")
    return str(key.get_secret_value())


def _job_out(row: ExportJobRecord, settings: Settings | None = None) -> ExportJobOut:
    """Project a job row onto the wire shape; mint the signed download when ready.

    The signed URL is minted fresh per read (short-lived by construction, API-R34); the
    single-use property lives on the row's one-time nonce, claimed atomically at the
    signed download itself.
    """
    download = None
    if settings is not None and row.status == "ready" and row.nonce_used_at is None:
        ttl = int(settings.exports__signed_url_ttl_seconds)
        exp = int(_dt.datetime.now(_dt.UTC).timestamp()) + ttl
        sig = sign_download(
            _signing_key(settings),
            athlete_id=str(row.athlete_id),
            job_id=str(row.export_job_id),
            exp=exp,
            nonce=row.nonce,
        )
        url = (
            f"/v1/exports/{row.export_job_id}/download"
            f"?exp={exp}&nonce={row.nonce}&sig={sig}"
        )
        download = DownloadOut(
            url=url,
            expires_at=_dt.datetime.fromtimestamp(exp, _dt.UTC).isoformat(),
        )
    return ExportJobOut(
        export_job_id=str(row.export_job_id),
        status=row.status,
        scope=row.scope,
        format=row.format,
        from_date=row.from_date,
        to_date=row.to_date,
        created_at=row.created_at.isoformat(),
        download=download,
    )


@router.post(
    "",
    response_model=ExportJobOut,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="createExport",
    dependencies=[RateLimit, Depends(require_scopes(Scope.EXPORT))],
)
async def create_export(
    body: ExportJobRequest,
    principal: CurrentPrincipal,
    session: StateSession,
    settings: AppSettings,
) -> ExportJobOut:
    """Create an export job -> ``202 ExportJob`` (API-R34, scope ``export``).

    The OSS artifact is rendered deterministically on demand from these stored
    parameters, so the job is immediately ``ready`` and already carries its signed
    ``download`` handle. Identity is server-derived (AUTH-R3); the one-time nonce that
    seeds the single-use signed URL is minted here.
    """
    row = ExportJobRecord(
        athlete_id=uuid.UUID(principal.athlete_id),
        scope=body.scope,
        format=body.format,
        from_date=body.frm.isoformat() if body.frm else None,
        to_date=body.to.isoformat() if body.to else None,
        status="ready",
        nonce=uuid.uuid4().hex,
    )
    session.add(row)
    await session.flush()
    return _job_out(row, settings)


@router.get(
    "",
    response_model=ExportJobList,
    operation_id="listExports",
    dependencies=[RateLimit, Depends(require_scopes(Scope.READ))],
)
async def list_exports(
    principal: CurrentPrincipal,
    session: StateSession,
    settings: AppSettings,
    limit: Annotated[int, Query(ge=1, json_schema_extra={"maximum": 200})] = 50,
    cursor: Annotated[str | None, Query()] = None,
) -> ExportJobList:
    """List the owner's export jobs, newest first, cursor-paginated (PAGE-R1/R5)."""
    bounded = clamp_limit(int(limit))
    stmt = (
        select(ExportJobRecord)
        .where(ExportJobRecord.athlete_id == uuid.UUID(principal.athlete_id))
        .order_by(ExportJobRecord.created_at.desc(), ExportJobRecord.export_job_id.desc())
        .limit(bounded + 1)
    )
    if cursor is not None:
        anchor, _item = decode_cursor(cursor, params={}, key=_signing_key(settings))
        stmt = stmt.where(ExportJobRecord.created_at < anchor)
    rows = (await session.execute(stmt)).scalars().all()
    has_more = len(rows) > bounded
    page_rows = rows[:bounded]
    nxt = None
    if has_more and page_rows:
        last = page_rows[-1]
        nxt = encode_cursor(
            last.created_at, str(last.export_job_id), params={}, key=_signing_key(settings)
        )
    return ExportJobList(
        data=[_job_out(r, settings) for r in page_rows],
        page=ExportPage(limit=bounded, next_cursor=nxt, has_more=has_more),
    )


async def _owned_job(
    session: AsyncSession, athlete_id: str, job_id: str
) -> ExportJobRecord | None:
    """The owner's job by id; a foreign / unknown / non-UUID id reads as absent."""
    try:
        job_uuid = uuid.UUID(job_id)
    except (ValueError, AttributeError):
        return None
    stmt = select(ExportJobRecord).where(
        ExportJobRecord.export_job_id == job_uuid,
        ExportJobRecord.athlete_id == uuid.UUID(athlete_id),
    )
    return (await session.execute(stmt)).scalar_one_or_none()


@router.get(
    "/{job_id}",
    response_model=ExportJobOut,
    operation_id="getExport",
    dependencies=[RateLimit, Depends(require_scopes(Scope.READ))],
)
async def get_export(
    job_id: str,
    principal: CurrentPrincipal,
    session: StateSession,
    settings: AppSettings,
) -> ExportJobOut:
    """One export job; a ready job carries its short-lived signed ``download`` (API-R34)."""
    row = await _owned_job(session, principal.athlete_id, job_id)
    if row is None:
        raise not_found()
    return _job_out(row, settings)


async def _claim_nonce(session: AsyncSession, row: ExportJobRecord) -> bool:
    """Atomically claim the job's one-time nonce (SINGLE-USE, API-R34); rowcount decides."""
    result = cast(
        "CursorResult[Any]",
        await session.execute(
            update(ExportJobRecord)
            .where(
                ExportJobRecord.export_job_id == row.export_job_id,
                ExportJobRecord.nonce_used_at.is_(None),
            )
            .values(nonce_used_at=_dt.datetime.now(_dt.UTC))
        ),
    )
    await session.commit()
    return result.rowcount == 1


def _bearer_principal(request: Request) -> Principal:
    """Resolve the bearer principal manually (the download's default auth path).

    The download route cannot DEPEND on ``authenticate`` (the signed-URL path is
    bearer-free by design, API-R34), so the bearer fallback authenticates explicitly
    through the SAME verifier — never a second auth implementation.
    """
    header = request.headers.get("Authorization", "")
    scheme, param = get_authorization_scheme_param(header)
    credentials = (
        HTTPAuthorizationCredentials(scheme=scheme, credentials=param)
        if scheme and param
        else None
    )
    return authenticate(request, credentials)


@router.get("/{job_id}/download", operation_id="downloadExport", dependencies=[PublicRateLimit])
async def download_export(
    request: Request,
    job_id: str,
    session: StateSession,
    settings: AppSettings,
    database: Annotated[Database, Depends(get_database)],
    exp: Annotated[int | None, Query()] = None,
    nonce: Annotated[str | None, Query()] = None,
    sig: Annotated[str | None, Query()] = None,
) -> Response:
    """The export artifact (API-R34): bearer GET, or the single-use signed URL.

    With ``exp``/``nonce``/``sig`` present this is the documented bearer-free signed
    path: the signature must bind the OWNING athlete + this job + an unexpired ``exp`` +
    the job's one-time nonce, and the nonce is claimed atomically — an expired / reused /
    tampered / replayed URL is ``403 invalid-signed-url`` with no leakage. Without them
    the normal bearer-authenticated GET applies (the first-party default). A job that is
    not ``ready`` -> ``409 conflict``. ``Content-Type`` follows the format and the body
    is served as an attachment.
    """
    signed = sig is not None or nonce is not None or exp is not None
    if signed:
        if sig is None or nonce is None or exp is None:
            raise ProblemError("invalid-signed-url")
        row = await _job_by_id(session, job_id)
        if row is None:
            raise ProblemError("invalid-signed-url")
        ok = verify_download(
            _signing_key(settings),
            athlete_id=str(row.athlete_id),
            job_id=str(row.export_job_id),
            exp=exp,
            nonce=nonce,
            sig=sig,
            now=_dt.datetime.now(_dt.UTC),
        )
        if not ok or nonce != row.nonce or not await _claim_nonce(session, row):
            raise ProblemError("invalid-signed-url")
    else:
        principal = _bearer_principal(request)
        if Scope.EXPORT.value not in principal.scopes:
            raise ProblemError("insufficient-scope")
        row = await _owned_job(session, principal.athlete_id, job_id)
        if row is None:
            raise not_found()
    if row.status != "ready":
        raise ProblemError("conflict")
    sessions = EngineSessionProvider(database)
    async with sessions.session(subject=str(row.athlete_id)) as canonical:
        artifact = await build_artifact(
            canonical,
            athlete_id=row.athlete_id,
            scope=row.scope,
            fmt=row.format,
            frm=row.from_date,
            to=row.to_date,
        )
    filename = f"wattwise-export.{row.format}"
    return Response(
        content=artifact,
        media_type=CONTENT_TYPES[row.format],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def _job_by_id(session: AsyncSession, job_id: str) -> ExportJobRecord | None:
    """The job row by id alone (the signed path binds the owner via the signature)."""
    try:
        job_uuid = uuid.UUID(job_id)
    except (ValueError, AttributeError):
        return None
    return (
        await session.execute(
            select(ExportJobRecord).where(ExportJobRecord.export_job_id == job_uuid)
        )
    ).scalar_one_or_none()


__all__ = ["router"]
