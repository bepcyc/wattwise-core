"""Per-athlete data export — portability (PRIV-R9 / PRIV-R12(d) / API-R34).

``GET /v1/users/me/export`` streams the authenticated owner's canonical data in a
documented machine-readable format: **NDJSON** — one JSON object per line, each
``{"table": <canonical table name>, "row": {<column>: <value>}}``, UTF-8, ordered
parents-first by the schema's topological order. Every athlete-scoped canonical table is
included (direct ``athlete_id`` tables plus the transitive children — laps, stream sets,
channels, derived metrics — scoped through the same parent-hop map the PRIV-R8 erasure
uses, so export and erasure cover the SAME data inventory by construction). Shared
registry tables (sports, source descriptors) carry no personal data and are excluded.

Identity is the SERVER-DERIVED owner (AUTH-R3); the route requires the ``export`` scope
(AUTH-R7). The response is STREAMED row-by-row (PERF-R10(c): a full-history export never
buffers the whole dataset). Each export is an explicit, authorized, audited operation
(PRIV-R6): it is recorded on the tamper-evident audit stream (LOG-R6.2).
"""

from __future__ import annotations

import datetime as _dt
import enum
import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import Table, select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.api.auth import Principal, Scope, require_scopes
from wattwise_core.api.deps import get_db
from wattwise_core.observability.audit import audit_event
from wattwise_core.persistence.base import Base
from wattwise_core.privacy.erasure import (
    _CANONICAL_SHARED_TABLES,
    _CANONICAL_TRANSITIVE,
    _owned_id_subquery,
)

#: The documented export media type: newline-delimited JSON (one row object per line).
EXPORT_MEDIA_TYPE = "application/x-ndjson"

router = APIRouter(prefix="/v1/users/me", tags=["users"])


def _jsonable(value: Any) -> Any:
    """Render one column value into its documented JSON form (PRIV-R9)."""
    if isinstance(value, (uuid.UUID, _dt.datetime, _dt.date)):
        return str(value) if isinstance(value, uuid.UUID) else value.isoformat()
    if isinstance(value, enum.Enum):
        return value.value
    return value


def _export_tables() -> list[Table]:
    """Every athlete-scoped canonical table, parents-first (the schema's topo order)."""
    return [
        table
        for table in Base.metadata.sorted_tables
        if table.name not in _CANONICAL_SHARED_TABLES
        and ("athlete_id" in table.columns or table.name in _CANONICAL_TRANSITIVE)
    ]


async def _rows_of(
    session: AsyncSession, table: Table, athlete_id: uuid.UUID
) -> AsyncIterator[dict[str, Any]]:
    """Stream the athlete's rows of one table (direct or transitive scope)."""
    if "athlete_id" in table.columns:
        stmt = select(table).where(table.c["athlete_id"] == athlete_id)
        result = await session.stream(stmt)
        async for row in result.mappings():
            yield dict(row)
        return
    spec = _CANONICAL_TRANSITIVE[table.name]
    for child_column, hops in spec.paths:
        owned = _owned_id_subquery(hops, athlete_id, dict(Base.metadata.tables))
        result = await session.stream(select(table).where(table.c[child_column].in_(owned)))
        async for row in result.mappings():
            yield dict(row)


@router.get(
    "/export",
    operation_id="exportMyData",
    dependencies=[Depends(require_scopes(Scope.EXPORT))],
)
async def export_my_data(
    principal: Annotated[Principal, Depends(require_scopes(Scope.EXPORT))],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> StreamingResponse:
    """Stream the owner's canonical data as NDJSON (PRIV-R9 portability)."""
    athlete_id = uuid.UUID(principal.athlete_id)
    audit_event("data_export_started", athlete_id=principal.athlete_id)

    async def _body() -> AsyncIterator[bytes]:
        for table in _export_tables():
            async for row in _rows_of(session, table, athlete_id):
                payload = {
                    "table": table.name,
                    "row": {k: _jsonable(v) for k, v in row.items()},
                }
                yield (json.dumps(payload, default=str) + "\n").encode()

    return StreamingResponse(_body(), media_type=EXPORT_MEDIA_TYPE)


__all__ = ["EXPORT_MEDIA_TYPE", "router"]
