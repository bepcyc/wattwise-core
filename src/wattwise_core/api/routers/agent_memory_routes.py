"""Agent durable-memory routes — per-item view + erase (API-R15a / MEM-R3).

The ``/v1/agent/memory*`` slice of the agent surface, split out of
:mod:`agent_breadth` to keep both modules inside the QUAL-R9 size ceiling. The router
is mounted onto the ``/v1/agent`` router exactly like the breadth router, and reuses
the breadth module's seam aliases (the SAME engine seam, identity seam, and limiter),
so the app factory's overrides wire this module with no extra seams.

Requirement IDs: API-R15a (view + per-item erase), MEM-R3 (erasure MUST), AUTH-R3,
AUTH-R13 (agent scope, request-rate limits only — non-LLM, no cost budget), API-R51
(foreign/unknown id reads as absent), PRIV-R8 (residual-row removal), LIMIT-R1.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query, status

from wattwise_core.api.problems import not_found
from wattwise_core.api.ratelimit import LimitClass
from wattwise_core.api.routers.agent_breadth import AthleteId, Engine, Limiter, _Agent
from wattwise_core.api.routers.agent_schemas import (
    MemoryItemList,
    MemoryItemOut,
    memory_item_out,
)

# No prefix: mounted ONTO the ``/v1/agent`` router (which prepends that prefix).
router = APIRouter(tags=["agent"])


@router.get(
    "/memory",
    response_model=MemoryItemList,
    dependencies=[_Agent],
    operation_id="agentMemoryList",
)
async def agent_memory_list(
    athlete_id: AthleteId,
    engine: Engine,
    limiter: Limiter,
    *,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> MemoryItemList:
    """List the owner's durable memory rows, newest first (API-R15a / MEM-R3).

    Requires the ``agent`` scope (AUTH-R13). NON-LLM and OUTSIDE the agent cost gate (a memory read
    debits no coaching budget). Scoped STRICTLY to the server-derived owner (AUTH-R3 / MEM-R3) —
    another athlete's rows are never listed. Returns personalization context only, never a canonical
    number (MEM-R1). ``limit`` is bounded ``[1, 200]`` and ``offset`` pages the newest-first list.
    Request-rate-limited like every endpoint (LIMIT-R1) — non-LLM, so it reserves NO cost budget.
    """
    limiter.check(athlete_id, LimitClass.AGENT)
    rows = await engine.list_memory(athlete_id=athlete_id, limit=limit, offset=offset)
    return MemoryItemList(data=[memory_item_out(r) for r in rows])


@router.get(
    "/memory/{memory_item_id}",
    response_model=MemoryItemOut,
    dependencies=[_Agent],
    operation_id="agentMemoryGet",
)
async def agent_memory_get(
    athlete_id: AthleteId,
    engine: Engine,
    limiter: Limiter,
    memory_item_id: Annotated[str, Path()],
) -> MemoryItemOut:
    """Fetch ONE durable memory row by id, scoped to the owner (API-R15a / MEM-R3, fail-closed).

    Requires the ``agent`` scope (AUTH-R13). NON-LLM / outside the cost gate. Looks up by BOTH the
    id AND the server-derived ``athlete_id`` (AUTH-R3): a foreign / unknown / non-UUID id is
    ``404`` ``not-found`` (API-R51), indistinguishable from truly absent and never disclosed
    (MEM-R3). Returns personalization context only, never a canonical number (MEM-R1).
    Request-rate-limited like every endpoint (LIMIT-R1).
    """
    limiter.check(athlete_id, LimitClass.AGENT)
    item = await engine.get_memory(athlete_id=athlete_id, memory_item_id=memory_item_id)
    if item is None:
        raise not_found()
    return memory_item_out(item)


@router.delete(
    "/memory/{memory_item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[_Agent],
    operation_id="agentMemoryErase",
)
async def agent_memory_erase(
    athlete_id: AthleteId,
    engine: Engine,
    limiter: Limiter,
    memory_item_id: Annotated[str, Path()],
) -> None:
    """Erase ONE durable memory row by id, scoped to the owner (API-R15a / MEM-R3 MUST / PRIV-R8).

    Requires the ``agent`` scope (AUTH-R13). NON-LLM / outside the cost gate. The guarded delete
    matches BOTH the id AND the server-derived ``athlete_id`` (AUTH-R3): a cross-athlete / unknown /
    non-UUID id erases nothing and is ``404`` ``not-found`` (API-R51), never disclosed. A successful
    erase removes the residual row entirely (PRIV-R8) so a re-GET of the id is ``404``. Per-item
    erase is a privacy MUST (MEM-R3). Returns ``204`` with no body (API-R15a).
    Request-rate-limited like every endpoint (LIMIT-R1).
    """
    limiter.check(athlete_id, LimitClass.AGENT)
    erased = await engine.delete_memory(athlete_id=athlete_id, memory_item_id=memory_item_id)
    if not erased:
        raise not_found()


__all__ = ["router"]
