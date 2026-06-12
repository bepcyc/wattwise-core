"""Coach numeric-detail preference seam (agent-state backed, no LLM required)."""

from __future__ import annotations

from typing import Protocol

from wattwise_core.agent.memory import (
    COACH_NUMERIC_DETAIL_LEVEL_PREF_PREFIX,
    COACH_NUMERIC_DETAIL_LEVELS,
    OssMemoryStore,
    coach_numeric_detail_level_from_items,
)
from wattwise_core.agent.state_db import AgentStateDatabase


class _NumericDetailSeam(Protocol):
    async def _agent_state_db(self) -> AgentStateDatabase: ...


async def read_stored_numeric_detail_level(state_db: AgentStateDatabase, *, athlete_id: str) -> int:
    """Read the persisted coach numeric-detail default, else balanced ``3``."""
    async with state_db.session() as session:
        store = OssMemoryStore(session)
        items = await store.fetch_relevant(
            athlete_id=athlete_id, query=COACH_NUMERIC_DETAIL_LEVEL_PREF_PREFIX, limit=50
        )
    return coach_numeric_detail_level_from_items(items)


class NumericDetailPreferenceMixin:
    """Read/write the coach numeric-detail preference from the agent-state store."""

    async def resolve_default_numeric_detail_level(
        self: _NumericDetailSeam, *, athlete_id: str, requested: int | None
    ) -> int:
        if requested in COACH_NUMERIC_DETAIL_LEVELS:
            return int(requested)
        state_db = await self._agent_state_db()
        return await read_stored_numeric_detail_level(state_db, athlete_id=athlete_id)

    async def get_numeric_detail_level_preference(
        self: _NumericDetailSeam, *, athlete_id: str
    ) -> int:
        state_db = await self._agent_state_db()
        return await read_stored_numeric_detail_level(state_db, athlete_id=athlete_id)

    async def set_numeric_detail_level_preference(
        self: _NumericDetailSeam, *, athlete_id: str, value: int
    ) -> None:
        if value not in COACH_NUMERIC_DETAIL_LEVELS:
            return
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            store = OssMemoryStore(session)
            await store.upsert_preference(
                athlete_id=athlete_id,
                marker=COACH_NUMERIC_DETAIL_LEVEL_PREF_PREFIX,
                content=f"{COACH_NUMERIC_DETAIL_LEVEL_PREF_PREFIX}{value}",
            )


__all__ = [
    "NumericDetailPreferenceMixin",
    "read_stored_numeric_detail_level",
]
