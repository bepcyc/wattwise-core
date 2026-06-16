"""Athlete safety-constraint capture seam (agent-state backed, no LLM required).

The focused sibling (QUAL-R9 size split) that owns the engine's CONSTRAINT capture methods — the
``POST``/``GET``/``DELETE`` ``/v1/user-settings/constraints`` surface and the constraint tier the
run path recalls (MEM-R6). A constraint is an agent-state memory item (MEM-R7 / GROUND-R14, ADR
0008 §5), never canonical master-data, so add/list/lift reach the dedicated agent-state store
through this mixin and never require a model — it is shared verbatim by the live
:class:`~wattwise_core.agent.engine.GraphAgentEngine` and the
:class:`~wattwise_core.agent.unconfigured.UnconfiguredAgentEngine`. Identity is server-derived
(AGT-SEC-R1); the seam never widens scope from a client argument.

Cited requirements: MEM-R3, MEM-R6, MEM-R7, GROUND-R14, AGT-SEC-R1, QUAL-R9.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from typing import Protocol

from wattwise_core.agent.memory import ConstraintSeverity, OssMemoryStore, RecalledItem
from wattwise_core.agent.state_db import AgentStateDatabase
from wattwise_core.persistence.types import utcnow


class _ConstraintSeam(Protocol):
    async def _agent_state_db(self) -> AgentStateDatabase: ...


class ConstraintCaptureMixin:
    """Add/list/lift the athlete's safety constraints in the agent-state store (MEM-R7)."""

    async def add_constraint(
        self: _ConstraintSeam,
        *,
        athlete_id: str,
        content: str,
        severity: ConstraintSeverity = ConstraintSeverity.SOFT,
        effective_until: _dt.datetime | None = None,
    ) -> RecalledItem:
        """Record an ACTIVE athlete-stated constraint into the agent-state store (MEM-R7, ADR 0008).

        The WRITE half of the capture surface (ADR 0008 §5): persists a CONSTRAINT-kind row
        (``status=ACTIVE``) in the athlete's own words, scoped to the server-derived owner (MEM-R3).
        ``inferred`` is ``False`` (the athlete's own explicit statement); ``severity`` selects veto
        (HARD) vs caution (SOFT) at the grounding gate. Needs no model. Returns the persisted row.
        """
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            store = OssMemoryStore(session)
            return await store.add_constraint(
                athlete_id=athlete_id,
                content=content,
                severity=severity,
                inferred=False,
                effective_until=effective_until,
            )

    async def list_active_constraints(
        self: _ConstraintSeam, *, athlete_id: str
    ) -> Sequence[RecalledItem]:
        """List the owner's ACTIVE, non-expired constraints (MEM-R6/-R7, HARD-first, no model).

        The READ half of the capture surface: the always-resident core tier as of now, owner-scoped
        (MEM-R3). Personalization context only, never a canonical number (MEM-R1).
        """
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            store = OssMemoryStore(session)
            return await store.fetch_active_constraints(athlete_id=athlete_id, now=utcnow())

    async def lift_constraint(
        self: _ConstraintSeam, *, athlete_id: str, memory_item_id: str
    ) -> bool:
        """Lift the owner's constraint by id; ``True`` iff one was lifted (MEM-R7, shared decision).

        Owner-scoped and fail-closed: a cross-athlete / unknown id lifts nothing and returns
        ``False`` (the router maps that to a 404). A LIFTED constraint stops gating. Needs no model.
        """
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            store = OssMemoryStore(session)
            return await store.lift_constraint(athlete_id=athlete_id, memory_item_id=memory_item_id)


__all__ = ["ConstraintCaptureMixin"]
