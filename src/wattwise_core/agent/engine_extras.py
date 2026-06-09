"""The diagnosis + memory READ/ERASE engine methods, factored off the engine (QUAL-R9 size split).

The focused sibling of :mod:`wattwise_core.agent.engine` that owns the DETERMINISTIC, non-graph
engine surfaces the deployable :class:`~wattwise_core.agent.engine.GraphAgentEngine` exposes
alongside the graph-driven deliverables: the data-quality / coverage DIAGNOSIS (API-R15) and the
athlete-scoped memory list / get / delete / erase seam (MEM-R3/-R4). They are split out as a mixin
so the main engine module stays under the size ceiling while these cohesive, model-free surfaces
live in one place (mirroring :mod:`engine_readiness`).

Neither surface routes through the durable checkpointer or the LLM: diagnosis projects the
canonical analytics envelope deterministically (fail-closed, GROUND-R7), and the memory seam is a
scoped relational read/delete over the dedicated agent-state store. Both keep identity
SERVER-DERIVED (AGT-SEC-R1) — the ``athlete_id`` is the authenticated owner and is never widened
from a client argument.

Cited requirements: API-R15, MEM-R1, MEM-R3, MEM-R4, GROUND-R7, OUTCOME-R3/-R4/-R5, AGT-SEC-R1.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from wattwise_core.agent.contracts import ChatModel
from wattwise_core.agent.deliverables import Readiness, readiness_assessment
from wattwise_core.agent.diagnose_deliverable import AgentDiagnosis, diagnose_coverage
from wattwise_core.agent.engine_memory import (
    delete_memory,
    erase_memory,
    get_memory,
    list_memory,
)
from wattwise_core.agent.engine_readiness import (
    gather_readiness_inputs,
    readiness_narrator,
)
from wattwise_core.agent.engine_services import CoachBundle
from wattwise_core.agent.memory import RecalledItem
from wattwise_core.agent.state_db import AgentStateDatabase
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.persistence import Database


class _EngineSeams(Protocol):
    """The engine seams the diagnosis/readiness/memory methods read (supplied by the engine).

    The mixin is structural: it depends only on the canonical ``Database`` (read-only here), the
    injected ``ChatModel`` + loaded ``CoachBundle`` the readiness narration uses, and the
    lazily-built dedicated agent-state database the host engine already owns — no graph / checkpoint
    coupling.
    """

    _db: Database
    _model: ChatModel
    _coach: CoachBundle

    async def _agent_state_db(self) -> AgentStateDatabase: ...


class DeliverableEngineMixin:
    """Diagnosis (API-R15) + athlete-scoped memory seam (MEM-R3/-R4) for the engine.

    Mixed into :class:`~wattwise_core.agent.engine.GraphAgentEngine`; every method is
    DETERMINISTIC (no LLM, no graph) and keeps ``athlete_id`` server-derived (AGT-SEC-R1). The
    diagnosis is read-only over the canonical store; the memory methods read/erase the dedicated
    agent-state store, scoped strictly to the owner.
    """

    async def diagnose(
        self: _EngineSeams, *, athlete_id: str, locale: str = "en"
    ) -> AgentDiagnosis:
        """Narrate canonical data-quality / coverage for the athlete (API-R15, fail-closed).

        DETERMINISTIC: probes each canonical analytic input through the analytics service and
        projects the typed ``Computed``/``Unavailable`` envelope into per-input coverage lines
        (present/missing/stale) with NO model call and NO retrieval planner, so there is nothing to
        fabricate (GROUND-R7 / OUTCOME-R5). Degrades visibly when the athlete has no usable
        canonical coverage at all (OUTCOME-R3). Read-only; no agent-state pool opened. ``locale`` is
        accepted for the API copy boundary; the deliverable carries no athlete-facing numbers
        (VOICE-R7).
        """
        async with self._db.session() as session:
            return await diagnose_coverage(AnalyticsService(session), athlete_id)

    async def readiness(
        self: _EngineSeams,
        *,
        athlete_id: str,
        locale: str = "en",
        response_length: str = "standard",
    ) -> Readiness:
        """Build the readiness/form deliverable from canonical inputs (QA-EVAL-R2.4).

        Gathers the readiness inputs DETERMINISTICALLY (the fixed readiness JTBD does NOT route
        through the retrieval planner) then drives :func:`readiness_assessment` with the same
        model-backed narrator + canonical grounder the answers use; the delivered verdict is always
        the deterministic oracle's (canonical wins), numbers surface only as grounded citations.
        Readiness does NOT route through the durable checkpointer (a single deterministic
        assessment, not a resumable conversation), so no agent-state pool is opened here.
        """
        async with self._db.session() as session:
            svc = AnalyticsService(session)
            form, as_of, rmssd, baseline = await gather_readiness_inputs(svc, athlete_id)
            return await readiness_assessment(
                athlete_id,
                form=form,
                as_of=as_of,
                hrv_rmssd=rmssd,
                hrv_baseline=baseline,
                narrate=readiness_narrator(self._model),
                grounder=self._coach.grounder(self._model, svc),
                response_length=response_length,  # type: ignore[arg-type]
            )

    async def list_memory(
        self: _EngineSeams, *, athlete_id: str, limit: int = 50, offset: int = 0
    ) -> Sequence[RecalledItem]:
        """List the athlete's durable memory rows, newest first, paginated (MEM-R3/-R4).

        The read seam over the dedicated agent-state memory table, scoped STRICTLY to the
        server-derived owner ``athlete_id`` (MEM-R3 / AGT-SEC-R1) — another athlete's rows are never
        listed. Returns personalization context only, never a canonical number (MEM-R1).
        """
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            return await list_memory(session, athlete_id=athlete_id, limit=limit, offset=offset)

    async def get_memory(
        self: _EngineSeams, *, athlete_id: str, memory_item_id: str
    ) -> RecalledItem | None:
        """Fetch ONE memory row by id, scoped to the owner, else ``None`` (MEM-R3, fail-closed).

        Looks up by BOTH the id AND the server-derived ``athlete_id`` (AGT-SEC-R1): a foreign /
        unknown / non-UUID id returns ``None`` and is never disclosed (the router maps that to a
        404, indistinguishable from truly absent).
        """
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            return await get_memory(session, athlete_id=athlete_id, memory_item_id=memory_item_id)

    async def delete_memory(
        self: _EngineSeams, *, athlete_id: str, memory_item_id: str
    ) -> bool:
        """Delete ONE memory row by id, scoped to the owner; True iff erased (MEM-R3 erasure).

        Privacy MUST (PRIV-R8 / CKPT-R8): the guarded delete matches BOTH the id AND the
        server-derived ``athlete_id``, so a cross-athlete / unknown id erases nothing and returns
        ``False`` (router -> 404). The session commits the delete (or rolls back on error).
        """
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            return await delete_memory(
                session, athlete_id=athlete_id, memory_item_id=memory_item_id
            )

    async def erase_memory(self: _EngineSeams, *, athlete_id: str) -> int:
        """Erase ALL of the athlete's memory rows; returns the count (MEM-R3 erasure / PRIV-R8).

        The whole-athlete erasure scoped to the server-derived owner only, never widening to another
        identity. Returns how many rows were removed so the endpoint reports it.
        """
        state_db = await self._agent_state_db()
        async with state_db.session() as session:
            return await erase_memory(session, athlete_id=athlete_id)


__all__ = ["DeliverableEngineMixin"]
