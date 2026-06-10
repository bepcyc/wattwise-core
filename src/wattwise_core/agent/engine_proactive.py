"""The Insight + Briefing engine methods, factored off the engine (QUAL-R9 size split).

The focused sibling of :mod:`wattwise_core.agent.engine` that owns the engine surfaces for the
last two named COACH-R1 deliverables — the short single-topic INSIGHT (#4) and the proactive
one-screen BRIEFING (#5) — as a mixin (mirroring :mod:`engine_extras`), so the main engine
module stays under the size ceiling. Both drive the SAME compiled grounded graph as every
other deliverable (one grounding pipeline, COACH-R1) over a DURABLE per-conversation saver, so
the returned ``thread_id`` + stable-id observations support COACH-R8 follow-ups on a stored
insight/briefing. Identity stays server-derived (AGT-SEC-R1).

Cited requirements: COACH-R1 #4/#5, COACH-R8, GRAPH-R2.1, CKPT-R3, AGT-SEC-R1, GBO-R38, MED-2.
"""

from __future__ import annotations

from typing import Protocol

from wattwise_core.agent.briefing_deliverable import Briefing, Insight
from wattwise_core.agent.briefing_deliverable import briefing as _briefing
from wattwise_core.agent.briefing_deliverable import insight as _insight
from wattwise_core.agent.checkpoint import SqlAlchemyCheckpointSaver
from wattwise_core.agent.engine_graph import CompiledCoachGraph, conversation_id_for
from wattwise_core.agent.engine_services import CoachBundle
from wattwise_core.agent.goals import active_goals_for
from wattwise_core.agent.state_db import AgentStateDatabase
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.entitlement import Entitlements
from wattwise_core.seams import SessionProvider


class _ProactiveSeams(Protocol):
    """The engine seams the insight/briefing methods drive (supplied by the host engine).

    Structural (like ``engine_extras``): the canonical-store ``SessionProvider`` choke point
    (SEAM-R11), the loaded ``CoachBundle`` (its presentation policy), the dedicated agent-state
    database + per-conversation durable saver (ARCH-R13 / CKPT-R3), and the engine's compiled
    grounded graph builder — the SAME one every other deliverable runs on (COACH-R1).
    """

    _sessions: SessionProvider
    _coach: CoachBundle

    async def _agent_state_db(self) -> AgentStateDatabase: ...

    def _saver(
        self, state_db: AgentStateDatabase, *, athlete_id: str, conversation_id: str
    ) -> SqlAlchemyCheckpointSaver: ...

    def _graph(
        self,
        svc: AnalyticsService,
        saver: SqlAlchemyCheckpointSaver,
        *,
        allow_names: frozenset[str] = ...,
        entitlement: Entitlements | None = ...,
    ) -> CompiledCoachGraph: ...


class ProactiveDeliverableMixin:
    """Insight (COACH-R1 #4) + Briefing (COACH-R1 #5) engine surfaces.

    Mixed into :class:`~wattwise_core.agent.engine.GraphAgentEngine`. Both methods run the full
    grounded agent graph (never a side channel), keep ``athlete_id`` server-derived
    (AGT-SEC-R1), and accept a per-request ``entitlement`` override (MED-2).
    """

    async def insight(
        self: _ProactiveSeams,
        *,
        athlete_id: str,
        topic: str,
        locale: str = "en",
        thread_id: str | None = None,
        entitlement: Entitlements | None = None,
    ) -> Insight:
        """Build a short, single-topic grounded insight (COACH-R1 #4).

        Drives a ``user_turn`` run over ``topic`` through the full grounded graph (one
        grounding pipeline for every deliverable, COACH-R1) on a DURABLE thread, so the
        returned ``thread_id`` + stable-id observations let the athlete follow up on a stored
        insight (COACH-R8). A per-request ``entitlement`` (MED-2) governs the run's bounds
        when supplied.
        """
        state_db = await self._agent_state_db()
        conversation_id = conversation_id_for(athlete_id, thread_id)
        saver = self._saver(state_db, athlete_id=athlete_id, conversation_id=conversation_id)
        async with self._sessions.session(subject=athlete_id) as session:
            graph = self._graph(AnalyticsService(session), saver, entitlement=entitlement)
            return await _insight(
                graph,
                athlete_id,
                topic,
                locale=locale,
                presentation=self._coach.presentation,
                thread_id=thread_id,
                conversation_id=conversation_id,
            )

    async def briefing(
        self: _ProactiveSeams,
        *,
        athlete_id: str,
        briefing_screen: str,
        locale: str = "en",
        entitlement: Entitlements | None = None,
    ) -> Briefing:
        """Build the proactive one-screen briefing (COACH-R1 #5, GRAPH-R2.1).

        Drives a ``scheduled_briefing`` run — intent fixed deterministically by
        ``(trigger, briefing_screen)``, no request text and no intent model call (GRAPH-R2.1) —
        over a deterministic per-screen conversation id, scoped to the authenticated athlete
        and through the same cost-admission gate as a user turn. The athlete's ACTIVE canonical
        goals flow in so the heads-up is goal-aware (GBO-R38). Abstains visibly (``degraded``
        + caveat) when the canonical inputs are missing (OUTCOME-R3/-R4, GROUND-R7).
        """
        state_db = await self._agent_state_db()
        conversation_id = f"briefing:{briefing_screen}"
        saver = self._saver(state_db, athlete_id=athlete_id, conversation_id=conversation_id)
        async with self._sessions.session(subject=athlete_id) as session:
            graph = self._graph(AnalyticsService(session), saver, entitlement=entitlement)
            return await _briefing(
                graph,
                athlete_id,
                briefing_screen,
                locale=locale,
                presentation=self._coach.presentation,
                conversation_id=conversation_id,
                active_goals=await active_goals_for(session, athlete_id),
            )


__all__ = ["ProactiveDeliverableMixin"]
