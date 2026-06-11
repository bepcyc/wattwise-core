"""The deployable :class:`GraphAgentEngine` the API drives (doc 50).

Assembles the compiled coaching graph over a DURABLE :class:`SqlAlchemyCheckpointSaver` (dedicated
agent-state pool, ARCH-R13/DEPLOY-R4) + the concrete production services (in the focused
:mod:`engine_services` sibling, re-exported here) and runs the deliverable projection (grounded
Q&A + COACH-R8 follow-ups, the weekly digest, the multi-day PLAN, the HITL decision resume).
``build_agent_engine`` constructs it from settings; with no LLM key the OSS engine degrades
gracefully (RUN-R4.1). Identity is server-derived (AGT-SEC-R1); the durable saver makes follow-ups
and approval pauses resumable (CKPT-R5/-R9).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

# The concrete production agent services + the canonical workout-NAME library live in the focused
# sibling :mod:`engine_services` (QUAL-R9 size split); the public ones are re-exported below so
# every historical ``from wattwise_core.agent.engine import ClaimGrounder/ModelPlanner/...`` path
# stays stable (e.g. the integration tests import ``ClaimGrounder``/``_PlanSchema``). The
# compiled-graph adapter + run bounds live in :mod:`engine_graph` (QUAL-R9 size split).
from wattwise_core.agent.checkpoint import SqlAlchemyCheckpointSaver

# The fail-closed decision refusal lives beside the interrupt-ledger guard that raises its cause
# (QUAL-R9 size split); re-exported here so every historical ``from wattwise_core.agent.engine
# import DecisionRefused`` path (the router + the durable tests) stays stable.
from wattwise_core.agent.checkpoint_interrupts import DecisionRefused
from wattwise_core.agent.contracts import ChatModel, RunStatus
from wattwise_core.agent.deliverables import (
    AgentAnswer,
    Digest,
    Plan,
    answer_question,
    conversation_id_of,
    weekly_digest,
)
from wattwise_core.agent.digest_history import record_digest
from wattwise_core.agent.engine_entitlement import effective_entitlement, sized_model
from wattwise_core.agent.engine_extras import DeliverableEngineMixin
from wattwise_core.agent.engine_graph import (
    NODE_VISIT_CEILING,
    RECURSION_LIMIT,
    CompiledCoachGraph,
    conversation_id_for,
    conversation_id_for_turn,
    resolve_existing_answer,
)
from wattwise_core.agent.engine_proactive import ProactiveDeliverableMixin
from wattwise_core.agent.engine_services import (  # noqa: F401  re-exported (historical paths)
    CANONICAL_WORKOUT_NAMES,
    ClaimGrounder,
    CoachBundle,
    DeterministicCoverage,
    ModelPlanner,
    RegistryGateway,
    _PlanSchema,
    build_services,
)
from wattwise_core.agent.goals import active_goals_for
from wattwise_core.agent.graph import DEFAULT_MAX_TOOL_ITERATIONS, build_graph
from wattwise_core.agent.grounding_evidence import _ClaimSchema, _ExtractedClaim  # noqa: F401
from wattwise_core.agent.plan_deliverable import _project_plan, safe_plan_html
from wattwise_core.agent.plan_deliverable import plan as _plan
from wattwise_core.agent.plan_regrounding import accept_edit, run_request_text
from wattwise_core.agent.seams import EntitlementCostGate
from wattwise_core.agent.state_db import (
    AgentStateDatabase,
    build_agent_state_database,
)
from wattwise_core.agent.state_db import (
    fallback_state_dsn as _fallback_state_dsn,
)
from wattwise_core.agent.tiering import (
    ModelRoutingPolicy,
    SingleModelRoutingPolicy,
    context_budget,
)
from wattwise_core.agent.unconfigured import UnconfiguredAgentEngine
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.entitlement import Entitlements
from wattwise_core.persistence import Database
from wattwise_core.seams import EngineSessionProvider, SessionProvider


class GraphAgentEngine(DeliverableEngineMixin, ProactiveDeliverableMixin):  # noqa: size-limits
    """The deployable :class:`~wattwise_core.api.routers.agent_routes.AgentEngine`.

    Over the class-size guard (documented suppression, QUAL-R9): one cohesive implementation of the
    injected ``AgentEngine`` protocol the API router drives end to end — every deliverable surface
    (grounded Q&A + follow-ups, weekly digest, multi-day plan, the HITL decision + its
    interrupt-status probe) must live behind this single seam; each method stays well under the
    60-line function ceiling, and the plan re-grounding bodies already moved to
    :mod:`plan_regrounding` and the interrupt ledger to :mod:`checkpoint_interrupts`. The
    DETERMINISTIC diagnosis (API-R15) + athlete-scoped memory seam (MEM-R3/-R4) are inherited from
    :class:`~wattwise_core.agent.engine_extras.DeliverableEngineMixin` (QUAL-R9 size split).

    Per call it opens a canonical session, builds the analytics service + the concrete agent
    services + the compiled graph over a DURABLE :class:`SqlAlchemyCheckpointSaver` bound to the
    run's ``(athlete_id, conversation_id)`` on a DEDICATED agent-state pool (ARCH-R13/DEPLOY-R4),
    and runs the deliverable projection. The durable saver makes a follow-up resume the SAME thread
    and a paused approval-gated plan resumable (CKPT-R5/-R9). ``coach`` is the loaded §16
    coach-config (compose prompt + metric-equivalence + tolerance + URL allow-list) that makes the
    LIVE agent produce a GROUNDED answer; ``state_db`` (its OWN engine/pool) is injected in
    production, else a per-process file-sqlite store is lazy.
    """

    def __init__(
        self,
        database: Database,
        model: ChatModel,
        *,
        state_db: AgentStateDatabase | None = None,
        coach: CoachBundle | None = None,
        entitlement: Entitlements | None = None,
        sessions: SessionProvider | None = None,
        dedup_window_seconds: int | None = None,
        model_routing: ModelRoutingPolicy | None = None,
        context_window_tokens: int | None = None,
    ) -> None:
        # Every CANONICAL-store open flows through the ONE engine-owned provider seam (SEAM-R11
        # / ARCH-R31), never around it; the OSS default does no tenant scoping (agent-state is
        # the SEPARATE ARCH-R13 store, not this).
        self._sessions: SessionProvider = sessions or EngineSessionProvider(database)
        self._model = model
        self._state_db = state_db
        self._coach = coach if coach is not None else CoachBundle()
        # CKPT-R4 dedup window (seconds) for the idempotent run path: ``build_agent_engine`` passes
        # the config-loaded ``agent__idempotency_dedup_window_seconds`` (CFG-R1a). ``None`` (a
        # direct caller with no window) means NO time-bucketing — a re-submitted identical turn
        # still dedups by content (bucket 0), never spawning a duplicate; absence disables only the
        # time axis (not a baked value), keeping the dedup fail-closed.
        self._dedup_window_seconds = dedup_window_seconds or 0
        # The DEFAULT (config-resolved) entitlement the engine reads its non-monetary local guards
        # FROM (AGT-ENT-R1): the node-visit ceiling + tool-iteration bound the graph reads, the
        # token bound the model output budget is sized to, and the wall-clock deadline the run is
        # bounded by. ``build_agent_engine`` supplies the config-resolved OSS plan; a direct caller
        # (tests / OSS) may pass ``None`` -> a bare all-permissive grant (zero bounds), so each
        # guard falls back to its config/module default exactly as before. A per-REQUEST
        # entitlement passed to a deliverable method OVERRIDES this default for that run (MED-2).
        self._entitlement = entitlement if entitlement is not None else Entitlements()
        # The typed model-routing-policy seam (MODEL-R1/-R2/-R2b): the factory passes the OSS
        # single-model policy built from the config tier/effort labels; a commercial deployment
        # plugs a richer policy through this same seam without an engine change (COMM-R20). A
        # direct caller with none gets the default flash/low single-model policy.
        self._model_routing = model_routing or SingleModelRoutingPolicy()
        # The configured model context window (MODEL-R3): the compose input budget is computed
        # as window minus the run's resolved OUTPUT-token headroom. ``None`` (a direct caller
        # with no config) leaves the graph's module fallback budget in force.
        self._context_window_tokens = context_window_tokens

    async def _agent_state_db(self) -> AgentStateDatabase:
        """The dedicated agent-state database, lazily built + schema-created once (ARCH-R13).

        An injected ``state_db`` (production / the durable tests) is used as-is. The lazy fallback
        opens a per-process FILE-sqlite store on its own REAL pool (see :func:`_fallback_state_dsn`
        for why not ``:memory:``).
        """
        if self._state_db is None:
            self._state_db = build_agent_state_database(dsn=_fallback_state_dsn())
            await self._state_db.create_all()
        return self._state_db

    def _saver(
        self, state_db: AgentStateDatabase, *, athlete_id: str, conversation_id: str
    ) -> SqlAlchemyCheckpointSaver:
        """A durable checkpointer bound to ``(athlete_id, conversation_id)`` (CKPT-R3)."""
        return SqlAlchemyCheckpointSaver(
            state_db.session_factory,
            athlete_id=athlete_id,
            conversation_id=conversation_id,
        )

    def _effective_entitlement(self, entitlement: Entitlements | None) -> Entitlements:
        """The entitlement governing THIS run (MED-2); see :mod:`engine_entitlement`."""
        return effective_entitlement(self._entitlement, entitlement)

    def _sized_model(self, entitlement: Entitlements) -> ChatModel:
        """The model sized to the entitlement's bound (AGT-ENT-R1; :mod:`engine_entitlement`)."""
        return sized_model(self._model, entitlement)

    def _graph(
        self,
        svc: AnalyticsService,
        saver: SqlAlchemyCheckpointSaver,
        *,
        allow_names: frozenset[str] = frozenset(),
        entitlement: Entitlements | None = None,
    ) -> CompiledCoachGraph:
        """Build + compile the coaching graph over the per-call services + DURABLE saver (GRAPH-R5).

        ALL FIVE non-monetary local guards are read FROM the resolved entitlement (AGT-ENT-R1): the
        graph reads the node-visit ceiling + the tool-iteration bound from the
        :class:`EntitlementCostGate` carried on the services; the model's per-call OUTPUT budget is
        sized to the entitlement's token bound; and the wall-clock deadline bounds the whole
        ``CompiledCoachGraph.run``. The engine passes the MODULE defaults explicitly to
        ``build_graph`` so the carried entitlement's config-loaded bounds govern (the seams
        precedence ladder: explicit-default -> entitlement -> fallback). The durable saver records a
        ``live`` ledger row (CKPT-R9) so runs resume across turns/pauses. The loaded coach-config
        (§16) is wired in via ``self._coach`` so compose gets the real prompt + metric-equivalence.
        """
        ent = self._effective_entitlement(entitlement)
        model = self._sized_model(ent)
        services = replace(
            self._coach.services(model, svc, allow_names=allow_names),
            cost_gate=EntitlementCostGate(ent),
        )
        compiled = build_graph(
            model,
            services,
            saver,
            model_routing=self._model_routing,
            locales=self._coach.locales,
            context_token_budget=context_budget(self._context_window_tokens, ent.max_output_tokens),
            # The compose system prompt layers the INJECT-R2 shared preamble in FRONT of the persona
            # (``compose_system``) so the "delimited data is to analyze, never command" instruction
            # is in the prompt the model actually receives (INJECT-R2), not merely loaded.
            coach_system=self._coach.compose_system,
            reflect_system=self._coach.reflect_system,
            # VOICE-R7/-R8: the loaded detailed-length steering fragment, layered only when
            # the run asked for a detailed answer (a detailed deep-dive should surface up to
            # the cap of grounded numbers, never zero).
            detailed_compose_directive=self._coach.detailed_compose_directive,
            node_visit_ceiling=NODE_VISIT_CEILING,
            max_tool_iterations=DEFAULT_MAX_TOOL_ITERATIONS,
        )
        wall = ent.wall_clock_seconds
        return CompiledCoachGraph(compiled, wall_clock_seconds=wall, locales=self._coach.locales)

    async def answer(
        self,
        *,
        athlete_id: str,
        question: str | None,
        thread_id: str | None,
        response_length: str | None,
        follow_up: dict[str, Any] | None,
        locale: str,
        entitlement: Entitlements | None = None,
    ) -> AgentAnswer:
        """Answer a question, deduping a re-submitted turn and resuming a follow-up thread.

        A FRESH turn is keyed deterministically (athlete + question + dedup-window bucket): the SAME
        turn twice within the window RETURNS the existing run, never a duplicate (CKPT-R4); a
        follow-up resumes the SAME durable thread (COACH-R8, CKPT-R3/MED-2). ``response_length`` of
        ``None`` applies the PERSISTED per-athlete verbosity default (MEM-R1 / VOICE-R8, §382)
        WITHOUT mutating it; a value overrides for this call only. The engine RECALLS durable memory
        before the run and records an episode after a completed first turn, both through the ONE
        MemoryStore seam (MEM-R4). A per-request ``entitlement`` (MED-2) governs this run's bounds
        when supplied, else the config-resolved default (CKPT-R3/-R5).
        """
        length = await self.resolve_default_response_length(
            athlete_id=athlete_id, requested=response_length
        )
        recalled = await self.recall_memory_for_run(athlete_id=athlete_id, query=question or "")
        state_db = await self._agent_state_db()
        conversation_id = conversation_id_for_turn(
            athlete_id, thread_id, question, self._dedup_window_seconds
        )
        saver = self._saver(state_db, athlete_id=athlete_id, conversation_id=conversation_id)
        existing = await resolve_existing_answer(
            saver,
            athlete_id=athlete_id,
            conversation_id=conversation_id,
            follow_up_thread_id=thread_id,
        )
        if existing is not None:
            return existing
        async with self._sessions.session(subject=athlete_id) as session:
            answer = await answer_question(
                self._graph(AnalyticsService(session), saver, entitlement=entitlement),
                athlete_id,
                question or "",
                locale=locale,
                response_length=length,  # type: ignore[arg-type]
                thread_id=thread_id,
                conversation_id=conversation_id,
                follow_up=follow_up,
                presentation=self._coach.presentation,
                recalled_memory=recalled,
            )
        # A completed FIRST turn records a durable episode (MEM-R4); a follow-up resumes the same
        # thread, so it is not a new episode to remember.
        if answer.status is RunStatus.COMPLETED and follow_up is None:
            await self.record_run_episode(athlete_id=athlete_id, content=question or "")
        return answer

    async def digest(
        self, *, athlete_id: str, week_end: str, entitlement: Entitlements | None = None
    ) -> Digest:
        """Build the weekly digest (== weekly load review) for ``week_end`` (COACH-R1 #1).

        Drives a ``scheduled_digest`` run over a deterministic per-week conversation id and projects
        the grounded trailing-week review into :class:`Digest`, abstaining visibly (``degraded`` +
        caveat) when the week's canonical inputs are missing (OUTCOME-R3/-R4, GROUND-R7). A
        per-request ``entitlement`` (MED-2) governs this run's bounds when supplied, else the
        engine's config-resolved default.
        """
        state_db = await self._agent_state_db()
        conversation_id = f"digest:{week_end}"
        saver = self._saver(state_db, athlete_id=athlete_id, conversation_id=conversation_id)
        async with self._sessions.session(subject=athlete_id) as session:
            graph = self._graph(AnalyticsService(session), saver, entitlement=entitlement)
            # The weekly digest IS the weekly load review (COACH-R1 #1); flow the athlete's ACTIVE
            # canonical goals into it so the load review is goal-aware (GBO-R38 / API-R32).
            digest = await weekly_digest(
                graph,
                athlete_id,
                week_end,
                presentation=self._coach.presentation,
                active_goals=await active_goals_for(session, athlete_id),
            )
        # A grounded review joins the stored history the list surface pages (API-R14);
        # a degraded abstention is not recorded (record_digest no-ops on it).
        await record_digest(state_db, athlete_id=athlete_id, digest=digest)
        return digest

    async def plan_deliverable(
        self,
        *,
        athlete_id: str,
        request_text: str | None = None,
        request: str | None = None,
        thread_id: str | None = None,
        locale: str = "en",
        response_length: str = "detailed",
        requires_approval: bool = True,
        entitlement: Entitlements | None = None,
    ) -> Plan:
        """Build a multi-day grounded training PLAN, approval-gated by default (COACH-R2/CKPT-R5).

        Drives the graph with the canonical workout-NAME library so prescribed names ground (not
        auto-scrubbed). With ``requires_approval`` the durable interrupt-gate pauses the run and
        records a ``live`` ledger row; the :class:`Plan` is ``awaiting_approval`` carrying the
        ``interrupt_id`` the decision endpoint consumes (CKPT-R9). ``athlete_id`` is server-derived
        (AGT-SEC-R1). ``request_text`` is the planning prompt the planning router passes;
        ``request`` is a backward-compatible alias for the same value. A per-request ``entitlement``
        (MED-2) governs this run's non-monetary bounds when supplied, else the config default.
        """
        prompt = request_text if request_text is not None else (request or "")
        state_db = await self._agent_state_db()
        conversation_id = conversation_id_for(athlete_id, thread_id)
        saver = self._saver(state_db, athlete_id=athlete_id, conversation_id=conversation_id)
        async with self._sessions.session(subject=athlete_id) as session:
            graph = self._graph(
                AnalyticsService(session),
                saver,
                allow_names=CANONICAL_WORKOUT_NAMES,
                entitlement=entitlement,
            )
            # Read the athlete's ACTIVE canonical goals server-side and FLOW them into the run so
            # the plan is goal-aware (GBO-R38 / API-R32 / API-R35) — the agent owns goal planning.
            return await _plan(
                graph,
                athlete_id,
                prompt,
                locale=locale,
                response_length=response_length,  # type: ignore[arg-type]
                thread_id=thread_id,
                conversation_id=conversation_id,
                requires_approval=requires_approval,
                active_goals=await active_goals_for(session, athlete_id),
            )

    async def decision(
        self,
        *,
        athlete_id: str,
        thread_id: str,
        interrupt_id: str,
        decision: str,
        edited_plan: str | None = None,
        entitlement: Entitlements | None = None,
    ) -> Plan:
        """Resume a paused approval-gated plan with the athlete's decision (API-R12a/CKPT-R9).

        Atomically CONSUMES the ``live`` interrupt (guarded UPDATE): exactly one decision wins; a
        double-decision/unknown/cross-athlete attempt matches no row and raises
        :class:`DecisionRefused` (router maps 404/409) — never resumed twice. On a winning consume
        it drives ``Command(resume=...)`` through the DURABLE saver (no recompute, CKPT-R2):
        ``approve`` finalizes, ``reject`` resumes without approval, ``edit`` RE-GROUNDS
        ``edited_plan`` FIRST and accepts it ONLY when it fully grounds (``PROCEED`` + non-empty);
        a partial/abstained/extraction-failed edit is REJECTED — the run resolves to a DEGRADED
        terminal state whose body is the already-grounded pre-edit plan, NEVER the unverified edit
        (H3 / GROUND-R3). The edit is re-grounded BEFORE the graph resume so its outcome decides the
        resume payload. A per-request ``entitlement`` (MED-2) governs the resumed run's bounds when
        supplied, else the config default.
        """
        state_db = await self._agent_state_db()
        conversation_id = conversation_id_of(thread_id)
        saver = self._saver(state_db, athlete_id=athlete_id, conversation_id=conversation_id)
        if not await saver.consume_interrupt(thread_id, interrupt_id):
            raise DecisionRefused(f"no live interrupt to consume for thread {thread_id!r}")
        async with self._sessions.session(subject=athlete_id) as session:
            svc = AnalyticsService(session)
            accepted_edit: str | None = None
            edit_rejected = False
            if decision == "edit":
                accepted_edit = await accept_edit(
                    self._coach,
                    self._model,
                    svc,
                    athlete_id,
                    edited_plan or "",
                    request_text=await run_request_text(saver, thread_id),
                )
                edit_rejected = accepted_edit is None
            resume: dict[str, Any] = {
                # A rejected edit resumes UN-approved (like a reject), so the run finalizes WITHOUT
                # the untrusted edit; an accepted edit carries its re-grounded body forward.
                "approved": decision == "approve",
                "decision": "reject" if edit_rejected else decision,
            }
            if accepted_edit is not None:
                resume["edited_plan"] = accepted_edit
            graph = self._graph(
                svc, saver, allow_names=CANONICAL_WORKOUT_NAMES, entitlement=entitlement
            )
            config: RunnableConfig = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": RECURSION_LIMIT,
            }
            final = await graph.resume(Command(resume=resume), config)
        projected = _project_plan(final)
        if accepted_edit is not None:
            # An accepted EDIT replaces the delivered body with the RE-GROUNDED edit (GROUND-R3) —
            # the athlete's edited prose, fully grounded, is what becomes final.
            return replace(
                projected, plan_text=accepted_edit, plan_html=safe_plan_html(accepted_edit)
            )
        if edit_rejected:
            # The edit did not fully ground: degrade VISIBLY (never silently ship it). The body is
            # the projected pre-edit grounded plan; the status is forced DEGRADED so the caller sees
            # the edit was not accepted, and the unverified edit text never reaches the athlete.
            return replace(projected, status=RunStatus.DEGRADED)
        return projected

    async def interrupt_status(
        self, *, athlete_id: str, thread_id: str, interrupt_id: str
    ) -> Literal["unknown", "live", "consumed"]:
        """Classify an interrupt for the decision router's 404-vs-409 split (API-R12a / CKPT-R9).

        The read-only probe the router consults ONLY after :meth:`decision` fails closed: an
        athlete-scoped read of the ``AgentInterrupt`` ledger by ``(thread_id, interrupt_id,
        athlete_id)`` returning ``unknown`` (no row this athlete owns — never disclosed) -> ``404``;
        ``consumed`` (already decided) or ``live`` (a concurrent decision won the race) -> ``409``.
        Identity is the server-derived ``athlete_id`` (AUTH-R3 / CKPT-R3), never a foreign row.
        """
        state_db = await self._agent_state_db()
        conversation_id = conversation_id_of(thread_id)
        saver = self._saver(state_db, athlete_id=athlete_id, conversation_id=conversation_id)
        return await saver.interrupt_status(thread_id, interrupt_id)


# The two engine factory functions live in the focused :mod:`engine_factory` sibling (QUAL-R9 size
# split) and are re-exported here so every historical ``from wattwise_core.agent.engine import
# build_agent_engine`` path stays stable. Imported at module end (after ``GraphAgentEngine`` is
# defined) so the sibling's ``from ...engine import GraphAgentEngine`` resolves without a cycle.
from wattwise_core.agent.engine_factory import (  # noqa: E402  (deferred to break import cycle)
    build_agent_engine,
    build_agent_engine_with_model,
)

__all__ = [
    "ClaimGrounder",
    "CoachBundle",
    "DecisionRefused",
    "DeterministicCoverage",
    "GraphAgentEngine",
    "ModelPlanner",
    "RegistryGateway",
    "UnconfiguredAgentEngine",
    "build_agent_engine",
    "build_agent_engine_with_model",
]
