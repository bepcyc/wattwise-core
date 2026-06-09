"""The deployable :class:`GraphAgentEngine` the API drives (doc 50).

Assembles the compiled coaching graph over a DURABLE :class:`SqlAlchemyCheckpointSaver` (on a
dedicated agent-state pool, ARCH-R13/DEPLOY-R4) + the concrete production services (in the focused
:mod:`engine_services` sibling, re-exported here) and runs the deliverable projection: grounded
Q&A + COACH-R8 follow-ups, the weekly digest, the multi-day PLAN, and the HITL decision resume.
``build_agent_engine`` constructs it from settings + the database; with no LLM key the OSS engine
boots without a model and the agent surface degrades gracefully (RUN-R4.1). Identity is server-
derived (AGT-SEC-R1); the model never self-certifies (OUTCOME-R5); the durable saver makes
follow-ups and approval pauses resumable (CKPT-R5/-R9).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

# The concrete production agent services + the canonical workout-NAME library live in the focused
# sibling :mod:`engine_services` (QUAL-R9 size split); the public ones are re-exported below so
# every historical ``from wattwise_core.agent.engine import ClaimGrounder/ModelPlanner/...`` path
# stays stable (e.g. the integration tests import ``ClaimGrounder``/``_PlanSchema``).
from wattwise_core.agent.checkpoint import SqlAlchemyCheckpointSaver
from wattwise_core.agent.contracts import AgentState, ChatModel, RunStatus
from wattwise_core.agent.deliverables import (
    AgentAnswer,
    Digest,
    Plan,
    answer_question,
    conversation_id_of,
    new_conversation_id,
    weekly_digest,
)
from wattwise_core.agent.engine_extras import DeliverableEngineMixin
from wattwise_core.agent.engine_services import (  # noqa: F401  re-exported (historical paths)
    CANONICAL_WORKOUT_NAMES,
    ClaimGrounder,
    CoachBundle,
    DeterministicCoverage,
    ModelPlanner,
    RegistryGateway,
    _ClaimSchema,
    _ExtractedClaim,
    _PlanSchema,
    build_services,
)
from wattwise_core.agent.graph import DEFAULT_NODE_VISIT_CEILING, build_graph
from wattwise_core.agent.model import OpenAICompatibleModel
from wattwise_core.agent.plan_deliverable import _project_plan, safe_plan_html
from wattwise_core.agent.plan_deliverable import plan as _plan
from wattwise_core.agent.plan_regrounding import accept_edit
from wattwise_core.agent.state_db import (
    AgentStateDatabase,
    build_agent_state_database,
)
from wattwise_core.agent.state_db import (
    fallback_state_dsn as _fallback_state_dsn,
)
from wattwise_core.agent.unconfigured import UnconfiguredAgentEngine
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.persistence import Database

# The compiled-graph type the deliverables drive through the ``CoachGraph`` seam.
_CompiledGraph = CompiledStateGraph[AgentState, Any, AgentState, AgentState]

# The node-visit ceiling the production graph is compiled with, and the langgraph
# superstep bound — kept TOGETHER so the invariant ``recursion_limit > ceiling`` holds for
# whatever ceiling is configured. The bound sits ABOVE the ceiling so a pathological run
# finalizes gracefully (degraded, OUTCOME-R3) via the graph's own ceiling rather than
# raising a GraphRecursionError first; the bounded reflect/redraft counters guarantee
# termination well before either bound on every legal path.
_NODE_VISIT_CEILING = DEFAULT_NODE_VISIT_CEILING
_RECURSION_LIMIT = _NODE_VISIT_CEILING + 20


def _conversation_id(athlete_id: str, thread_id: str | None) -> str:
    """The saver-bound conversation id for a run, REVERSIBLE with the thread_id (CKPT-R3).

    A follow-up/decision passes the prior ``thread_id`` back: the conversation id is recovered
    from it (``conversation_id_of``) so the saver binds to the SAME durable thread. A fresh turn
    (no ``thread_id``) mints a new conversation id; the SAME value is passed to the deliverable so
    the thread_id it builds (``{athlete_id}:{conversation_id}``) matches the saver's bound scope —
    otherwise the graph config's thread_id and the saver's thread row would diverge.
    """
    if thread_id is not None:
        return conversation_id_of(thread_id)
    return new_conversation_id()


class _CompiledCoachGraph:
    """Adapt a compiled LangGraph to the deliverables' :class:`CoachGraph` seam (GRAPH-R1).

    ``deliverables.answer_question`` drives the graph through the typed async ``run(state)``
    seam; a compiled langgraph instead exposes ``ainvoke`` and REQUIRES a per-run config
    carrying the durable ``thread_id`` (the checkpointer key, CKPT-R3) plus a recursion
    bound. This wrapper supplies both from the immutable input state so the production engine
    invokes the graph exactly as the grounded-Q&A deliverable expects — without it the
    deliverable's ``graph.run`` call would not resolve against the bare compiled graph.
    """

    def __init__(self, compiled: _CompiledGraph) -> None:
        self._compiled = compiled

    async def run(self, state: AgentState) -> AgentState:
        """Invoke the compiled graph with the durable-thread config (CKPT-R3, OUTCOME-R2).

        The thread id MUST come from the run's own ``(athlete_id, conversation_id)`` scope
        (CKPT-R3); it fails closed if absent rather than aliasing onto a shared key.
        """
        thread_id = state.get("thread_id") or state.get("idempotency_key")
        if not thread_id:
            raise ValueError("agent run state carries no durable thread id (CKPT-R3)")
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": _RECURSION_LIMIT,
        }
        result = await self._compiled.ainvoke(state, config=config)
        return cast(AgentState, result)

    async def resume(self, command: Command[Any], config: RunnableConfig) -> AgentState:
        """Resume a paused run with ``Command(resume=...)`` on the SAME durable thread (CKPT-R2).

        The head node does NOT re-run (no recompute, no fresh turn_id); the pre-interrupt nodes
        replay from the checkpoint rather than re-executing. Returns the terminal state.
        """
        result = await self._compiled.ainvoke(command, config=config)
        return cast(AgentState, result)


class DecisionRefused(RuntimeError):
    """A HITL decision could not consume a live interrupt (CKPT-R9; fail-closed).

    Raised by :meth:`GraphAgentEngine.decision` when ``consume_interrupt`` returns ``False`` —
    the atomic guarded UPDATE matched no ``live`` row owned by the caller (an already-consumed
    double-decision F-409, an unknown/never-recorded interrupt F-404, or a cross-athlete attempt
    F-XID). The run is NEVER resumed in that case; the API router maps this to 404/409.
    """


class GraphAgentEngine(DeliverableEngineMixin):  # noqa: size-limits
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
    ) -> None:
        self._db = database
        self._model = model
        self._state_db = state_db
        self._coach = coach if coach is not None else CoachBundle()

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

    def _graph(
        self,
        svc: AnalyticsService,
        saver: SqlAlchemyCheckpointSaver,
        *,
        allow_names: frozenset[str] = frozenset(),
    ) -> _CompiledCoachGraph:
        """Build + compile the coaching graph over the per-call services + DURABLE saver (GRAPH-R5).

        The ceiling is tied to ``_RECURSION_LIMIT``; the durable saver records a ``live`` ledger row
        (CKPT-R9) so runs resume across turns/pauses. The loaded coach-config (§16) is wired in here
        via ``self._coach`` — compose gets the real system prompt and the grounder gets the metric-
        equivalence + tolerance, so a natural metric the model cites grounds (the headline fix).
        """
        compiled = build_graph(
            self._model,
            self._coach.services(self._model, svc, allow_names=allow_names),
            saver,
            coach_system=self._coach.system_prompt,
            node_visit_ceiling=_NODE_VISIT_CEILING,
        )
        return _CompiledCoachGraph(compiled)

    async def answer(
        self,
        *,
        athlete_id: str,
        question: str | None,
        thread_id: str | None,
        response_length: str,
        follow_up: dict[str, Any] | None,
        locale: str,
    ) -> AgentAnswer:
        """Answer a question, resuming the SAME durable thread on a follow-up (COACH-R8).

        A first turn opens a fresh durable thread; a follow-up (the caller passes the prior
        ``thread_id`` back) lands on the SAME ``(athlete_id, conversation_id)`` so an
        expand/drill/reveal turn continues the conversation. The durable saver is bound to that
        scope; ``follow_up`` shapes the turn in the deliverable (CKPT-R3/-R5).
        """
        state_db = await self._agent_state_db()
        conversation_id = _conversation_id(athlete_id, thread_id)
        saver = self._saver(state_db, athlete_id=athlete_id, conversation_id=conversation_id)
        async with self._db.session() as session:
            return await answer_question(
                self._graph(AnalyticsService(session), saver),
                athlete_id,
                question or "",
                locale=locale,
                response_length=response_length,  # type: ignore[arg-type]
                thread_id=thread_id,
                conversation_id=conversation_id,
                follow_up=follow_up,
            )

    async def digest(self, *, athlete_id: str, week_end: str) -> Digest:
        """Build the weekly digest (== weekly load review) for ``week_end`` (COACH-R1 #1).

        Drives a ``scheduled_digest`` run over a deterministic per-week conversation id and projects
        the grounded trailing-week review into :class:`Digest`, abstaining visibly (``degraded`` +
        caveat) when the week's canonical inputs are missing (OUTCOME-R3/-R4, GROUND-R7).
        """
        state_db = await self._agent_state_db()
        conversation_id = f"digest:{week_end}"
        saver = self._saver(state_db, athlete_id=athlete_id, conversation_id=conversation_id)
        async with self._db.session() as session:
            graph = self._graph(AnalyticsService(session), saver)
            return await weekly_digest(graph, athlete_id, week_end)

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
    ) -> Plan:
        """Build a multi-day grounded training PLAN, approval-gated by default (COACH-R2/CKPT-R5).

        Drives the graph with the canonical workout-NAME library so prescribed names ground (not
        auto-scrubbed). With ``requires_approval`` the durable interrupt-gate pauses the run and
        records a ``live`` ledger row; the :class:`Plan` is ``awaiting_approval`` carrying the
        ``interrupt_id`` the decision endpoint consumes (CKPT-R9). ``athlete_id`` is server-derived
        (AGT-SEC-R1). ``request_text`` is the planning prompt the planning router passes;
        ``request`` is a backward-compatible alias for the same value.
        """
        prompt = request_text if request_text is not None else (request or "")
        state_db = await self._agent_state_db()
        conversation_id = _conversation_id(athlete_id, thread_id)
        saver = self._saver(state_db, athlete_id=athlete_id, conversation_id=conversation_id)
        async with self._db.session() as session:
            graph = self._graph(
                AnalyticsService(session), saver, allow_names=CANONICAL_WORKOUT_NAMES
            )
            return await _plan(
                graph,
                athlete_id,
                prompt,
                locale=locale,
                response_length=response_length,  # type: ignore[arg-type]
                thread_id=thread_id,
                conversation_id=conversation_id,
                requires_approval=requires_approval,
            )

    async def decision(
        self,
        *,
        athlete_id: str,
        thread_id: str,
        interrupt_id: str,
        decision: str,
        edited_plan: str | None = None,
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
        resume payload.
        """
        state_db = await self._agent_state_db()
        conversation_id = conversation_id_of(thread_id)
        saver = self._saver(state_db, athlete_id=athlete_id, conversation_id=conversation_id)
        if not await saver.consume_interrupt(thread_id, interrupt_id):
            raise DecisionRefused(f"no live interrupt to consume for thread {thread_id!r}")
        async with self._db.session() as session:
            svc = AnalyticsService(session)
            accepted_edit: str | None = None
            edit_rejected = False
            if decision == "edit":
                accepted_edit = await accept_edit(
                    self._coach, self._model, svc, athlete_id, edited_plan or ""
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
            graph = self._graph(svc, saver, allow_names=CANONICAL_WORKOUT_NAMES)
            config: RunnableConfig = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": _RECURSION_LIMIT,
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

        The read-only, side-effect-free probe the router consults ONLY after :meth:`decision` fails
        closed: an athlete-scoped read of the ``AgentInterrupt`` ledger by
        ``(thread_id, interrupt_id, athlete_id)`` returning ``unknown`` (no row this athlete owns —
        unknown thread/interrupt OR a foreign athlete's, never disclosed) -> router ``404``;
        ``consumed`` (already decided) or ``live`` (a concurrent decision won the race) -> router
        ``409``. Identity is the server-derived ``athlete_id`` (AUTH-R3 / CKPT-R3), scoped via the
        durable saver bound to the thread's ``(athlete_id, conversation_id)`` — never disclosing a
        foreign row.
        """
        state_db = await self._agent_state_db()
        conversation_id = conversation_id_of(thread_id)
        saver = self._saver(state_db, athlete_id=athlete_id, conversation_id=conversation_id)
        return await saver.interrupt_status(thread_id, interrupt_id)


def build_agent_engine(database: Database, settings: Any) -> GraphAgentEngine | None:
    """Build the production engine from settings, or ``None`` when no model is configured.

    The OSS engine boots without an LLM key (RUN-R4.1 does not require one); when the key is
    absent this returns ``None`` and the API leaves the agent endpoints surfacing a typed,
    jargon-free unavailable rather than failing the whole boot. When a model IS configured the
    engine is wired with a DEDICATED agent-state database (its own engine/pool, ARCH-R13/DEPLOY-R4)
    so the durable checkpointer never contends with the canonical pool (SPIKE-3 deadlock-freedom).
    """
    if settings.llm_api_key is None:
        return None
    state_db = build_agent_state_database(settings)
    return GraphAgentEngine(
        database,
        OpenAICompatibleModel(settings=settings),
        state_db=state_db,
        coach=CoachBundle.from_settings(settings),
    )


def build_agent_engine_with_model(
    database: Database,
    model: ChatModel,
    *,
    state_db: AgentStateDatabase | None = None,
    coach: CoachBundle | None = None,
) -> GraphAgentEngine:
    """Build the engine with an injected model + optional ``state_db``/``coach`` (FakeModel seam).

    The durable tests pass a REAL pooled ``state_db`` (file-sqlite/PG/MariaDB); when omitted the
    engine lazily builds the per-process file-sqlite fallback. ``coach`` injects a §16 coach-config
    (the live test passes the loaded bundle; deterministic FakeModel tests pass the empty default,
    since FakeModel scripts exact canonical claims needing no prompt steering or equivalence).
    """
    return GraphAgentEngine(database, model, state_db=state_db, coach=coach)


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
