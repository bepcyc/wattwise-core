"""Production agent runtime: concrete services + the deployable :class:`GraphAgentEngine`.

The agent graph, grounding, capabilities, and deliverables are built and tested with
injected seams; this module supplies the CONCRETE production implementations of those
seams (a model-driven planner, the canonical capability gateway, a deterministic
coverage assessor, and a model-extract + code-verify grounder) and assembles them into
an :class:`AgentEngine` the API drives. ``build_agent_engine`` constructs it from
settings + the database; when no LLM key is configured the OSS engine boots without a
model and the agent surface degrades gracefully rather than failing the boot (RUN-R4.1).

Identity is server-derived end to end (AGT-SEC-R1); the model never self-certifies a
verdict (OUTCOME-R5) — it only emits the structured retrieval plan and candidate claims;
deterministic code resolves capabilities and verifies every claim against canonical data.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from wattwise_core.agent import grounding as _grounding
from wattwise_core.agent.capabilities import (
    CAPABILITY_BY_KEY,
    CanonicalEvidence,
    gather,
)
from wattwise_core.agent.contracts import (
    AgentState,
    ChatModel,
    Claim,
    ClaimKind,
    GroundingResult,
    RetrievalRequest,
    RunStatus,
)
from wattwise_core.agent.deliverables import (
    AgentAnswer,
    CoachGraph,
    answer_question,
    weekly_digest,
)
from wattwise_core.agent.graph import DEFAULT_NODE_VISIT_CEILING, build_graph
from wattwise_core.agent.model import OpenAICompatibleModel
from wattwise_core.agent.seams import AgentServices
from wattwise_core.agent.structured import StructuredOutputError, run_structured
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.persistence import Database
from wattwise_core.persistence.types import utcnow

# Date-range capabilities the headline planner can request without an activity id; the
# per-activity/per-day capabilities need an id the planner does not have at plan time.
_DATE_RANGE_CAPABILITIES = ("weekly_load", "critical_power", "power_curve")
_DEFAULT_WINDOW_DAYS = 42

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


class _PlanSchema(BaseModel):
    """Provider-enforced retrieval plan (PLAN-R2): which canonical capabilities to gather."""

    model_config = {"extra": "forbid"}
    capabilities: list[str] = Field(default_factory=list)
    window_days: int = Field(default=_DEFAULT_WINDOW_DAYS, ge=1, le=365)


class _ExtractedClaim(BaseModel):
    """One candidate claim the model points at (STRUCT-R5); code verifies it, not the model."""

    model_config = {"extra": "forbid"}
    kind: ClaimKind = ClaimKind.NUMBER
    text: str = ""
    metric: str | None = None
    value: float | None = None
    as_of: str | None = None


class _ClaimSchema(BaseModel):
    """The structured claim-extraction output (GROUND-R2/STRUCT-R5)."""

    model_config = {"extra": "forbid"}
    claims: list[_ExtractedClaim] = Field(default_factory=list)


_PLAN_SYSTEM = (
    "You are the coaching agent's retrieval planner. Choose which canonical analytics "
    "capabilities to gather to answer the athlete, from the closed set "
    f"{_DATE_RANGE_CAPABILITIES}, plus a window in days. Return ONLY the structured plan."
)
_CLAIM_SYSTEM = (
    "Extract every factual numeric claim in the draft as a candidate claim with its "
    "metric name, value, and the local date (ISO 8601) it is as-of when the draft states "
    "one. Do NOT judge correctness — only point at candidates."
)


class ModelPlanner:
    """Model-driven retrieval planner (PLAN-R1/R2): the structured plan IS the selection."""

    def __init__(self, model: ChatModel, *, reference_date: _dt.date | None = None) -> None:
        self._model = model
        self._today = reference_date or utcnow().date()

    async def plan(
        self, *, request_text: str | None, gaps: Sequence[str], already: Sequence[str]
    ) -> Sequence[RetrievalRequest]:
        """Emit the next batch of capability requests; fail-closed to a default on error."""
        try:
            plan = await run_structured(
                self._model,
                system=_PLAN_SYSTEM,
                data=f"question: {request_text}\nopen_gaps: {list(gaps)}\nalready: {list(already)}",
                schema=_PlanSchema,
            )
            keys = [k for k in plan.capabilities if k in _DATE_RANGE_CAPABILITIES]
            window = plan.window_days
        except (StructuredOutputError, NotImplementedError):
            keys, window = ["weekly_load"], _DEFAULT_WINDOW_DAYS
        if not keys:
            keys = ["weekly_load"]
        frm = self._today - _dt.timedelta(days=window)
        params = {"from_date": frm.isoformat(), "to_date": self._today.isoformat()}
        seen = set(already)
        return [
            RetrievalRequest(capability=k, params=dict(params))
            for k in keys
            if k in CAPABILITY_BY_KEY and k not in seen
        ]


class RegistryGateway:
    """Resolves capability requests to canonical evidence via the one registry (TOOL-R1)."""

    def __init__(self, svc: AnalyticsService) -> None:
        self._svc = svc

    async def gather(
        self, *, athlete_id: str, requests: Sequence[RetrievalRequest]
    ) -> Mapping[str, Any]:
        result = await gather(self._svc, athlete_id, list(requests))
        return result.records


class DeterministicCoverage:
    """Reports planned capabilities that resolved to no canonical evidence (pure)."""

    def assess(self, *, request_text: str | None, retrieved: Mapping[str, Any]) -> set[str]:
        # A turn with no retrieved evidence at all is the only structural gap the headline
        # flow reports; per-capability emptiness is surfaced by the gather records.
        return set() if retrieved else {"no_canonical_evidence"}


class _SnapshotEvidence:
    """Sync grounding evidence: pre-resolved canonical snapshots + first-party URL gate.

    The deterministic grounder (GROUND-R*) is synchronous and reads canonical values via a
    sync ``metric_snapshot``; the canonical :class:`CanonicalEvidence` exposes only the
    async ``metric_value``. This wrapper carries the snapshots an async pass resolved ahead
    of time over the extracted claims, so a NUMBER claim is verified VERBATIM against
    canonical analytics (GROUND-R7) WITHOUT the grounder ever awaiting. ``url_allowed`` /
    ``metric_value`` delegate to the wrapped evidence; no name library is implemented, so
    NAME claims fail closed (Phase-1 ships no canonical workout library) — the conservative
    default (GROUND-R3).
    """

    def __init__(
        self,
        evidence: CanonicalEvidence,
        snapshots: Mapping[tuple[str, str | None], float | None],
    ) -> None:
        self._evidence = evidence
        self._snapshots = snapshots

    def metric_snapshot(self, metric: str, as_of: str | None) -> float | None:
        """The pre-resolved canonical value for ``(metric, as_of)``, or ``None`` (GROUND-R7)."""
        return self._snapshots.get((metric, as_of))

    async def metric_value(self, metric: str, as_of: str | None) -> float | None:
        """Satisfy the async :class:`GroundingEvidence` contract by delegating (GROUND-R2)."""
        return await self._evidence.metric_value(metric, as_of)

    def url_allowed(self, url: str) -> bool:
        """First-party URL allow-list, delegated to the canonical evidence (GROUND-R4)."""
        return self._evidence.url_allowed(url)


class ClaimGrounder:
    """Model-extract + code-verify grounder over canonical evidence (GROUND-R1/R2/R7)."""

    def __init__(self, model: ChatModel, svc: AnalyticsService) -> None:
        self._model = model
        self._svc = svc

    async def ground(
        self, *, athlete_id: str, draft: str, retrieved: Mapping[str, Any]
    ) -> GroundingResult:
        try:
            extracted = await run_structured(
                self._model, system=_CLAIM_SYSTEM, data=draft, schema=_ClaimSchema
            )
            claims = [
                Claim(kind=c.kind, text=c.text, metric=c.metric, value=c.value, ref=c.as_of)
                for c in extracted.claims
            ]
        except (StructuredOutputError, NotImplementedError):
            claims = []
        evidence = CanonicalEvidence(self._svc, athlete_id)
        snapshots = await _resolve_snapshots(evidence, claims)
        snapshot_evidence = _SnapshotEvidence(evidence, snapshots)
        return _grounding.ground(draft, claims, snapshot_evidence, allow_urls=())


async def _resolve_snapshots(
    evidence: CanonicalEvidence, claims: Sequence[Claim]
) -> dict[tuple[str, str | None], float | None]:
    """Resolve each NUMBER claim's canonical value ahead of the synchronous grounder.

    Reads the canonical analytic VERBATIM via the async ``metric_value`` for every distinct
    ``(metric, as_of)`` a NUMBER claim points at (GROUND-R7); the grounder then verifies
    against this snapshot without awaiting. A metric the service cannot compute resolves to
    ``None`` so the grounder scrubs the claim (fail-closed), never a placeholder.
    """
    snapshots: dict[tuple[str, str | None], float | None] = {}
    for claim in claims:
        if claim.kind is not ClaimKind.NUMBER or claim.metric is None:
            continue
        key = (claim.metric, claim.ref)
        if key not in snapshots:
            snapshots[key] = await evidence.metric_value(claim.metric, claim.ref)
    return snapshots


def _build_services(model: ChatModel, svc: AnalyticsService) -> AgentServices:
    """Assemble the concrete production service bundle for the graph (GRAPH-R5)."""
    return AgentServices(
        planner=ModelPlanner(model),
        gateway=RegistryGateway(svc),
        coverage=DeterministicCoverage(),
        grounder=ClaimGrounder(model, svc),
    )


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
        (CKPT-R3); it fails closed if absent rather than aliasing onto a shared constant key
        that could mix checkpoint state across runs under a durable checkpointer.
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


class GraphAgentEngine:
    """The deployable :class:`~wattwise_core.api.routers.agent_routes.AgentEngine`.

    Per call it opens a canonical session, builds the analytics service + the concrete
    agent services + the compiled graph (an in-memory checkpointer per call in OSS — the
    durable SQLAlchemy checkpointer is wired when an agent-state store is configured), and
    runs the Phase-1 deliverable projection (grounded Q&A / weekly digest).
    """

    def __init__(self, database: Database, model: ChatModel) -> None:
        self._db = database
        self._model = model

    def _graph(self, svc: AnalyticsService) -> CoachGraph:
        """Build + compile the coaching graph over the per-call services (GRAPH-R5).

        The ceiling is passed explicitly so it stays tied to ``_RECURSION_LIMIT`` (the
        superstep bound the adapter applies must remain above the graph's own ceiling).
        """
        compiled = build_graph(
            self._model,
            _build_services(self._model, svc),
            InMemorySaver(),
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
        async with self._db.session() as session:
            return await answer_question(
                self._graph(AnalyticsService(session)),
                athlete_id,
                question or "",
                locale=locale,
                response_length=response_length,  # type: ignore[arg-type]
            )

    async def digest(self, *, athlete_id: str, week_end: str) -> Any:
        async with self._db.session() as session:
            return await weekly_digest(self._graph(AnalyticsService(session)), athlete_id, week_end)


class UnconfiguredAgentEngine:
    """Graceful no-op engine when the OSS deployment has no LLM configured (RUN-R4.1).

    The engine boots without a model; the coaching surface then returns a typed,
    jargon-free ``degraded`` answer (no internals leaked, VOICE-R2/-R3) rather than the
    boot failing or the endpoint erroring. Configuring a model upgrades it in place.
    """

    _MESSAGE: ClassVar[dict[str, str]] = {
        "en": "Coaching isn't switched on for this account yet.",
        "de": "Coaching ist fuer dieses Konto noch nicht aktiviert.",
        "ru": "Trener poka ne podklyuchyon dlya etoy uchyotnoy zapisi.",
    }

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
        text = self._MESSAGE.get((locale or "en").split("-", 1)[0].lower(), self._MESSAGE["en"])
        return AgentAnswer(
            status=RunStatus.DEGRADED,
            thread_id=thread_id or "unconfigured",
            answer_html=f"<p>{text}</p>",
            answer_text=text,
            coverage_caveat={"reason": "agent_unconfigured"},
        )


def build_agent_engine(database: Database, settings: Any) -> GraphAgentEngine | None:
    """Build the production engine from settings, or ``None`` when no model is configured.

    The OSS engine boots without an LLM key (RUN-R4.1 does not require one); when the key
    is absent this returns ``None`` and the API leaves the agent endpoints surfacing a
    typed, jargon-free unavailable rather than failing the whole boot.
    """
    if settings.llm_api_key is None:
        return None
    return GraphAgentEngine(database, OpenAICompatibleModel(settings=settings))


def build_agent_engine_with_model(database: Database, model: ChatModel) -> GraphAgentEngine:
    """Build the engine with an injected model (the test seam for a deterministic FakeModel)."""
    return GraphAgentEngine(database, model)


__all__ = [
    "ClaimGrounder",
    "DeterministicCoverage",
    "GraphAgentEngine",
    "ModelPlanner",
    "RegistryGateway",
    "UnconfiguredAgentEngine",
    "build_agent_engine",
    "build_agent_engine_with_model",
]
