"""Pure state readers and the deterministic terminal-status logic for the agent graph.

This module factors the side-effect-free helpers out of :mod:`wattwise_core.agent.graph`
so each file stays focused and under the size ceilings (QUAL-R9). Everything here is a pure
function of the typed :class:`~wattwise_core.agent.contracts.AgentState` (GRAPH-R4) — node
readers, the INJECT-R1 context envelope + MODEL-R3 token budget, and the OUTCOME-R1/-R4/-R5
terminal-status + coverage-caveat logic. It depends only on
:mod:`wattwise_core.agent.contracts` and the contracts-only structured-output helper
(REFLECT-R2), never on a sibling in-flight agent file (ARCH-R21).
"""

from __future__ import annotations

import html as _html
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from wattwise_core.agent.contracts import (
    RETRIEVED_MAX_RECORDS,
    RETRIEVED_TRUNCATION_KEY,
    TURN_COUNTER_FLOOR,
    AgentState,
    ChatModel,
    CoverageCaveat,
    GroundDecision,
    ReflectDecision,
    ReflectVerdict,
    RetrievalRequest,
    RunStatus,
    stamp_coverage_gaps,
    stamp_retrieved,
    turn_gaps,
    turn_records,
)

# Re-export the COACH-R8 stable-id observation projection from its focused LEAF module
# (:mod:`observations`, QUAL-R9 size split) so the existing ``gs.build_observations`` call site in
# the ``ground`` node keeps resolving unchanged (ARCH-R21: the leaf depends only on contracts).
from wattwise_core.agent.observations import build_observations
from wattwise_core.agent.structured import StructuredOutputError, run_structured

# Bounded recovery budgets (REFLECT-R4), shared with the graph for routing decisions.
MAX_REFLECTIONS = 2
MAX_REDRAFTS = 2


@runtime_checkable
class InterruptRecorder(Protocol):
    """Seam the interrupt-gate uses to persist a ``live`` AgentInterrupt row (CKPT-R9).

    The durable checkpointer (``SqlAlchemyCheckpointSaver``) satisfies this Protocol via its
    ``record_interrupt`` method; an in-memory checkpointer (the OSS/test default) does NOT, in
    which case ``build_graph`` passes ``None`` and the gate raises the interrupt without a
    ledger row (no durable approval can be consumed against an in-memory saver anyway). Keeping
    the seam contracts-only preserves ARCH-R21 (the graph never imports the concrete saver).
    """

    async def record_interrupt(self, thread_id: str, interrupt_id: str) -> None: ...


def athlete_id(state: AgentState) -> str:
    """Read the server-derived athlete id from immutable input (AGT-SEC-R1).

    Fail-closed: a run with no server-set identity cannot proceed; we never invent
    or accept a model/tool-supplied id.
    """
    value = state.get("athlete_id")
    if not value:
        raise ValueError("athlete_id is server-derived and required (AGT-SEC-R1)")
    return value


def tick_visit(state: AgentState, update: dict[str, Any]) -> dict[str, Any]:
    """Advance the monotonic node-visit counter on a node's partial update (GRAPH-R5)."""
    update["node_visits"] = state.get("node_visits", 0) + 1
    return update


def over_ceiling(state: AgentState, ceiling: int) -> bool:
    """True once the configured node-visit ceiling is reached (GRAPH-R5)."""
    return state.get("node_visits", 0) >= ceiling


def over_tool_ceiling(state: AgentState, max_tool_iterations: int) -> bool:
    """True once the resolved entitlement's tool-iteration bound is reached (AGT-ENT-R4).

    Reads the monotonic ``tool_iterations`` counter (advanced by ``gather`` on each real
    capability resolution) against the bound the graph carries FROM the resolved entitlement
    (AGT-ENT-R1), so the gather/tool loop is bounded independently of ``node_visits``. The
    routers consult this to stop re-planning (route to compose) on a breach — a GRACEFUL bound,
    never a raise.
    """
    return state.get("tool_iterations", 0) >= max_tool_iterations


# --- turn boundary: the run-scoped reset + turn-keyed accumulator views (CKPT-R5) ---


def turn_id(state: AgentState) -> str:
    """The current turn id (``""`` when none is set, e.g. a legacy single-turn run).

    Minted fresh per normal ``/ask`` ``ainvoke`` by the caller; a ``Command(resume)`` never
    mints or changes it (the head node does not run on resume). The head node stamps it onto
    every run-scoped accumulator write so the turn-keyed reducers can self-reset (CKPT-R5).
    """
    return state.get("turn_id") or ""


def is_new_turn(state: AgentState) -> bool:
    """True when this invocation opens a NEW turn on a durable thread (CKPT-R5).

    A new turn is one whose ``turn_id`` differs from the ``run_epoch`` the run-scoped channels
    currently belong to. On the very first turn ``run_epoch`` is the unset sentinel ``""`` so
    any non-empty ``turn_id`` reads as new (and the reset is a harmless no-op on empty state).
    A ``Command(resume)`` does not run the head node, so this is never evaluated on resume —
    the run-scoped channels are preserved across the pause (CKPT-R5 "no recomputation").
    """
    tid = turn_id(state)
    if not tid:
        return False
    return tid != (state.get("run_epoch") or "")


def read_retrieved(state: AgentState) -> dict[str, Any]:
    """The ``retrieved`` channel with the in-band turn marker stripped (reader view)."""
    return turn_records(state.get("retrieved", {}))


def read_coverage_gaps(state: AgentState) -> set[str]:
    """The ``coverage_gaps`` channel with the in-band turn marker stripped (reader view)."""
    return turn_gaps(state.get("coverage_gaps", set()))


def reset_run_scoped(state: AgentState) -> dict[str, Any]:
    """The head-node partial update that resets every run-scoped channel on a new turn.

    A durable thread reuses ONE checkpoint across many turns, so the single head node MUST
    reset the RUN-SCOPED channels at a new-turn boundary or turn-1 evidence/counters leak into
    turn-2 (the reverted force-degrade + leak bug). The three counters go to the sentinel floor
    ``0`` (the ONLY decrease :func:`~wattwise_core.agent.contracts._turn_monotonic` allows,
    single-writer = this node); ``retrieved`` / ``coverage_gaps`` are stamped EMPTY with the new
    ``turn_id`` so the turn-keyed reducers drop the prior turn's value; ``run_epoch`` advances to
    the new ``turn_id`` so subsequent nodes this turn see ``is_new_turn() is False``. This update
    is the SINGLE writer of the floor / new epoch — node-local ticks never reset.
    """
    tid = turn_id(state)
    return {
        "node_visits": TURN_COUNTER_FLOOR,
        "reflection_count": TURN_COUNTER_FLOOR,
        "redraft_count": TURN_COUNTER_FLOOR,
        "tool_iterations": TURN_COUNTER_FLOOR,
        "retrieved": stamp_retrieved(tid, {}),
        "coverage_gaps": stamp_coverage_gaps(tid, set()),
        "run_epoch": tid,
    }


def budget_exceeded(state: AgentState) -> bool:
    """Read whether the cost-admission gate refused this run (COST-R4)."""
    for msg in reversed(state.get("messages", [])):
        if msg.get("kind") == "budget":
            return not bool(msg.get("admitted", True))
    return False


def last_plan_requests(state: AgentState) -> list[RetrievalRequest]:
    """Recover the most recent plan's requests from working memory (pure read)."""
    for msg in reversed(state.get("messages", [])):
        if msg.get("kind") == "plan":
            raw = msg.get("requests", [])
            return [RetrievalRequest(capability=r["capability"], params=r["params"]) for r in raw]
    return []


def retrieved_truncation_gaps(state: AgentState, incoming: Mapping[str, Any]) -> set[str]:
    """Surface a STATE-R6 retrieved-truncation as a coverage gap, once (pure read).

    ``prior`` is read through :func:`read_retrieved` so the in-band turn marker is stripped
    before the count check — the marker is bookkeeping, not a retrieved record, and must not
    inflate the count toward the bound.
    """
    prior = read_retrieved(state)
    will_total = {**prior, **incoming}
    will_total.pop(RETRIEVED_TRUNCATION_KEY, None)
    if len(will_total) > RETRIEVED_MAX_RECORDS:
        return {"retrieved_truncated"}
    return set()


def open_gaps(state: AgentState) -> list[str]:
    """Read the freshest coverage assessment (pure)."""
    for msg in reversed(state.get("messages", [])):
        if msg.get("kind") == "coverage":
            gaps = msg.get("open_gaps", [])
            return list(gaps)
    return []


def last_reflect_verdict(state: AgentState) -> ReflectVerdict | None:
    """Read the freshest structured reflect verdict (pure)."""
    for msg in reversed(state.get("messages", [])):
        if msg.get("kind") == "reflect":
            raw = msg.get("verdict")
            return ReflectVerdict(raw) if raw is not None else None
    return None


def last_ground_decision(state: AgentState) -> GroundDecision | None:
    """Read the freshest grounding decision (pure)."""
    for msg in reversed(state.get("messages", [])):
        if msg.get("kind") == "ground":
            return GroundDecision(msg["decision"])
    return None


def plan_requires_approval(state: AgentState) -> bool:
    """Whether the grounded deliverable is a PLAN that product policy gates (CKPT-R5).

    Approval is driven by the DELIVERABLE TYPE + product approval policy — NEVER by a
    grounder abstain (GROUND-R9/CKPT-R5). Phase-1 ships no PLAN deliverable, so this is
    essentially always ``False`` and ``awaiting_approval`` never fires.
    """
    for msg in reversed(state.get("messages", [])):
        if msg.get("kind") == "plan_deliverable":
            return bool(msg.get("requires_approval"))
    return False


# --- reflect verdict (REFLECT-R2 / STRUCT-R1) ---


async def reflect_decision(
    model: ChatModel, state: AgentState, *, system: str = ""
) -> ReflectDecision:
    """Obtain the structured reflect verdict, fail-closed on a structured-output error.

    ``system`` is the externalized reflection system prompt (§16 / SKILL-R1, CFG-R3): the engine
    embeds NO prompt inline (ARCH-R29) — the graph threads the verbatim fragment loaded from the
    coach-config bundle. The empty default keeps the FakeModel suite green (it scripts the verdict,
    so the prompt text is immaterial offline).
    """
    gaps = sorted(read_coverage_gaps(state))
    already = sorted(read_retrieved(state).keys())
    data = f"open_gaps: {gaps}\nalready_retrieved: {already}"
    try:
        return await run_structured(model, system=system, data=data, schema=ReflectDecision)
    except (StructuredOutputError, NotImplementedError):
        return ReflectDecision(verdict=ReflectVerdict.GIVE_UP_GRACEFULLY)


# --- compose context: INJECT-R1 envelope + MODEL-R3 token budget ---

# A coarse input-token budget for the assembled context (MODEL-R3). The OSS engine ships
# no provider tokenizer offline, so the default counter is a deterministic word/char
# estimate; a deployment injects the real tokenizer via the same seam. Overflow trims
# the lowest-relevance retrieved records FIRST and records the trim in coverage_gaps.
_CONTEXT_TOKEN_BUDGET = 6000


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate (MODEL-R3 default; ~4 chars/token, words floor)."""
    return max(len(text) // 4, len(text.split()))


def context_relevance(record: Any) -> float:
    """The relevance gather stamped on a record, for lowest-relevance-first trim."""
    if isinstance(record, dict):
        rel = record.get("relevance")
        if isinstance(rel, (int, float)) and not isinstance(rel, bool):
            return float(rel)
    return 0.0


def render_context(
    request_text: str | None,
    retrieved: Mapping[str, Any],
    *,
    active_goals: Sequence[Mapping[str, Any]] | None = None,
    recalled_memory: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, bool]:
    """Serialise canonical evidence within a measured token budget (MODEL-R3, INJECT-R1, MEM-R4).

    Untrusted content (the user request, every retrieved record body, the athlete's active-goal
    labels/notes, AND any recalled durable-memory items) is wrapped in an explicit delimited
    ``<untrusted-data>`` envelope so injected instructions in titles/notes/memory bodies cannot read
    as instructions (INJECT-R1/MEM-R3). The ``active_goals`` block carries the athlete's ACTIVE
    canonical goals (GBO-R38 / API-R32) so the agent plans TOWARD them; the ``recalled_memory``
    carries durable personalization context recalled through the MemoryStore seam (MEM-R4): stated
    goals/constraints/preferences in the athlete's own words (MEM-R1/-R2). BOTH are user-authored
    INTENT/personalization, never an analytic number (MEM-R1), so they steer the draft but are NOT
    grounding facts (the §7 grounder still reads every number LIVE from canonical analytics). The
    assembled input is measured with the token estimator; on overflow the lowest-relevance records
    are dropped FIRST and a flag is returned so the caller records the trim in coverage_gaps.
    Returns ``(context, trimmed)``.
    """
    header = f"request:\n<untrusted-data>\n{request_text or ''}\n</untrusted-data>"
    rendered: list[str] = [header]
    trimmed = False
    memory_block = _render_recalled_memory(recalled_memory)
    if memory_block is not None:
        rendered.append(memory_block)
    goals_block = _render_active_goals(active_goals)
    if goals_block is not None:
        rendered.append(goals_block)
    items = sorted(
        ((k, v) for k, v in retrieved.items() if k != RETRIEVED_TRUNCATION_KEY),
        key=lambda kv: context_relevance(kv[1]),
        reverse=True,
    )
    for key, value in items:
        candidate = f"{key}:\n<untrusted-data>\n{value}\n</untrusted-data>"
        if estimate_tokens("\n".join([*rendered, candidate])) > _CONTEXT_TOKEN_BUDGET:
            trimmed = True
            continue
        rendered.append(candidate)
    return "\n".join(rendered), trimmed


def _render_active_goals(active_goals: Sequence[Mapping[str, Any]] | None) -> str | None:
    """Render the active-goal context block, or ``None`` when there are none (GBO-R38 / INJECT-R1).

    Each goal is summarised from its canonical typed fields (title, goal type, sport, target event/
    date/metric/value) so the planner reasons over what the athlete is working toward. The whole
    block is wrapped in ``<untrusted-data>`` because ``title``/``target_event`` are user-authored
    free text (MAP-R7): they are DATA the agent plans toward, never instructions it obeys.
    """
    if not active_goals:
        return None
    lines = [_goal_line(goal) for goal in active_goals]
    body = "\n".join(line for line in lines if line)
    if not body:
        return None
    return f"active_goals:\n<untrusted-data>\n{body}\n</untrusted-data>"


def _render_recalled_memory(recalled: Sequence[Mapping[str, Any]] | None) -> str | None:
    """Render the recalled durable-memory block, or ``None`` when none (MEM-R4 / INJECT-R1).

    Each item is summarised from its ``kind`` + raw ``content`` (the athlete's own words, MEM-R2) so
    the agent personalizes its answer to stated goals/constraints/preferences. The whole block is
    wrapped in ``<untrusted-data>`` because the content is user-authored free text that an attacker
    could have planted (MEM-R3): it is personalization DATA the agent considers, NEVER instructions
    it obeys, and NEVER a source of an analytic number (MEM-R1 — numbers always come live from §7).
    """
    if not recalled:
        return None
    lines = [_memory_line(item) for item in recalled]
    body = "\n".join(line for line in lines if line)
    if not body:
        return None
    return f"athlete_memory:\n<untrusted-data>\n{body}\n</untrusted-data>"


def _memory_line(item: Mapping[str, Any]) -> str:
    """One recalled memory item summarised from its ``kind`` + raw content (MEM-R2)."""
    content = item.get("content")
    if not content:
        return ""
    kind = item.get("kind")
    return f"{kind}: {content}" if kind else str(content)


def _goal_line(goal: Mapping[str, Any]) -> str:
    """One athlete goal summarised from its typed canonical fields (GBO-R36)."""
    parts: list[str] = []
    title = goal.get("title")
    if title:
        parts.append(f"goal: {title}")
    for key in ("goal_type", "sport", "target_event", "target_date", "target_metric"):
        value = goal.get(key)
        if value:
            parts.append(f"{key}={value}")
    target_value = goal.get("target_value")
    if target_value is not None:
        parts.append(f"target_value={target_value}")
    return "; ".join(parts)


def safe_html(text: str) -> str:
    """Server-side-sanitized HTML body from grounded text (AGT-SEC-R2).

    The engine never emits model-composed HTML into a client-facing field; it escapes the
    grounded TEXT (so any ``<``/``>``/``&`` from a draft cannot become live markup) and
    wraps it in a single paragraph. No raw model HTML is ever aliased into ``answer_html``.
    """
    return f"<p>{_html.escape(text)}</p>" if text else ""


# Deterministic, jargon-free limitation copy for an abstaining run (GROUND-R6, VOICE-R2/R3).
# This is the fail-closed safety floor the engine guarantees when grounding cannot verify
# enough to answer; a loaded coach persona MAY re-voice it, but the engine never ships the
# scrubbed draft as if it were an answer. No internal terms; warm + truthful.
_LIMITATION_TEXT = {
    "en": "I don't have enough confirmed data to answer that reliably yet. "
    "Sync your sources and I'll take another look.",
    "de": "Mir fehlen noch genug gesicherte Daten, um das verlaesslich zu beantworten. "
    "Synchronisiere deine Quellen und ich schaue noch einmal.",
    "ru": "Poka nedostatochno podtverzhdyonnyh dannyh, chtoby otvetit' nadyozhno. "
    "Sinhroniziruj istochniki i ya posmotryu snova.",
}


def limitation_text(state: AgentState) -> str:
    """The localized fail-closed limitation statement for an abstaining run (GROUND-R6)."""
    locale = (state.get("locale") or "en").split("-", 1)[0].lower()
    return _LIMITATION_TEXT.get(locale, _LIMITATION_TEXT["en"])


# --- terminal status + coverage caveat (OUTCOME-R1/-R4/-R5; no self-grading) ---


def terminal_status(
    state: AgentState, decision: GroundDecision | None, ceiling: int
) -> RunStatus:
    """Deterministically pick the single terminal status (OUTCOME-R1/-R5; no self-grading).

    ``awaiting_approval`` is NEVER produced here (it is yielded by the durable interrupt at
    interrupt_gate). ``budget_exceeded`` on a refused admission; ``degraded`` on a ceiling
    breach, a grounder abstain, a bound-exhausted non-PROCEED recovery, or open gaps;
    otherwise ``completed``.
    """
    if budget_exceeded(state):
        return RunStatus.BUDGET_EXCEEDED
    if over_ceiling(state, ceiling):
        return RunStatus.DEGRADED
    if decision is GroundDecision.ABSTAIN:
        return RunStatus.DEGRADED
    # Reaching finalize with a non-PROCEED ground decision means a bound-exhausted
    # recovery (regenerate/replan) that could not be re-grounded -> DEGRADED, never a
    # complete-looking answer (GROUND-R9/REFLECT-R4).
    if decision is not None and decision is not GroundDecision.PROCEED:
        return RunStatus.DEGRADED
    if open_gaps(state):
        return RunStatus.DEGRADED
    return RunStatus.COMPLETED


def build_caveat(
    state: AgentState, status: RunStatus, decision: GroundDecision | None
) -> dict[str, Any] | None:
    """Build the typed OUTCOME-R4 coverage caveat for a non-completed outcome."""
    gaps = open_gaps(state)
    if status is RunStatus.COMPLETED and not gaps:
        return None
    fidelity = "degraded" if decision is GroundDecision.ABSTAIN else "partial"
    caveat = CoverageCaveat(missing=tuple(sorted(gaps)), fidelity=fidelity)
    return caveat.model_dump()


def cost_rollup(state: AgentState, status: RunStatus) -> dict[str, Any]:
    """Per-run cost/latency rollup carried on every outcome (OUTCOME-R2)."""
    events = state.get("cost_events", [])
    return {
        "node_visits": state.get("node_visits", 0),
        "cost_event_count": len(events),
        "status": status.value,
    }


__all__ = [
    "MAX_REDRAFTS",
    "MAX_REFLECTIONS",
    "InterruptRecorder",
    "athlete_id",
    "budget_exceeded",
    "build_caveat",
    "build_observations",
    "context_relevance",
    "cost_rollup",
    "estimate_tokens",
    "is_new_turn",
    "last_ground_decision",
    "last_plan_requests",
    "last_reflect_verdict",
    "limitation_text",
    "open_gaps",
    "over_ceiling",
    "over_tool_ceiling",
    "plan_requires_approval",
    "read_coverage_gaps",
    "read_retrieved",
    "reflect_decision",
    "render_context",
    "reset_run_scoped",
    "retrieved_truncation_gaps",
    "safe_html",
    "terminal_status",
    "tick_visit",
    "turn_id",
]
