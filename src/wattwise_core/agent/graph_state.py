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
from collections.abc import Mapping
from typing import Any

from wattwise_core.agent.contracts import (
    RETRIEVED_MAX_RECORDS,
    RETRIEVED_TRUNCATION_KEY,
    AgentState,
    ChatModel,
    CoverageCaveat,
    GroundDecision,
    ReflectDecision,
    ReflectVerdict,
    RetrievalRequest,
    RunStatus,
)
from wattwise_core.agent.structured import StructuredOutputError, run_structured

# Bounded recovery budgets (REFLECT-R4), shared with the graph for routing decisions.
MAX_REFLECTIONS = 2
MAX_REDRAFTS = 2


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
    """Surface a STATE-R6 retrieved-truncation as a coverage gap, once (pure read)."""
    prior = state.get("retrieved", {})
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

_REFLECT_SYSTEM = (
    "You are the coaching agent's reflection step. Given the open coverage gaps and "
    "what has already been retrieved, decide the next move over the CLOSED verdict set "
    "{replan, answer_with_caveat, give_up_gracefully}. Choose replan only when adding or "
    "widening capability requests could close a gap; answer_with_caveat when a useful "
    "grounded answer is possible despite a gap; give_up_gracefully when nothing more can "
    "be retrieved. For replan, list the capability keys to add/widen."
)


async def reflect_decision(model: ChatModel, state: AgentState) -> ReflectDecision:
    """Obtain the structured reflect verdict, fail-closed on a structured-output error."""
    gaps = sorted(state.get("coverage_gaps", set()))
    already = sorted(state.get("retrieved", {}).keys())
    data = f"open_gaps: {gaps}\nalready_retrieved: {already}"
    try:
        return await run_structured(
            model, system=_REFLECT_SYSTEM, data=data, schema=ReflectDecision
        )
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
    request_text: str | None, retrieved: Mapping[str, Any]
) -> tuple[str, bool]:
    """Serialise canonical evidence within a measured token budget (MODEL-R3, INJECT-R1).

    Untrusted content (the user request and every retrieved record body) is wrapped in an
    explicit delimited ``<untrusted-data>`` envelope so injected instructions in
    titles/notes/tool bodies cannot read as instructions (INJECT-R1). The assembled input
    is measured with the token estimator; on overflow the lowest-relevance records are
    dropped FIRST and a flag is returned so the caller records the trim in coverage_gaps.
    Returns ``(context, trimmed)``.
    """
    header = f"request:\n<untrusted-data>\n{request_text or ''}\n</untrusted-data>"
    items = sorted(
        ((k, v) for k, v in retrieved.items() if k != RETRIEVED_TRUNCATION_KEY),
        key=lambda kv: context_relevance(kv[1]),
        reverse=True,
    )
    rendered: list[str] = [header]
    trimmed = False
    for key, value in items:
        candidate = f"{key}:\n<untrusted-data>\n{value}\n</untrusted-data>"
        if estimate_tokens("\n".join([*rendered, candidate])) > _CONTEXT_TOKEN_BUDGET:
            trimmed = True
            continue
        rendered.append(candidate)
    return "\n".join(rendered), trimmed


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
    "athlete_id",
    "budget_exceeded",
    "build_caveat",
    "context_relevance",
    "cost_rollup",
    "estimate_tokens",
    "last_ground_decision",
    "last_plan_requests",
    "last_reflect_verdict",
    "limitation_text",
    "open_gaps",
    "over_ceiling",
    "plan_requires_approval",
    "reflect_decision",
    "render_context",
    "retrieved_truncation_gaps",
    "safe_html",
    "terminal_status",
    "tick_visit",
]
