"""Shared agent seams: typed state, model interface, grounding + capability contracts.

These are the stable interfaces every agent module builds against (doc 50). Defining
them in one place lets the graph, grounding, capabilities, tools, and deliverables be
authored independently without circular imports. Implementations live in sibling
modules; this file holds only the contracts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol, TypedDict, runtime_checkable

from pydantic import BaseModel

# --- run outcome (OUTCOME-R1, closed union) ---


class RunStatus(StrEnum):
    """Closed status-discriminated run outcome (OUTCOME-R1)."""

    COMPLETED = "completed"
    AWAITING_APPROVAL = "awaiting_approval"
    DEGRADED = "degraded"
    BUDGET_EXCEEDED = "budget_exceeded"


Trigger = Literal["user_turn", "scheduled_digest", "scheduled_briefing"]


# --- typed graph state (STATE-R1..R6) ---

# STATE-R6 bounds: the ``retrieved`` channel is serialized into every checkpoint, so it
# MUST stay bounded. On overflow the keyed-merge reducer drops the lowest-relevance
# records (relevance assigned by ``gather`` onto each record) and records the drop so the
# truncation surfaces in ``coverage_gaps`` (the gather node reads the marker back).
RETRIEVED_MAX_RECORDS = 64
RETRIEVED_MAX_BYTES = 256 * 1024
# Marker key carried INSIDE the retrieved dict recording that the reducer trimmed; the
# gather node lifts it into ``coverage_gaps`` (a binop reducer cannot reach that field).
RETRIEVED_TRUNCATION_KEY = "__truncated__"

# --- turn-keying (CKPT-R5; the run-scoped self-reset backstop) ---
#
# TURN-KEYING INVARIANT (read before touching ``retrieved``/``coverage_gaps``):
# A langgraph channel reducer only sees ``(stored, incoming)`` — it cannot reach
# ``turn_id`` from sibling channels. So the *run-scoped accumulators* carry their owning
# turn id IN-BAND, under a reserved double-underscore key (mirroring
# ``RETRIEVED_TRUNCATION_KEY``). Their reducers (``_turn_keyed_merge`` / ``_turn_keyed_union``)
# compare the incoming turn marker to the stored one: a DIFFERENT marker REPLACES (resets)
# the whole channel; the SAME marker merges exactly as before (``_keyed_merge`` /
# ``_set_union`` semantics). This is the backstop against an evidence LEAK across turns — a
# missed head-node reset cannot silently carry turn-1 records into turn-2, because the first
# turn-2 write (carrying turn-2's marker) drops the stale value at the reducer.
#
# Every write to these two channels MUST be stamped with the current ``turn_id`` via
# ``stamp_retrieved`` / ``stamp_coverage_gaps`` (the head node and every node that writes
# evidence). An UNSTAMPED write inherits the stored turn (no reset) and merges — safe within
# a turn, but it forfeits the cross-turn reset, so always stamp.
#
# Do NOT add a plain-accumulator channel (``_append``/``_keyed_merge``/``_set_union`` without
# turn-keying) for anything that must reset per turn — it would silently reintroduce the leak
# this invariant exists to prevent.
RETRIEVED_TURN_KEY = "__turn__"
# Reserved token prefix carrying the owning turn id inside the ``coverage_gaps`` set.
COVERAGE_GAPS_TURN_PREFIX = "__turn__:"


def _append(left: list[Any], right: list[Any]) -> list[Any]:
    """messages reducer: append-only (STATE-R3)."""
    return [*left, *right]


def _record_relevance(record: Any) -> float:
    """The gather-assigned relevance of a retrieved record (default 0.0; STATE-R6).

    ``gather`` stamps a numeric ``relevance`` onto each record so the reducer can rank and
    drop the least-relevant on overflow. An un-stamped record sorts as least relevant.
    """
    if isinstance(record, dict):
        rel = record.get("relevance")
        if isinstance(rel, (int, float)) and not isinstance(rel, bool):
            return float(rel)
    return 0.0


def _serialized_size(records: dict[str, Any]) -> int:
    """Best-effort serialized byte size of the retrieved dict (STATE-R6 size bound)."""
    try:
        return len(json.dumps(records, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return len(str(records).encode("utf-8"))


def _keyed_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """retrieved reducer: merge by canonical id, bounded by count + size (STATE-R3/R6).

    Later records replace earlier ones by key. On exceeding ``RETRIEVED_MAX_RECORDS`` or
    ``RETRIEVED_MAX_BYTES`` the lowest-relevance records are dropped (relevance assigned by
    ``gather``); the number dropped is recorded under ``RETRIEVED_TRUNCATION_KEY`` so the
    ``gather`` node can surface the truncation into ``coverage_gaps`` (STATE-R6).
    """
    merged = {**left, **right}
    prior = int(merged.pop(RETRIEVED_TRUNCATION_KEY, 0) or 0)
    payload = {k: v for k, v in merged.items() if k != RETRIEVED_TRUNCATION_KEY}
    payload, dropped = _enforce_retrieved_bounds(payload)
    total_dropped = prior + dropped
    if total_dropped:
        payload[RETRIEVED_TRUNCATION_KEY] = total_dropped
    return payload


def _enforce_retrieved_bounds(records: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Drop lowest-relevance records until within the count + size bounds (STATE-R6)."""
    ranked = sorted(records.items(), key=lambda kv: _record_relevance(kv[1]), reverse=True)
    if len(ranked) > RETRIEVED_MAX_RECORDS:
        ranked = ranked[:RETRIEVED_MAX_RECORDS]
    kept = dict(ranked)
    while len(kept) > 1 and _serialized_size(kept) > RETRIEVED_MAX_BYTES:
        # Drop the current least-relevant survivor (ranked ascending tail).
        victim = min(kept, key=lambda k: _record_relevance(kept[k]))
        del kept[victim]
    return kept, len(records) - len(kept)


def _set_union(left: set[str], right: set[str]) -> set[str]:
    """coverage_gaps reducer: set-union (STATE-R3)."""
    return left | right


# --- turn-keyed self-resetting accumulators (CKPT-R5 leak backstop) ---


def stamp_retrieved(turn_id: str, records: dict[str, Any]) -> dict[str, Any]:
    """Stamp a ``retrieved`` update with the owning turn id (CKPT-R5 turn-keying).

    Every write to the ``retrieved`` channel MUST go through this so :func:`_turn_keyed_merge`
    can reset across a turn boundary. The marker rides under :data:`RETRIEVED_TURN_KEY`
    in-band (like :data:`RETRIEVED_TRUNCATION_KEY`); readers strip it via :func:`turn_records`.
    """
    payload = {k: v for k, v in records.items() if k != RETRIEVED_TURN_KEY}
    return {**payload, RETRIEVED_TURN_KEY: turn_id}


def turn_records(stored: dict[str, Any]) -> dict[str, Any]:
    """The ``retrieved`` payload with the turn marker stripped (reader helper)."""
    return {k: v for k, v in stored.items() if k != RETRIEVED_TURN_KEY}


def _turn_keyed_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """``retrieved`` reducer: merge within a turn, REPLACE across a turn (CKPT-R5).

    Carries the owning ``turn_id`` in-band under :data:`RETRIEVED_TURN_KEY`. If the incoming
    update names a turn DIFFERENT from the stored one, the prior turn's records are dropped
    (self-reset) and only the incoming records survive; otherwise the merge is exactly
    :func:`_keyed_merge` (later-by-key wins, bounded by count + size). An unstamped incoming
    update inherits the stored turn and merges (safe within a turn). This reducer is the
    backstop that makes a missed head-node reset non-leaking — turn-2's first stamped write
    discards turn-1 evidence here, at the channel boundary.
    """
    incoming_turn = right.get(RETRIEVED_TURN_KEY)
    stored_turn = left.get(RETRIEVED_TURN_KEY)
    if incoming_turn is not None and incoming_turn != stored_turn:
        base: dict[str, Any] = {}
        carry_turn = incoming_turn
    else:
        base = {k: v for k, v in left.items() if k != RETRIEVED_TURN_KEY}
        carry_turn = incoming_turn if incoming_turn is not None else stored_turn
    merged = _keyed_merge(base, {k: v for k, v in right.items() if k != RETRIEVED_TURN_KEY})
    if carry_turn is not None:
        merged[RETRIEVED_TURN_KEY] = carry_turn
    return merged


def stamp_coverage_gaps(turn_id: str, gaps: set[str]) -> set[str]:
    """Stamp a ``coverage_gaps`` update with the owning turn id (CKPT-R5 turn-keying).

    Every write to ``coverage_gaps`` MUST go through this so :func:`_turn_keyed_union` can
    reset across a turn boundary. The marker rides as a reserved
    :data:`COVERAGE_GAPS_TURN_PREFIX` token inside the set; readers strip it via
    :func:`turn_gaps`.
    """
    real = {g for g in gaps if not g.startswith(COVERAGE_GAPS_TURN_PREFIX)}
    return {f"{COVERAGE_GAPS_TURN_PREFIX}{turn_id}", *real}


def turn_gaps(stored: set[str]) -> set[str]:
    """The ``coverage_gaps`` payload with the turn marker stripped (reader helper)."""
    return {g for g in stored if not g.startswith(COVERAGE_GAPS_TURN_PREFIX)}


def _turn_marker(values: set[str]) -> str | None:
    """The owning turn id encoded in a ``coverage_gaps`` set, if any."""
    for value in values:
        if value.startswith(COVERAGE_GAPS_TURN_PREFIX):
            return value[len(COVERAGE_GAPS_TURN_PREFIX) :]
    return None


def _turn_keyed_union(left: set[str], right: set[str]) -> set[str]:
    """``coverage_gaps`` reducer: union within a turn, REPLACE across a turn (CKPT-R5).

    Mirrors :func:`_turn_keyed_merge` for the gap set: the owning ``turn_id`` rides as a
    :data:`COVERAGE_GAPS_TURN_PREFIX` token. An incoming update naming a DIFFERENT turn drops
    the prior turn's gaps (self-reset); a same-turn (or unstamped) update unions as before.
    """
    incoming_turn = _turn_marker(right)
    stored_turn = _turn_marker(left)
    base = set() if incoming_turn is not None and incoming_turn != stored_turn else turn_gaps(left)
    merged = base | turn_gaps(right)
    carry_turn = incoming_turn if incoming_turn is not None else stored_turn
    if carry_turn is not None:
        merged.add(f"{COVERAGE_GAPS_TURN_PREFIX}{carry_turn}")
    return merged


def _monotonic(left: int, right: int) -> int:
    """count reducer: strictly non-decreasing (STATE-R3); raises on a decrease."""
    if right < left:
        raise ValueError("reflection/redraft counters are monotonic (STATE-R3)")
    return right


# Sentinel floor the head node may reset a run-scoped counter to on a NEW turn (CKPT-R5).
TURN_COUNTER_FLOOR = 0


def _turn_monotonic(left: int, right: int) -> int:
    """Run-scoped counter reducer: monotonic WITHIN a turn, resettable to 0 at a turn boundary.

    EVAL-R7 bounded-termination needs the counter to be strictly non-decreasing *inside* a
    turn (so a reflect/redraft/visit loop cannot evade its budget by rewinding). But a durable
    thread reuses the same checkpoint across turns, so the single head node MUST be able to
    reset the counter at a new-turn boundary (CKPT-R5) — otherwise the next turn's first write
    of ``count=1`` decreases below the stored ``count=N`` and the strict ``_monotonic`` guard
    raises (the cross-turn force-degrade bug).

    The compromise: allow an increase (or no change), and allow a decrease ONLY to the
    sentinel floor :data:`TURN_COUNTER_FLOOR` (``0``) — the head node's reset write. Any OTHER
    decrease (e.g. ``5 -> 3``) is a mid-turn rewind and still raises. Single writer of the
    floor = the head node on a new turn, so a node cannot smuggle a mid-turn reset.
    """
    if right == TURN_COUNTER_FLOOR:
        return TURN_COUNTER_FLOOR
    if right < left:
        raise ValueError("run-scoped counter decreased mid-turn (EVAL-R7); only reset-to-0 allowed")
    return right


def _last_write_wins(left: str, right: str) -> str:
    """``run_epoch`` reducer: last non-empty write wins (CKPT-R5 turn boundary).

    The single writer is the head node, which stamps ``run_epoch = turn_id`` when it resets
    the run-scoped channels on a new turn. An empty incoming write (no update) keeps the prior
    epoch. Explicit reducer (over langgraph's implicit last-value default) so the turn-boundary
    intent is documented and the channel reads with a typed binop like its siblings.
    """
    return right if right else left


def _write_once(left: str, right: str) -> str:
    """Write-once reducer for immutable identity inputs (STATE-R4 / AGT-SEC-R1).

    The first non-empty write sets the value; any later write of a DIFFERENT non-empty
    value is rejected at the reducer level (raises). A node-, model-, or tool-produced
    attempt to overwrite ``athlete_id``/``idempotency_key`` therefore fails closed rather
    than silently winning under langgraph's default last-value-wins channel. langgraph's
    ``BinaryOperatorAggregate`` seeds a ``str`` channel with ``""``, which is treated as
    the unset sentinel.
    """
    if not left:
        return right
    if right and right != left:
        raise ValueError("write-once identity field changed (STATE-R4 / AGT-SEC-R1)")
    return left


class AgentState(TypedDict, total=False):
    """The agent graph's typed serializable state (STATE-R1/R2).

    Immutable inputs are write-once (STATE-R4); accumulating fields carry the reducers
    above; outputs are filled by ``compose``/``ground``/``finalize``. ``athlete_id`` is
    server-derived only (AGT-SEC-R1) and never set by a model/tool output.

    TURN-KEYING (CKPT-R5; the durable-resume run-scoped reset, read before adding a channel):
    a durable thread reuses ONE checkpoint across many turns. Channels in group (c) below are
    RUN-SCOPED — they must reset at each new-turn boundary, NOT accumulate forever. ``turn_id``
    is minted fresh for each normal ``/ask`` ``ainvoke`` (a ``Command(resume)`` NEVER mints or
    changes it, since the head node does not run on resume); ``run_epoch`` is the turn the
    run-scoped channels currently belong to (last-write-wins, written by the head node when it
    resets). The three counters use :func:`_turn_monotonic` (monotonic within a turn, but the
    head node may reset to ``0`` at a boundary). ``retrieved`` and ``coverage_gaps`` use the
    TURN-KEYED reducers (:func:`_turn_keyed_merge` / :func:`_turn_keyed_union`): they carry
    their owning turn id in-band and self-reset when an incoming write names a different turn —
    the leak backstop. Any NEW run-scoped channel MUST use a turn-keyed/turn-monotonic reducer;
    a plain ``_append``/``_keyed_merge``/``_set_union`` channel would silently leak turn-1 data
    into turn-2. See the turn-keying invariant comment near :data:`RETRIEVED_TURN_KEY`.
    """

    # (a) immutable inputs (write-once, STATE-R4)
    athlete_id: Annotated[str, _write_once]
    trigger: Trigger
    request_text: str | None
    briefing_screen: str | None
    locale: str
    idempotency_key: Annotated[str, _write_once]
    thread_id: str | None
    response_length: str | None
    coach_numeric_detail_level: int | None
    # The athlete's ACTIVE canonical goals, read SERVER-side from the GBO store and projected into
    # the run inputs so the agent plans TOWARD them (GBO-R38 / API-R32 / API-R35): goal-aware
    # planning/load-review is owned by the agent, which reads the canonical Goal entity. Each item
    # is a plain serializable projection (title/goal_type/sport/target_*/status) — user-authored
    # INTENT, never a canonical analytic NUMBER (MEM-R1), so it steers the compose prompt context
    # but is NOT a grounding fact. An immutable input (set once by ingest, never by a model/tool).
    active_goals: list[dict[str, Any]]
    # Durable athlete-memory items recalled SERVER-side through the ONE MemoryStore/recall seam
    # (MEM-R4) and projected into the run inputs so the agent personalizes its answer (stated
    # goals/constraints/preferences/load-responses in the athlete's own words, MEM-R1/-R2). Each
    # item is a plain serializable projection (kind/content/inferred) — personalization context,
    # NEVER a canonical analytic NUMBER (MEM-R1, EVAL-R2a), so it steers the compose prompt but is
    # NOT a grounding fact (the §7 grounder still reads every number LIVE from canonical analytics).
    # An immutable input (recalled once by the engine before the run, never written by a model).
    recalled_memory: list[dict[str, Any]]
    # (b) per-turn identity for the run-scoped reset (CKPT-R5)
    #   turn_id: fresh per normal /ask; never minted/changed on Command(resume).
    #   run_epoch: the turn the run-scoped channels belong to; head node sets it on reset.
    turn_id: str
    run_epoch: Annotated[str, _last_write_wins]
    # (c) accumulating working memory — RUN-SCOPED, resets each turn (see TURN-KEYING above)
    messages: Annotated[list[dict[str, Any]], _append]
    retrieved: Annotated[dict[str, Any], _turn_keyed_merge]
    coverage_gaps: Annotated[set[str], _turn_keyed_union]
    reflection_count: Annotated[int, _turn_monotonic]
    redraft_count: Annotated[int, _turn_monotonic]
    node_visits: Annotated[int, _turn_monotonic]
    # The tool-iteration counter (AGT-ENT-R4 tool-iteration guard): incremented by ``gather`` each
    # time it actually resolves planned capability requests. RUN-SCOPED (resets per turn) and uses
    # the SAME ``_turn_monotonic`` reducer as ``node_visits``/``reflection_count``: monotonic within
    # a turn (a tool loop cannot evade its budget by rewinding), resettable only to the floor 0 by
    # the head node at a turn boundary. When it reaches the resolved entitlement's
    # ``max_tool_iterations`` the routers stop re-planning and route to compose — a graceful bound
    # on the gather/tool loop independent of ``node_visits`` (AGT-ENT-R1, read from entitlement).
    tool_iterations: Annotated[int, _turn_monotonic]
    cost_events: Annotated[list[dict[str, Any]], _append]
    # (c) outputs
    draft: str | None
    grounded_html: str | None
    grounded_text: str | None
    observations: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    status: RunStatus
    coverage_caveat: dict[str, Any] | None
    interrupt_id: str | None
    cost_rollup: dict[str, Any] | None


# --- model-routing seam (MODEL-R*, STRUCT-R*) ---


@runtime_checkable
class ChatModel(Protocol):
    """One OpenAI-compatible model behind a typed seam (MODEL-R4).

    ``structured`` does provider-enforced JSON-schema-constrained decoding for a
    verdict (STRUCT-R1); ``compose`` does bounded-temperature prose. A model that
    cannot enforce structured output MUST NOT back a verdict node.
    """

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M: ...

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str: ...


# --- grounding contract (GROUND-R*) ---


class ClaimKind(StrEnum):
    NUMBER = "number"
    NAME = "name"
    URL = "url"
    STATEMENT = "statement"


@dataclass(frozen=True, slots=True)
class Claim:
    """One extracted candidate claim to be verified by deterministic code (STRUCT-R5).

    ``workout_type`` is the LANGUAGE-INDEPENDENT canonical workout type the model emits as a
    typed enum on a prescribed-workout NAME claim (COACH-R2: "a typed prescription the canonical
    model can represent"; STRUCT-R1 plan-structure verdict). A plan written in ANY language carries
    the SAME structured type — so grounding checks the type, never the translated surface name. It
    is ``None`` for non-prescription claims and for an older extractor that emits no type (the
    grounder then falls back to the surface-name match, preserving prior behaviour).
    """

    kind: ClaimKind
    text: str
    metric: str | None = None
    value: float | None = None
    ref: str | None = None
    prescriptive: bool = False
    workout_type: str | None = None


class GroundVerdict(StrEnum):
    """Per-claim grounding verdict (GROUND-R9)."""

    GROUNDED = "grounded"
    UNGROUNDED = "ungrounded"
    CONTRADICTED = "contradicted"
    COMPLEMENTARY = "complementary"


class GroundDecision(StrEnum):
    """Aggregate bounded recovery decision (GROUND-R9)."""

    PROCEED = "proceed"
    REGENERATE = "regenerate"
    REPLAN = "replan"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class GroundedClaim:
    claim: Claim
    verdict: GroundVerdict
    citation: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class GroundingResult:
    """Result of deterministic grounding over a draft's claims (GROUND-R1/R3/R9)."""

    decision: GroundDecision
    claims: tuple[GroundedClaim, ...]
    scrubbed_text: str

    @property
    def survivors(self) -> tuple[GroundedClaim, ...]:
        return tuple(c for c in self.claims if c.verdict is GroundVerdict.GROUNDED)


@runtime_checkable
class GroundingEvidence(Protocol):
    """Read-only canonical evidence the grounder verifies claims against (GROUND-R2/R7).

    Numbers come VERBATIM from the canonical analytics service; the grounder matches a
    claimed value against the canonical computation within tolerance and scrubs anything
    unmatched (GROUND-R3, "when in doubt, scrub").
    """

    async def metric_value(self, metric: str, as_of: str | None) -> float | None: ...

    def url_allowed(self, url: str) -> bool: ...


# --- capability registry (PLAN-R*, TOOL-R1) ---


@dataclass(frozen=True, slots=True)
class Capability:
    """One model-facing capability mapping 1:1 to a canonical-service call (PLAN-R3).

    The SAME registry backs the planner's structured plan and the MCP tool layer; both
    resolve to ``service_method`` on the analytics/canonical service (one data path).
    """

    key: str
    description: str
    service_method: str
    param_schema: type[BaseModel]


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """A planner-selected capability request with typed params (PLAN-R2)."""

    capability: str
    params: dict[str, Any]


# --- reflect verdict (REFLECT-R2, closed enum, STRUCT-R1) ---


class ReflectVerdict(StrEnum):
    """Closed reflect decision over the §6 verdict set (REFLECT-R2)."""

    REPLAN = "replan"
    ANSWER_WITH_CAVEAT = "answer_with_caveat"
    GIVE_UP_GRACEFULLY = "give_up_gracefully"


class ReflectDecision(BaseModel):
    """A provider-enforced structured reflect verdict (REFLECT-R2 / STRUCT-R1).

    ``verdict`` is the closed §6 decision; ``add_requests`` names the capability keys a
    ``replan`` should add/widen (considering the open coverage gaps). It is obtained via
    ``run_structured`` so the verdict is schema-constrained, never free-text parsed.
    """

    verdict: ReflectVerdict
    rationale: str = ""
    add_requests: tuple[str, ...] = ()


# --- typed coverage caveat (OUTCOME-R4, structured not prose) ---


class CoverageCaveat(BaseModel):
    """Source-agnostic typed coverage caveat for a degraded outcome (OUTCOME-R4).

    Names — in source-agnostic terms — which canonical inputs were missing/substituted/
    stale and the resulting fidelity, rather than leaking raw coverage-internal tokens.
    """

    missing: tuple[str, ...] = ()
    substituted: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    fidelity: Literal["full", "partial", "degraded"] = "partial"


@dataclass(slots=True)
class CoachConfig:
    """Loaded coach persona/voice + grounding/abstention refs (COACH-CFG-R*).

    DATA the engine consumes; the OSS default is one persona. Prompt/voice text loads
    from external config, never inline (QUAL-R13 / DELIV-R2).
    """

    name: str = "default"
    system_prompt: str = ""
    voice_examples: Sequence[str] = field(default_factory=tuple)


__all__ = [
    "COVERAGE_GAPS_TURN_PREFIX",
    "RETRIEVED_MAX_BYTES",
    "RETRIEVED_MAX_RECORDS",
    "RETRIEVED_TRUNCATION_KEY",
    "RETRIEVED_TURN_KEY",
    "TURN_COUNTER_FLOOR",
    "AgentState",
    "Capability",
    "ChatModel",
    "Claim",
    "ClaimKind",
    "CoachConfig",
    "CoverageCaveat",
    "GroundDecision",
    "GroundVerdict",
    "GroundedClaim",
    "GroundingEvidence",
    "GroundingResult",
    "ReflectDecision",
    "ReflectVerdict",
    "RetrievalRequest",
    "RunStatus",
    "Trigger",
    "stamp_coverage_gaps",
    "stamp_retrieved",
    "turn_gaps",
    "turn_records",
]
