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


def _monotonic(left: int, right: int) -> int:
    """count reducer: strictly non-decreasing (STATE-R3); raises on a decrease."""
    if right < left:
        raise ValueError("reflection/redraft counters are monotonic (STATE-R3)")
    return right


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
    # (b) accumulating working memory
    messages: Annotated[list[dict[str, Any]], _append]
    retrieved: Annotated[dict[str, Any], _keyed_merge]
    coverage_gaps: Annotated[set[str], _set_union]
    reflection_count: Annotated[int, _monotonic]
    redraft_count: Annotated[int, _monotonic]
    node_visits: Annotated[int, _monotonic]
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

    async def structured[M: BaseModel](
        self, *, system: str, data: str, schema: type[M]
    ) -> M: ...

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str: ...


# --- grounding contract (GROUND-R*) ---


class ClaimKind(StrEnum):
    NUMBER = "number"
    NAME = "name"
    URL = "url"
    STATEMENT = "statement"


@dataclass(frozen=True, slots=True)
class Claim:
    """One extracted candidate claim to be verified by deterministic code (STRUCT-R5)."""

    kind: ClaimKind
    text: str
    metric: str | None = None
    value: float | None = None
    ref: str | None = None
    prescriptive: bool = False


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
    "RETRIEVED_MAX_BYTES",
    "RETRIEVED_MAX_RECORDS",
    "RETRIEVED_TRUNCATION_KEY",
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
]
