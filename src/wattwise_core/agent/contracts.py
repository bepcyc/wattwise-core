"""Shared agent seams: typed state, model interface, grounding + capability contracts.

These are the stable interfaces every agent module builds against (doc 50). Defining
them in one place lets the graph, grounding, capabilities, tools, and deliverables be
authored independently without circular imports. Implementations live in sibling
modules; this file holds only the contracts.
"""

from __future__ import annotations

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


def _append(left: list[Any], right: list[Any]) -> list[Any]:
    """messages reducer: append-only (STATE-R3)."""
    return [*left, *right]


def _keyed_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """retrieved reducer: merge by canonical record id; later replaces earlier (STATE-R3)."""
    return {**left, **right}


def _set_union(left: set[str], right: set[str]) -> set[str]:
    """coverage_gaps reducer: set-union (STATE-R3)."""
    return left | right


def _monotonic(left: int, right: int) -> int:
    """count reducer: strictly non-decreasing (STATE-R3); raises on a decrease."""
    if right < left:
        raise ValueError("reflection/redraft counters are monotonic (STATE-R3)")
    return right


class AgentState(TypedDict, total=False):
    """The agent graph's typed serializable state (STATE-R1/R2).

    Immutable inputs are write-once (STATE-R4); accumulating fields carry the reducers
    above; outputs are filled by ``compose``/``ground``/``finalize``. ``athlete_id`` is
    server-derived only (AGT-SEC-R1) and never set by a model/tool output.
    """

    # (a) immutable inputs
    athlete_id: str
    trigger: Trigger
    request_text: str | None
    briefing_screen: str | None
    locale: str
    idempotency_key: str
    # (b) accumulating working memory
    messages: Annotated[list[dict[str, Any]], _append]
    retrieved: Annotated[dict[str, Any], _keyed_merge]
    coverage_gaps: Annotated[set[str], _set_union]
    reflection_count: Annotated[int, _monotonic]
    redraft_count: Annotated[int, _monotonic]
    # (c) outputs
    draft: str | None
    grounded_html: str | None
    grounded_text: str | None
    observations: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    status: RunStatus
    coverage_caveat: dict[str, Any] | None


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
    "AgentState",
    "Capability",
    "ChatModel",
    "Claim",
    "ClaimKind",
    "CoachConfig",
    "GroundDecision",
    "GroundVerdict",
    "GroundedClaim",
    "GroundingEvidence",
    "GroundingResult",
    "RetrievalRequest",
    "RunStatus",
    "Trigger",
]
