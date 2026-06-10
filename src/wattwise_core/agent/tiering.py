"""The model-tier / reasoning-effort routing-policy seam (doc 50 §11, MODEL-R1/-R2/-R2b).

This LEAF module owns the OSS-shipped half of model tiering: the fixed
``{flash, pro, frontier} x {low, medium, high}`` taxonomy (MODEL-R1 — owned by doc 50; doc 60's
enums must match it exactly) and the typed **model-routing-policy seam** (``SEAM-R8`` /
MODEL-R2b) through which tier/effort are chosen PER NODE from a typed difficulty signal
(MODEL-R2). It depends only on the closed contracts (ARCH-R21) and the observability logger.

What ships in OSS vs commercially (COMM-R20):

* OSS ships the TAXONOMY, the SEAM (:class:`ModelRoutingPolicy`), and the single default
  policy (:class:`SingleModelRoutingPolicy`): every ``route`` call resolves deterministically
  to the ONE configured model at the configured tier/effort labels — no escalation, no
  failover, no budget. The tier/effort labels it carries are loaded config
  (``agent__tier`` / ``agent__reasoning_effort``, CFG-R1a), never code literals.
* The task-aware multi-tier SELECTION (mapping a node/difficulty signal to a non-default
  tier/effort or a different model) is COMMERCIAL: a richer policy plugs in through this SAME
  typed Protocol without touching node logic (MODEL-R4). When a plugged-in policy escalates
  (returns a decision marked ``escalated``), the graph-side helper :func:`routed_model` LOGS
  the explicit decision with its recorded reason (MODEL-R2 "an explicit, logged decision").

Model/tier selection is SERVER-side only (MODEL-R2b): nothing here reads athlete input, and no
athlete-facing surface exposes or accepts a tier/model name.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from wattwise_core.agent.contracts import ChatModel
from wattwise_core.observability import metrics as obs_metrics
from wattwise_core.observability.logging import get_logger

_logger = get_logger(__name__)


class ModelTier(StrEnum):
    """The fixed three-tier taxonomy (MODEL-R1; doc 60's enum must match exactly)."""

    FLASH = "flash"
    PRO = "pro"
    FRONTIER = "frontier"


class ReasoningEffort(StrEnum):
    """The orthogonal reasoning-effort axis (MODEL-R1)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class DifficultySignal:
    """The typed difficulty signal routing is deterministic over (MODEL-R2).

    Carries the run-state facts a policy may key tier/effort decisions on: the intent class
    (where one is fixed by the trigger), the reflection state (how many self-correction cycles
    have been spent), and the plan complexity (how many evidence records are in play). All are
    server-derived run state — never athlete-supplied (MODEL-R2b).
    """

    intent: str | None = None
    reflection_count: int = 0
    plan_complexity: int = 0


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """One deterministic routing decision for one node call (MODEL-R2).

    ``escalated`` marks a decision above the policy's default path (a non-``flash`` tier or a
    raised effort); the graph logs such a decision explicitly with its ``reason`` (MODEL-R2).
    The OSS default policy never sets it.
    """

    tier: ModelTier
    effort: ReasoningEffort
    reason: str
    escalated: bool = False


@runtime_checkable
class ModelRoutingPolicy(Protocol):
    """The typed model-routing-policy seam (``SEAM-R8`` / MODEL-R2b; ships in OSS).

    ``route`` MUST be deterministic given ``(node, signal)`` (MODEL-R2). ``resolve`` maps a
    decision to the concrete model for the call, given the run's one configured model
    (MODEL-R4): the OSS default returns that model for EVERY tier (the §11 OSS policy "always
    the one configured model"); a commercial policy may return a different per-tier model
    through this same seam without an agent-graph change (COMM-R20).
    """

    def route(self, *, node: str, signal: DifficultySignal) -> RouteDecision: ...

    def resolve(self, decision: RouteDecision, default: ChatModel) -> ChatModel: ...


class SingleModelRoutingPolicy:
    """The OSS default routing policy: every tier resolves to the one configured model.

    Deterministic and escalation-free (MODEL-R1/-R2 OSS default): every ``(node, signal)``
    routes to the configured tier/effort labels — the default path is the ``flash`` tier at
    ``low`` effort unless config says otherwise (CFG-R1a: the labels are loaded content) — and
    ``resolve`` returns the run's one configured model unchanged (MODEL-R4). Invalid configured
    labels fail closed at construction (never a silently-coerced tier).
    """

    __slots__ = ("_effort", "_tier")

    def __init__(
        self,
        *,
        tier: ModelTier = ModelTier.FLASH,
        effort: ReasoningEffort = ReasoningEffort.LOW,
    ) -> None:
        self._tier = tier
        self._effort = effort

    @classmethod
    def from_labels(cls, tier: str, effort: str) -> SingleModelRoutingPolicy:
        """Build from the config-loaded label strings, failing closed on a non-member value."""
        return cls(tier=ModelTier(tier), effort=ReasoningEffort(effort))

    def route(self, *, node: str, signal: DifficultySignal) -> RouteDecision:
        """Deterministically route every node to the single configured tier/effort."""
        return RouteDecision(
            tier=self._tier,
            effort=self._effort,
            reason="oss-default-single-model",
            escalated=False,
        )

    def resolve(self, decision: RouteDecision, default: ChatModel) -> ChatModel:
        """Every tier resolves to the one configured model (MODEL-R4, OSS default)."""
        return default


def routed_model(
    policy: ModelRoutingPolicy,
    default: ChatModel,
    *,
    node: str,
    signal: DifficultySignal,
) -> ChatModel:
    """Route one node call through the policy seam and return the model to use (MODEL-R2).

    The single graph-side entry point for per-node tier/effort selection: obtains the
    deterministic :class:`RouteDecision` for ``(node, signal)`` and resolves it to the concrete
    model. An ESCALATED decision (a plugged-in commercial policy choosing above its default
    path) is logged explicitly with its recorded reason (MODEL-R2 "an explicit, logged
    decision"); the OSS default policy never escalates, so the default path stays silent.
    """
    decision = policy.route(node=node, signal=signal)
    if decision.escalated:
        # AGT-OBS-R5: a tier escalation is an explicit, logged decision with its reason,
        # counted in metrics — never surfaced to the athlete (§13a).
        _logger.info(
            "model tier escalation",
            node=node,
            tier=decision.tier.value,
            reasoning_effort=decision.effort.value,
            reason=decision.reason,
        )
        obs_metrics.get_registry().increment(
            obs_metrics.TIER_ESCALATIONS,
            labels={"node": node, "tier": decision.tier.value},
        )
    return policy.resolve(decision, default)


# --- the MODEL-R3 token-counter seam (lives with the model-routing seam) -----------------

#: The injectable token-counter seam (MODEL-R3): a callable measuring a text in the target
#: model's tokens. The compose node injects the MODEL's own counter when the model seam
#: exposes one (``count_tokens``); :func:`estimate_tokens` is the deterministic fallback for a
#: model that exposes none (the OSS engine ships no provider tokenizer offline).
TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate (MODEL-R3 fallback counter; ~4 chars/token, words floor).

    Used ONLY when the model seam exposes no real tokenizer (``count_tokens``); a deployment's
    model that does is injected through the same :data:`TokenCounter` seam by the compose node.
    """
    return max(len(text) // 4, len(text.split()))


def model_token_counter(model: ChatModel) -> TokenCounter:
    """The MODEL-R3 token counter for a model: its own tokenizer when exposed, else the estimate.

    The seam is structural: a model exposing a callable ``count_tokens(text) -> int`` (the real
    tokenizer for its vocabulary) measures the assembled context; otherwise the deterministic
    estimator bounds it — no character heuristic is used when a real tokenizer is available.
    """
    counter = getattr(model, "count_tokens", None)
    return counter if callable(counter) else estimate_tokens


def context_budget(window: int | None, output_headroom: int | None) -> int | None:
    """The MODEL-R3 compose INPUT budget: context window minus the reserved output headroom.

    Computed from the config-loaded ``agent__context_window_tokens`` and the run's resolved
    OUTPUT-token bound (the entitlement's ``max_output_tokens``, AGT-ENT-R1), so the assembled
    input always reserves headroom for the structured output + a reasoning model's thinking
    trace (MODEL-R5a). ``None`` when no window is configured (a direct caller) — the graph's
    module fallback bounds it. Floored at a small positive bound so a misconfigured pair never
    yields a non-positive budget.
    """
    if window is None or window <= 0:
        return None
    return max(window - (output_headroom or 0), 1024)


__all__ = [
    "DifficultySignal",
    "ModelRoutingPolicy",
    "ModelTier",
    "ReasoningEffort",
    "RouteDecision",
    "SingleModelRoutingPolicy",
    "TokenCounter",
    "context_budget",
    "estimate_tokens",
    "model_token_counter",
    "routed_model",
]
