"""The two MODEL-calling spine-node factories: reflect + compose (QUAL-R9 size split).

Factored out of :mod:`wattwise_core.agent.graph` so that module stays under the size ceiling.
These are the only spine nodes that call the chat model directly, and therefore the only ones
that route through the typed model-routing-policy seam (MODEL-R1/-R2: the model for EACH node
call is chosen deterministically from the node + a typed difficulty signal; under the OSS
default policy every tier resolves to the one configured model). ``compose`` additionally owns
the MODEL-R3 measured token budget (the model's own token counter when exposed) and the
LANG-R1/-R3 localized prompt-variant selection at composition time (with the LANG-R4 fallback
recorded). Same purity contract as every node (GRAPH-R4): pure ``(state, injected services) ->
partial update``; no sibling in-flight import beyond the leaf seams (ARCH-R21).
"""

from __future__ import annotations

from typing import Any

from wattwise_core.agent import graph_state as gs
from wattwise_core.agent import tiering
from wattwise_core.agent.contracts import (
    AgentState,
    ChatModel,
    parse_tagged_answer,
    stamp_coverage_gaps,
)
from wattwise_core.agent.locale import LocalePolicy
from wattwise_core.agent.seams import AgentServices, GraphNode
from wattwise_core.observability import metrics as obs_metrics


def make_reflect(
    model: ChatModel, reflect_system: str, routing: tiering.ModelRoutingPolicy
) -> GraphNode:
    """Build the reflect node (REFLECT-R2/-R4, MODEL-R1/-R2)."""

    async def reflect(state: AgentState) -> dict[str, Any]:
        """Emit a structured §6 reflect verdict over the closed enum (REFLECT-R2).

        Spends one unit of the bounded reflection budget (REFLECT-R4) and obtains a
        provider-enforced ``ReflectDecision`` via ``run_structured`` (STRUCT-R1) using the
        externalized reflection system prompt (§16 / SKILL-R1, CFG-R3 — not inline, ARCH-R29);
        the routing function reads the verdict. The model for THIS node call is chosen through
        the typed model-routing-policy seam from the node + a typed difficulty signal
        (MODEL-R1/-R2; under the OSS default policy every tier resolves to the one configured
        model).
        """
        gs.athlete_id(state)
        count = state.get("reflection_count", 0) + 1
        # OBS-R4: count this self-correction iteration on the production metrics surface so the
        # reflection rate / reflection-exhaustion rate are observable (AGT-OBS-R7).
        obs_metrics.get_registry().increment(obs_metrics.REFLECTIONS)
        node_model = tiering.routed_model(
            routing,
            model,
            node="reflect",
            signal=tiering.DifficultySignal(reflection_count=count),
        )
        decision = await gs.reflect_decision(node_model, state, system=reflect_system)
        note = {
            "role": "system",
            "kind": "reflect",
            "reflection_count": count,
            "verdict": decision.verdict.value,
            "add_requests": list(decision.add_requests),
        }
        return gs.tick_visit(state, {"reflection_count": count, "messages": [note]})

    return reflect


def make_compose(
    svc: AgentServices,
    model: ChatModel,
    coach_system: str,
    routing: tiering.ModelRoutingPolicy,
    locales: LocalePolicy,
    context_token_budget: int | None,
    detailed_directive: str = "",
) -> GraphNode:
    """Build the compose node (DELIV-R*, MODEL-R1/-R2/-R3, LANG-R1/-R3/-R4)."""

    async def compose(state: AgentState) -> dict[str, Any]:
        """Draft prose from canonical evidence within a token budget (DELIV-R*, MODEL-R3).

        Untrusted content is wrapped in delimited data envelopes (INJECT-R1); on a context
        overflow the lowest-relevance records are trimmed and the trim is recorded in
        coverage_gaps. The redraft counter is already spent by the router upstream. The model
        is chosen per node through the routing-policy seam (MODEL-R1/-R2); the context is
        measured with the model's token counter against the engine-computed input budget
        (MODEL-R3); and the system prompt layers the run locale's LOADED language variant
        chosen at composition time (LANG-R1/-R3 — one resolved language per deliverable, with
        the LANG-R4 fallback to the config-driven default recorded for observability).
        """
        gs.athlete_id(state)
        retrieved = gs.read_retrieved(state)
        node_model = tiering.routed_model(
            routing,
            model,
            node="compose",
            signal=tiering.DifficultySignal(
                reflection_count=state.get("reflection_count", 0),
                plan_complexity=len(retrieved),
            ),
        )
        context, trimmed = gs.render_context(
            state.get("request_text"),
            retrieved,
            active_goals=state.get("active_goals"),
            recalled_memory=state.get("recalled_memory"),
            token_counter=tiering.model_token_counter(node_model),
            token_budget=context_token_budget,
        )
        system = locales.compose_system(coach_system, state.get("locale"))
        if detailed_directive and state.get("response_length") == "detailed":
            # VOICE-R7/-R8 steering for a DETAILED run: the loaded config fragment telling the
            # model to weave up to the detailed number cap of grounded figures into the prose
            # (a detailed deep-dive with zero cited numbers is under-informative). Loaded
            # content (CFG-R1a / ARCH-R29) — never an inline literal; grounding still decides
            # truth downstream (§7).
            system = f"{system}\n\n{detailed_directive}"
        # COMPOSE-R3 (inline tags): the model emits ONE plain-text answer carrying a
        # `<technical_proof>…</technical_proof>` evidence block plus the warm visible prose OUTSIDE
        # it; a simple deterministic regex (`parse_tagged_answer`) splits it into the two-layer
        # ComposedAnswer — `visible_answer` (carried downstream as `draft`) and `evidence_claims`
        # (the candidate-claim layer the grounder verifies). Fail-closed: an unclosed/duplicate/
        # stray tag is stripped from the visible prose, a malformed claim line is dropped, and a
        # block-only answer yields an empty `visible_answer` that grounding degrades honestly
        # (STATUS-R1). The evidence layer is persisted for grounding + reveal-on-demand (COACH-R8)
        # and is NEVER shown to the athlete (VOICE-R2) or serialized to the API (OUTCOME-R2); the
        # presentation strip (voice.enforce_presentation) is the second fail-closed tag guard.
        raw = await node_model.compose(system=system, context=context)
        composed = parse_tagged_answer(raw)
        update: dict[str, Any] = {
            "draft": composed.visible_answer,
            "evidence_claims": [ec.model_dump() for ec in composed.evidence_claims],
            "messages": [{"role": "assistant", "kind": "draft"}],
        }
        if trimmed:
            update["coverage_gaps"] = stamp_coverage_gaps(gs.turn_id(state), {"context_trimmed"})
        return gs.tick_visit(state, update)

    return compose


__all__ = ["make_compose", "make_reflect"]
