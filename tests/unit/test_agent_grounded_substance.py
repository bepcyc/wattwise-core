"""COMPOSE-R1 / COMPOSE-R2 / STATUS-R1: the coach is grounded in a rendered fact sheet and
``completed`` is coupled to grounded substance (issues #44/#45/#46).

These pin the root-caused fix for the live-model defect where the coach emitted vague,
number-free prose with zero citations yet terminated ``completed`` (inverting honesty —
``completed`` on an empty answer, ``degraded`` on a full record):

* COMPOSE-R1 — :func:`render_context` renders each gathered capability through the
  deterministic fact-sheet path so the model sees the CURRENT canonical values + trend FIRST
  (a compact day-series tail follows), never a raw ``PmcDay(...)`` repr whose warm-up zeros
  dominate the salience and steer the answer away from the one claim that can ground.
* STATUS-R1 — a data-grounded PROCEED that published ZERO grounded survivors degrades to the
  honest fail-closed outcome (the same localized limitation copy as a grounder abstain), never
  ``completed``. A run that gathered no metric capability stays ``completed`` (exempt by
  construction).

Offline and self-contained (TIER-R1): the end-to-end leg drives the PRODUCTION graph through
``build_graph`` with in-test fakes only (no sibling agent in-flight module imported, ARCH-R21).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from wattwise_core.agent.contracts import (
    AgentState,
    Claim,
    ClaimKind,
    ComposedAnswer,
    GroundDecision,
    GroundedClaim,
    GroundingResult,
    GroundVerdict,
    ReflectDecision,
    ReflectVerdict,
    RetrievalRequest,
    RunStatus,
)
from wattwise_core.agent.graph import AgentServices, build_graph
from wattwise_core.agent.graph_state import (
    build_caveat,
    gathered_metric_capability,
    grounded_survivor_count,
    render_context,
    stamp_retrieved,
    terminal_status,
    turn_id,
)
from wattwise_core.agent.grounding_factsheet import render_capability_factsheet
from wattwise_core.analytics.pmc import PmcDay
from wattwise_core.analytics.result import Computed, Unavailable, UnavailableReason

pytestmark = pytest.mark.unit


# --- COMPOSE-R1: the rendered fact sheet leads with current values, never a repr ---------


def _zero_dominated_pmc() -> list[Computed[PmcDay]]:
    """A warm-up-from-zero PMC series whose ONLY non-zero day is the latest (the bug shape)."""
    warmup = [Computed(PmcDay(ctl=0.0, atl=0.0, tsb=0.0)) for _ in range(10)]
    return [*warmup, Computed(PmcDay(ctl=1.81, atl=6.60, tsb=-2.99))]


def test_factsheet_leads_with_current_pmc_values_no_repr() -> None:
    """COMPOSE-R1: the weekly-load sheet leads with the CURRENT CTL/ATL/form, not a repr."""
    sheets = render_capability_factsheet({"weekly_load": _zero_dominated_pmc()})
    body = sheets["weekly_load"]
    first_line = body.splitlines()[0]
    # The CURRENT (latest-day) values lead — the one claim that can ground, not the zeros.
    assert "current fitness (ctl) 1.81" in first_line
    assert "current fatigue (atl) 6.6" in first_line
    assert "current form (tsb) -2.99" in first_line
    # The warm-up-from-zero history reads as a TREND, not as the dominant signal.
    assert "rising from 0" in first_line
    # No raw object repr anywhere (the pre-fix primary rendering, forbidden by COMPOSE-R1).
    assert "PmcDay(" not in body
    assert "Computed(" not in body


def test_render_context_leads_with_current_values_and_has_no_pmcday_repr() -> None:
    """COMPOSE-R1 end-to-end: the compose context envelopes the fact sheet, not a raw dump."""
    retrieved = {
        "weekly_load": _zero_dominated_pmc(),
        "critical_power": Unavailable(UnavailableReason.INSUFFICIENT_DATA),
    }
    context, _trimmed = render_context("how is my training load?", retrieved)
    # Forbidden raw repr is absent; the answer-bearing current value is present and leads its block.
    assert "PmcDay(" not in context
    assert "current fitness (ctl) 1.81" in context
    # An unavailable capability renders an HONEST line, never a fabricated number or a 0.
    assert "no current value available" in context


def test_factsheet_unavailable_metric_states_no_value_not_zero() -> None:
    """COMPOSE-R1 fail-closed: an Unavailable record renders an honest line, never a 0."""
    sheets = render_capability_factsheet(
        {"hrv": Unavailable(UnavailableReason.MISSING_REQUIRED_INPUT)}
    )
    assert sheets["hrv"] == "no current value available"


def test_factsheet_single_scalar_metric_leads_with_current_value() -> None:
    """COMPOSE-R1: a single computed scalar metric (CP) leads with its current value verbatim."""

    class _CP:
        cp_w = 312.0
        w_prime_j = 21000.0

    sheets = render_capability_factsheet({"critical_power": Computed(_CP())})
    body = sheets["critical_power"]
    assert "current critical power (critical_power_w) 312" in body
    assert "current anaerobic capacity (w_prime_j) 21000" in body


# --- STATUS-R1: completed is coupled to grounded substance, not the routing decision -----


def _data_grounded_state(*, citations: Sequence[Mapping[str, Any]]) -> AgentState:
    """A finalize-time state for a DATA-GROUNDED run (a metric capability WAS gathered)."""
    state: AgentState = AgentState(
        athlete_id="athlete-status",
        trigger="user_turn",
        request_text="how is my training load?",
        locale="en",
        idempotency_key="idem-status",
    )
    # A gathered canonical capability == the request was data-grounded (STATUS-R1 gate input).
    state["retrieved"] = stamp_retrieved(turn_id(state), {"weekly_load": _zero_dominated_pmc()})
    state["citations"] = list(citations)
    return state


def test_status_r1_proceed_zero_survivors_degrades() -> None:
    """STATUS-R1: a data-grounded PROCEED with ZERO grounded survivors degrades, never completes."""
    state = _data_grounded_state(citations=[])
    assert gathered_metric_capability(state) is True
    assert grounded_survivor_count(state) == 0
    status = terminal_status(state, GroundDecision.PROCEED, ceiling=99)
    assert status is RunStatus.DEGRADED


def test_status_r1_proceed_with_survivors_completes() -> None:
    """STATUS-R1: a data-grounded PROCEED that DID ground a survivor completes normally."""
    state = _data_grounded_state(citations=[{"metric": "ctl", "value": 1.81}])
    assert grounded_survivor_count(state) == 1
    status = terminal_status(state, GroundDecision.PROCEED, ceiling=99)
    assert status is RunStatus.COMPLETED


def test_status_r1_number_free_run_exempt_when_no_metric_gathered() -> None:
    """STATUS-R1 exemption: a run that gathered NO metric capability stays completed.

    A pure motivational/scheduling follow-up legitimately has no grounded number and is exempt
    by construction — the gate must not force-degrade an honestly number-free answer.
    """
    state: AgentState = AgentState(
        athlete_id="athlete-nogather",
        trigger="user_turn",
        request_text="give me a pep talk",
        locale="en",
        idempotency_key="idem-nometric",
    )
    state["retrieved"] = stamp_retrieved(turn_id(state), {})
    state["citations"] = []
    assert gathered_metric_capability(state) is False
    status = terminal_status(state, GroundDecision.PROCEED, ceiling=99)
    assert status is RunStatus.COMPLETED


def test_status_r1_caveat_reads_degraded_fidelity_not_partial() -> None:
    """STATUS-R1: the empty-survivor degrade is an honest refusal -> ``degraded`` fidelity."""
    state = _data_grounded_state(citations=[])
    status = terminal_status(state, GroundDecision.PROCEED, ceiling=99)
    caveat = build_caveat(state, status, GroundDecision.PROCEED)
    assert caveat is not None
    assert caveat["fidelity"] == "degraded"


# --- STATUS-R1 end-to-end through the production graph ------------------------------------


class _NarrateZerosModel:
    """A model that narrates the zero-dominated series WITHOUT stating the current value.

    It is the real-model failure shape from issue #44: groundable-sounding prose with no
    canonical number that survives. ``compose`` returns number-free filler; ``structured``
    scripts a reflect ``proceed`` so the run does not loop.
    """

    def __init__(self) -> None:
        self.compose_calls = 0

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M:
        if schema is ReflectDecision:
            return ReflectDecision(verdict=ReflectVerdict.ANSWER_WITH_CAVEAT)  # type: ignore[return-value]
        if schema.__name__ == "ComposedAnswer":
            return ComposedAnswer(  # type: ignore[return-value]
                visible_answer=await self.compose(system=system, context=data),
                evidence_claims=(),
            )
        raise NotImplementedError(schema.__name__)

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        self.compose_calls += 1
        return "There isn't a single number, but the pattern shows a clear ramp-up."


class _PmcPlanner:
    async def plan(
        self, *, request_text: str | None, gaps: Sequence[str], already: Sequence[str]
    ) -> Sequence[RetrievalRequest]:
        return [RetrievalRequest(capability="weekly_load", params={})]


class _PmcGateway:
    async def gather(
        self, *, athlete_id: str, requests: Sequence[RetrievalRequest]
    ) -> Mapping[str, Any]:
        return {"weekly_load": _zero_dominated_pmc()}


class _NoGapCoverage:
    def assess(self, *, request_text: str | None, retrieved: Mapping[str, Any]) -> set[str]:
        return set()


class _ScrubAllGrounder:
    """PROCEED but EVERY checkable number was scrubbed -> zero grounded survivors (issue #44)."""

    async def ground(
        self,
        *,
        athlete_id: str,
        draft: str,
        retrieved: Mapping[str, Any],
        request_text: str | None = None,
        active_constraints: object = None,
        evidence_claims: object = None,
    ) -> GroundingResult:
        # A publishable complementary (number-free) claim -> PROCEED with NO citation survivor.
        claim = Claim(kind=ClaimKind.STATEMENT, text="a clear ramp-up")
        survivor = GroundedClaim(claim=claim, verdict=GroundVerdict.COMPLEMENTARY, citation=None)
        return GroundingResult(
            decision=GroundDecision.PROCEED, claims=(survivor,), scrubbed_text=draft
        )


def _config(thread: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread}, "recursion_limit": 50}


def _zero_input() -> AgentState:
    return AgentState(
        athlete_id="athlete-zero",
        trigger="user_turn",
        request_text="what's my total training load over the last weeks?",
        locale="en",
        idempotency_key="idem-zero",
    )


async def test_e2e_narrate_zeros_ends_degraded_not_completed() -> None:
    """STATUS-R1 e2e: the model narrates zeros, grounds nothing -> DEGRADED honest refusal.

    The PRE-FIX behaviour was ``completed`` with ``citations=[]`` (issue #44/#45). With STATUS-R1
    the run degrades and ships ONLY the honest localized limitation copy — never a confident
    number-free non-answer dressed up as a completed one.
    """
    svc = AgentServices(
        planner=_PmcPlanner(),
        gateway=_PmcGateway(),
        coverage=_NoGapCoverage(),
        grounder=_ScrubAllGrounder(),
    )
    graph = build_graph(_NarrateZerosModel(), svc, InMemorySaver())

    out = await graph.ainvoke(_zero_input(), config=_config("status-r1-e2e"))

    assert out["status"] is RunStatus.DEGRADED
    assert out["citations"] == []
    # The honest fail-closed limitation replaced the number-free draft (GROUND-R6 voice).
    assert "enough confirmed data" in out["grounded_text"]
    assert out["coverage_caveat"]["fidelity"] == "degraded"
    # No grounded observation to drill into a non-answer (COACH-R8).
    assert out["observations"] == []
