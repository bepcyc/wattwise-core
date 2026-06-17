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
    activity_refs_from_requests,
    build_caveat,
    gathered_metric_capability,
    grounded_survivor_count,
    render_context,
    stamp_retrieved,
    terminal_status,
    turn_id,
)
from wattwise_core.agent.grounding_factsheet import render_capability_factsheet
from wattwise_core.analytics.np_if_tss import LoadMetricsBundle, NormalizedPowerValue
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


# --- COMPOSE-R1a (#95): a per-ride load sheet names its activity + leads with activity_tss ---


def _load_bundle(*, computed: bool = True) -> Computed[LoadMetricsBundle]:
    """A per-ride load-metrics bundle (the ``coggan`` record shape).

    ``computed=True`` is the power path with a Computed per-ride TSS; ``computed=False`` makes
    every RENDERED field Unavailable so the sheet must fall to an honest "no value" line.
    """
    npv = Computed(
        NormalizedPowerValue(
            np_w=210.0, avg_power_w=190.0, mean_r_w=44100.0, analysis_window_s=3000
        )
    )
    gap = Unavailable(UnavailableReason.INSUFFICIENT_DATA, "no valid power channel")
    return Computed(
        LoadMetricsBundle(
            duration_valid_s=Computed(3600) if computed else gap,
            np=npv if computed else gap,
            if_=Computed(0.84) if computed else gap,
            tss=Computed(78.0) if computed else gap,
            hr_load=Unavailable(UnavailableReason.OUT_OF_DOMAIN, "power won"),
            tss_per_hour=Computed(78.0) if computed else gap,
            efficiency_factor=Unavailable(UnavailableReason.MISSING_REQUIRED_INPUT, "no hr"),
            variability_index=Computed(1.1) if computed else gap,
            intensity_class=Computed("threshold") if computed else gap,
            load_model="power_tss",
        )
    )


def test_factsheet_load_bundle_names_activity_and_leads_with_activity_tss() -> None:
    """COMPOSE-R1a: a per-ride load sheet names its activity and leads with activity_tss.

    The activity_id (from the planner's request) + the canonical ``activity_tss`` code are what
    let the model author a per-ride claim the §7 grounder binds to the right activity (#47/#99).
    """
    sheets = render_capability_factsheet(
        {"load_metrics": _load_bundle()}, {"load_metrics": "ride-42"}
    )
    body = sheets["load_metrics"]
    assert body.startswith("for activity ride-42: ")
    assert "current training stress score (activity_tss) 78" in body
    assert "intensity factor (if) 0.84" in body
    # Never a raw dataclass / Computed repr (COMPOSE-R1).
    assert "LoadMetricsBundle(" not in body
    assert "Computed(" not in body


def test_factsheet_load_bundle_without_activity_ref_omits_activity_line() -> None:
    """COMPOSE-R1a fail-closed: with no planned id, render the figures but never a guessed id."""
    body = render_capability_factsheet({"load_metrics": _load_bundle()})["load_metrics"]
    assert "for activity" not in body
    # The per-ride TSS still leads (salience is independent of whether an id was recoverable).
    assert body.startswith("current training stress score (activity_tss) 78")


def test_factsheet_load_bundle_all_unavailable_states_no_value_not_zero() -> None:
    """COMPOSE-R1a fail-closed: an all-Unavailable bundle is an honest line, never a 0 or a repr."""
    body = render_capability_factsheet(
        {"load_metrics": _load_bundle(computed=False)}, {"load_metrics": "ride-9"}
    )["load_metrics"]
    assert body == "for activity ride-9: no current value available"
    assert "LoadMetricsBundle(" not in body


def test_render_context_threads_activity_ref_into_per_ride_sheet() -> None:
    """COMPOSE-R1a end-to-end: render_context carries activity_refs into the per-ride block."""
    retrieved = {"load_metrics": _load_bundle()}
    context, _trimmed = render_context(
        "what was my TSS on that ride?",
        retrieved,
        activity_refs={"load_metrics": "ride-42"},
    )
    assert "for activity ride-42: current training stress score (activity_tss) 78" in context
    assert "LoadMetricsBundle(" not in context


def test_activity_refs_from_requests_maps_only_real_per_ride_ids() -> None:
    """COMPOSE-R1a wiring: only requests carrying a real string activity_id map; last wins on a dup.

    A non-activity capability (no activity_id) and an empty/blank id contribute NO entry
    (fail-closed) — the fact sheet then renders those without a guessed activity line.
    """
    requests = [
        RetrievalRequest(capability="weekly_load", params={}),  # range-scoped: no id
        RetrievalRequest(capability="load_metrics", params={"activity_id": "ride-1"}),
        RetrievalRequest(capability="decoupling", params={"activity_id": ""}),  # blank -> dropped
        RetrievalRequest(capability="load_metrics", params={"activity_id": "ride-2"}),  # dup: wins
    ]
    refs = activity_refs_from_requests(requests)
    assert refs == {"load_metrics": "ride-2"}


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
    # A real completable run carries a non-empty athlete-visible body; set one so the empty-visible
    # fail-closed guard (COMPOSE-R3 point 1) does not fire on these synthetic status-only states.
    state["grounded_text"] = "Your training load is holding steady."
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


def test_exhausted_recovery_with_grounded_gapfree_answer_completes() -> None:
    """#85: a bounded-recovery decision (REGENERATE/REPLAN) that ran out of retries but published
    a GROUNDED, gap-free, CITED answer finalizes COMPLETED — not a trust-eroding degraded.

    The residual non-deterministic tail of #45: an earlier recovery cycle left a non-PROCEED
    ground decision in state, yet the final published answer grounded cleanly (citations>0) with no
    open coverage gap. terminal_status must read the published substance, not the stale recovery
    signal. A non-PROCEED decision only degrades when the answer did NOT come out grounded+gap-free.
    """
    state = _data_grounded_state(citations=[{"metric": "ctl", "value": 1.81}])
    assert grounded_survivor_count(state) == 1
    for decision in (GroundDecision.REGENERATE, GroundDecision.REPLAN):
        assert terminal_status(state, decision, ceiling=99) is RunStatus.COMPLETED


def test_exhausted_recovery_without_survivors_still_degrades() -> None:
    """#85 guard: a non-PROCEED recovery that published NO grounded survivor still degrades."""
    state = _data_grounded_state(citations=[])
    assert grounded_survivor_count(state) == 0
    assert terminal_status(state, GroundDecision.REGENERATE, ceiling=99) is RunStatus.DEGRADED


def test_abstain_always_degrades_even_with_citations() -> None:
    """#85 guard: an ABSTAIN is an honest refusal and degrades regardless of any citations."""
    state = _data_grounded_state(citations=[{"metric": "ctl", "value": 1.81}])
    assert terminal_status(state, GroundDecision.ABSTAIN, ceiling=99) is RunStatus.DEGRADED


def test_empty_visible_body_degrades_even_with_grounded_survivors() -> None:
    """COMPOSE-R3 point 1: an EMPTY athlete-visible body never ships as completed.

    A compose answer that was ALL evidence layer (a ``<technical_proof>`` block with no surrounding
    prose) parses to an empty ``visible_answer`` -> empty ``grounded_text``; even though its claims
    grounded (a survivor is present), a blank body MUST degrade to the honest fail-closed outcome.
    """
    state = _data_grounded_state(citations=[{"metric": "ctl", "value": 1.81}])
    assert grounded_survivor_count(state) == 1
    state["grounded_text"] = "   "  # all-evidence-layer answer -> blank visible prose
    status = terminal_status(state, GroundDecision.PROCEED, ceiling=99)
    assert status is RunStatus.DEGRADED


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
    state["grounded_text"] = "You've got this — keep showing up."  # a real number-free body
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
