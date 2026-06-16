"""Unit tests for the Insight + Briefing deliverables (COACH-R1 #4/#5) and the COACH-R8
stable-id observations on the readiness deliverable.

The unit under test is the typed PROJECTION over the agent-graph seam (mirroring
``test_deliverables``): a ``FakeGraph`` records the immutable inputs the producer built (so
the GRAPH-R2.1 trigger contract is assertable) and replays a pre-grounded terminal state.
Asserted: the briefing is driven by ``scheduled_briefing`` with NO request text and the ONE
``briefing_screen`` carried in the immutable inputs; the insight is a ``user_turn`` over its
single topic at the SHORT length; both carry the originating durable ``thread_id`` + stable-id
observations + follow-up prompts (COACH-R8); only grounded outputs are projected (OUTCOME-R2);
and a missing-data run degrades visibly with its caveat (OUTCOME-R3/-R4). The readiness check
drives the REAL ``readiness_assessment`` with a scripted grounder and asserts its grounded
survivors project to stable-id observations (COACH-R8).
"""

from __future__ import annotations

from typing import Any

import pytest

from wattwise_core.agent.briefing_deliverable import briefing, insight
from wattwise_core.agent.contracts import (
    AgentState,
    Claim,
    ClaimKind,
    GroundDecision,
    GroundedClaim,
    GroundingResult,
    GroundVerdict,
    RunStatus,
)
from wattwise_core.agent.readiness_deliverable import readiness_assessment

pytestmark = pytest.mark.unit


class FakeGraph:
    """A :class:`CoachGraph` recording inputs and replaying a fixed terminal state."""

    def __init__(self, terminal: AgentState) -> None:
        self._terminal = terminal
        self.received: AgentState | None = None

    async def run(self, state: AgentState) -> AgentState:
        self.received = state
        return self._terminal


def _grounded_terminal() -> AgentState:
    """A healthy grounded terminal state with one stable-id observation (COACH-R8)."""
    return {
        "status": RunStatus.COMPLETED,
        "thread_id": "ath-7:briefing:today",
        "grounded_html": "<p>You're fresh and your fitness is trending up.</p>",
        "grounded_text": "You're fresh and your fitness is trending up.",
        "observations": [
            {
                "observation_id": "obs-form-up",
                "text": "Your form is coming around.",
                "citations": [
                    {"record_id": "pmc-1", "metric": "tsb", "value": 4.0, "as_of": "2026-06-09"}
                ],
            }
        ],
        "citations": [{"record_id": "pmc-1", "metric": "tsb", "value": 4.0, "as_of": "2026-06-09"}],
        "coverage_caveat": None,
    }


def _degraded_terminal() -> AgentState:
    """A missing-data terminal state: degraded with a truthful caveat (OUTCOME-R3/-R4)."""
    return {
        "status": RunStatus.DEGRADED,
        "thread_id": "ath-7:briefing:today",
        "grounded_html": "<p>There isn't enough recent data for a heads-up yet.</p>",
        "grounded_text": "There isn't enough recent data for a heads-up yet.",
        "observations": [],
        "citations": [],
        "coverage_caveat": {"missing": ["weekly_load"], "fidelity": "degraded"},
    }


async def test_briefing_drives_scheduled_trigger_for_one_screen() -> None:
    """briefing builds a scheduled_briefing run: no request text, one screen (GRAPH-R2.1)."""
    graph = FakeGraph(_grounded_terminal())
    await briefing(graph, "ath-7", "today")
    assert graph.received is not None
    assert graph.received["trigger"] == "scheduled_briefing"
    assert graph.received.get("request_text") is None
    assert graph.received.get("briefing_screen") == "today"
    assert graph.received["athlete_id"] == "ath-7"


async def test_briefing_projects_grounded_outputs_and_followups() -> None:
    """briefing surfaces only graph-grounded outputs + COACH-R8 affordances (OUTCOME-R2)."""
    result = await briefing(FakeGraph(_grounded_terminal()), "ath-7", "today")
    assert result.status is RunStatus.COMPLETED
    assert result.briefing_screen == "today"
    assert result.thread_id == "ath-7:briefing:today"  # follow-up handle (COACH-R8)
    assert result.observations[0].observation_id == "obs-form-up"
    assert result.citations[0].record_id == "pmc-1"
    assert result.suggested_followups  # engine-generated copy, client only renders


async def test_briefing_degrades_visibly_on_missing_data() -> None:
    """A missing-input briefing ships degraded + caveat, never a guess (OUTCOME-R3/-R4)."""
    result = await briefing(FakeGraph(_degraded_terminal()), "ath-7", "today")
    assert result.status is RunStatus.DEGRADED
    assert result.coverage_caveat == {"missing": ["weekly_load"], "fidelity": "degraded"}


async def test_insight_drives_user_turn_over_its_single_topic() -> None:
    """insight is a user_turn run over the one topic at the SHORT length (COACH-R1 #4)."""
    graph = FakeGraph(_grounded_terminal())
    await insight(graph, "ath-7", "decoupling trend")
    assert graph.received is not None
    assert graph.received["trigger"] == "user_turn"
    assert graph.received["request_text"] == "decoupling trend"
    assert graph.received["response_length"] == "short"


async def test_insight_carries_thread_id_and_stable_observations() -> None:
    """insight carries its originating thread + stable-id observations (COACH-R8)."""
    result = await insight(FakeGraph(_grounded_terminal()), "ath-7", "decoupling trend")
    assert result.thread_id == "ath-7:briefing:today"
    assert result.observations[0].observation_id == "obs-form-up"
    assert result.insight_text  # the grounded body, never un-grounded model text


class _ScriptedReadinessGrounder:
    """A readiness grounder double replaying one grounded form survivor (GROUND-R5/R7)."""

    async def ground(
        self,
        *,
        athlete_id: str,
        draft: str,
        retrieved: Any,
        request_text: str | None = None,
        active_constraints: object = None,
    ) -> GroundingResult:
        claim = Claim(kind=ClaimKind.NUMBER, text="form 4.0", metric="form", value=4.0)
        survivor = GroundedClaim(
            claim=claim,
            verdict=GroundVerdict.GROUNDED,
            citation={"record_id": "pmc-1", "metric": "tsb", "value": 4.0, "as_of": "2026-06-09"},
        )
        return GroundingResult(
            decision=GroundDecision.PROCEED, claims=(survivor,), scrubbed_text=draft
        )


async def test_readiness_projects_stable_id_observations_from_survivors() -> None:
    """The readiness deliverable carries stable-id observations for drill/reveal (COACH-R8)."""
    readiness = await readiness_assessment(
        "ath-7",
        form=4.0,
        as_of="2026-06-09",
        hrv_rmssd=60.0,
        hrv_baseline=58.0,
        narrate=None,
        grounder=_ScriptedReadinessGrounder(),
    )
    assert readiness.status is RunStatus.COMPLETED
    assert readiness.observations, "grounded survivors must project to observations"
    obs = readiness.observations[0]
    assert obs.observation_id.startswith("obs-")
    assert obs.citations and obs.citations[0].record_id == "pmc-1"
