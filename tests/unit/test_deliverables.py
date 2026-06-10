"""Unit tests for the Phase-1 coach deliverables projection (doc 50).

These cover the two Phase-1 deliverables (:func:`answer_question`, the weekly
digest) and the deterministic presentation gate (COACH-R7 / EVAL-R5b.1). The unit
under test is the typed PROJECTION over the agent-graph seam, so the fixture is a
``FakeGraph`` (a :class:`CoachGraph`) that records the immutable inputs it received
and returns a crafted terminal :class:`AgentState` — mirroring how a recorded/mocked
model run drives the eval suite (EVAL-R1) without any live source. Asserted: the
right trigger is driven (GRAPH-R2.1), the digest leads with a state phrase and cites
canonical records, it abstains visibly on missing data (OUTCOME-R3/-R4), only
grounded outputs are projected (OUTCOME-R2), and athlete identity is server-derived
(AGT-SEC-R1).
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from wattwise_core.agent.contracts import AgentState, RunStatus
from wattwise_core.agent.deliverables import (
    AgentAnswer,
    Citation,
    CoachGraph,
    Digest,
    Observation,
    answer_question,
    count_foregrounded_numbers,
    first_sentence,
    leads_with_state,
    number_cap,
    weekly_digest,
)
from wattwise_core.agent.voice import VoicePresentation


class FakeGraph:
    """A :class:`CoachGraph` that records inputs and returns a fixed terminal state.

    Stands in for the in-flight stateful graph: it captures the write-once inputs the
    deliverable built (so the trigger/identity contract is assertable) and replays a
    pre-grounded terminal :class:`AgentState`, the way a recorded model run does in the
    offline eval suite (EVAL-R1).
    """

    def __init__(self, terminal: AgentState) -> None:
        self._terminal = terminal
        self.received: AgentState | None = None

    async def run(self, state: AgentState) -> AgentState:
        self.received = state
        return self._terminal


def _digest_terminal() -> AgentState:
    """A healthy, fully grounded weekly-digest terminal state (leads with a state phrase)."""
    return {
        "status": RunStatus.COMPLETED,
        "idempotency_key": "thread-week-1",
        "grounded_html": (
            "<p>Strong week — you stacked three solid rides and your fitness is "
            "trending up. One thing to watch: the load jump was a little steep.</p>"
        ),
        "grounded_text": (
            "Strong week — you stacked three solid rides and your fitness is "
            "trending up. One thing to watch: the load jump was a little steep."
        ),
        "observations": [
            {
                "observation_id": "obs-load-ramp",
                "text": "Your load ramped up faster than usual this week.",
                "citations": [
                    {
                        "record_id": "pmc-2026-06-06",
                        "metric": "ramp",
                        "value": 8.0,
                        "as_of": "2026-06-06",
                    },
                ],
            }
        ],
        "citations": [
            {"record_id": "act-101", "metric": "tss", "value": 95.0, "as_of": "2026-06-02"},
            {"record_id": "pmc-2026-06-06", "metric": "ctl", "value": 62.0, "as_of": "2026-06-06"},
        ],
        "coverage_caveat": None,
    }


async def test_weekly_digest_drives_scheduled_trigger() -> None:
    """weekly_digest builds a scheduled_digest run with no request text (GRAPH-R2.1)."""
    graph = FakeGraph(_digest_terminal())
    await weekly_digest(graph, "athlete-7", "2026-06-06")
    assert graph.received is not None
    assert graph.received["trigger"] == "scheduled_digest"
    assert graph.received["request_text"] is None
    assert graph.received["athlete_id"] == "athlete-7"


async def test_weekly_digest_leads_with_state_phrase() -> None:
    """The digest's first athlete-facing sentence reads as a state phrase (COACH-R7)."""
    graph = FakeGraph(_digest_terminal())
    digest = await weekly_digest(graph, "athlete-7", "2026-06-06")
    assert isinstance(digest, Digest)
    assert leads_with_state(digest.digest_html)
    lead = first_sentence(digest.digest_html)
    assert lead.lower().startswith("strong week")


async def test_weekly_digest_cites_canonical_records() -> None:
    """Surviving claims carry citations to canonical record ids, never source ids (GROUND-R5)."""
    graph = FakeGraph(_digest_terminal())
    digest = await weekly_digest(graph, "athlete-7", "2026-06-06")
    assert digest.citations == (
        Citation(record_id="act-101", metric="tss", value=95.0, as_of="2026-06-02"),
        Citation(record_id="pmc-2026-06-06", metric="ctl", value=62.0, as_of="2026-06-06"),
    )
    assert digest.observations[0] == Observation(
        observation_id="obs-load-ramp",
        text="Your load ramped up faster than usual this week.",
        citations=(
            Citation(record_id="pmc-2026-06-06", metric="ramp", value=8.0, as_of="2026-06-06"),
        ),
    )
    assert digest.week_end == "2026-06-06"


async def test_weekly_digest_abstains_on_missing_data() -> None:
    """No week data -> degraded + a truthful caveat, never a guess (OUTCOME-R3/-R4)."""
    terminal: AgentState = {
        "status": RunStatus.DEGRADED,
        "idempotency_key": "thread-empty",
        "grounded_html": (
            "<p>I don't have enough of this week's rides synced yet to call how it went. "
            "Sync your watch and I'll take another look.</p>"
        ),
        "grounded_text": (
            "I don't have enough of this week's rides synced yet to call how it went. "
            "Sync your watch and I'll take another look."
        ),
        "observations": [],
        "citations": [],
        "coverage_caveat": {"missing": ["weekly_activities", "pmc"]},
    }
    digest = await weekly_digest(FakeGraph(terminal), "athlete-7", "2026-06-06")
    assert digest.status is RunStatus.DEGRADED
    assert digest.citations == ()
    assert digest.observations == ()
    assert digest.coverage_caveat == {"missing": ["weekly_activities", "pmc"]}
    # A degraded run offers only the warm expand prompt — no numbers-reveal it can't honor.
    assert digest.suggested_followups == ("Tell me more",)


# VOICE-R2 forbidden internals (must never appear in any athlete-facing follow-up copy).
_FORBIDDEN_WORDS = (
    "api",
    "endpoint",
    "database",
    "schema",
    "token",
    "model",
    "tier",
    "flash",
    "pro",
    "frontier",
    "checkpoint",
    "thread",
    "mcp",
    "grounding",
    "scrub",
    "coverage",
    "budget",
)


async def test_followups_are_generated_and_jargon_free() -> None:
    """A healthy digest offers engine-generated jargon-free follow-ups (COACH-R8, VOICE-R2)."""
    digest = await weekly_digest(FakeGraph(_digest_terminal()), "athlete-7", "2026-06-06")
    assert digest.suggested_followups == ("Show me the numbers behind that", "Tell me more")
    for prompt in digest.suggested_followups:
        lowered = prompt.lower()
        assert all(word not in lowered.split() for word in _FORBIDDEN_WORDS)


async def test_answer_question_drives_user_turn_with_question() -> None:
    """answer_question builds a user_turn run carrying the question text (GRAPH-R2.1, STATE-R2)."""
    terminal: AgentState = {
        "status": RunStatus.COMPLETED,
        "idempotency_key": "thread-q-1",
        "grounded_text": "You're recovered and sharp today — a great day to push.",
        "grounded_html": "<p>You're recovered and sharp today — a great day to push.</p>",
        "observations": [],
        "citations": [],
    }
    graph = FakeGraph(terminal)
    answer = await answer_question(
        graph, "athlete-9", "How am I doing today?", locale="en", response_length="short"
    )
    assert isinstance(answer, AgentAnswer)
    assert graph.received is not None
    assert graph.received["trigger"] == "user_turn"
    assert graph.received["request_text"] == "How am I doing today?"
    assert graph.received["locale"] == "en"
    assert answer.status is RunStatus.COMPLETED
    assert answer.thread_id == "thread-q-1"


async def test_identity_is_server_derived_not_from_graph_output() -> None:
    """Scope follows the caller's athlete_id; a graph output cannot change it (AGT-SEC-R1)."""
    terminal = _digest_terminal()
    # A malicious/buggy terminal state asserting a different athlete must NOT leak through:
    terminal_with_other: dict[str, Any] = dict(terminal)
    terminal_with_other["athlete_id"] = "victim-athlete"
    graph = FakeGraph(terminal_with_other)  # type: ignore[arg-type]
    await weekly_digest(graph, "athlete-7", "2026-06-06")
    assert graph.received is not None
    assert graph.received["athlete_id"] == "athlete-7"


async def test_only_citations_with_record_ids_survive() -> None:
    """A citation lacking a record id is dropped: no claim without a citation (GROUND-R5)."""
    terminal: AgentState = {
        "status": RunStatus.COMPLETED,
        "idempotency_key": "thread-q-2",
        "grounded_text": "Your fitness is trending up after a steady block.",
        "grounded_html": "<p>Your fitness is trending up after a steady block.</p>",
        "observations": [],
        "citations": [
            {"record_id": "ctl-1", "metric": "ctl", "value": 60.0, "as_of": "2026-06-06"},
            {"metric": "phantom", "value": 999.0, "as_of": "2026-06-06"},  # no record_id
        ],
    }
    answer = await answer_question(FakeGraph(terminal), "a", "How is my fitness?", locale="en")
    assert answer.citations == (
        Citation(record_id="ctl-1", metric="ctl", value=60.0, as_of="2026-06-06"),
    )


async def test_missing_status_falls_closed_to_degraded() -> None:
    """A terminal state with no status projects as degraded, not a fake completed (OUTCOME-R5)."""
    terminal: AgentState = {
        "idempotency_key": "thread-x",
        "grounded_text": "We're still gathering your numbers.",
        "grounded_html": "<p>We're still gathering your numbers.</p>",
        "observations": [],
        "citations": [],
    }
    answer = await answer_question(FakeGraph(terminal), "a", "status?", locale="en")
    assert answer.status is RunStatus.DEGRADED


async def test_html_text_bodies_both_present_with_one_missing() -> None:
    """When the graph fills only one body, the other mirrors it so the API always has both."""
    terminal: AgentState = {
        "status": RunStatus.COMPLETED,
        "idempotency_key": "t",
        "grounded_text": "You're trending up nicely.",
        "observations": [],
        "citations": [],
    }
    answer = await answer_question(FakeGraph(terminal), "a", "q", locale="en")
    assert answer.answer_text == "You're trending up nicely."
    assert answer.answer_html == "You're trending up nicely."


def test_leads_with_state_rejects_bare_metric_token() -> None:
    """A lead that is only a number/metric token fails the COACH-R7 deterministic gate."""
    assert not leads_with_state("<p>62</p>")
    assert not leads_with_state("CTL 62.")
    assert leads_with_state("You're recovered and sharp today.")


def test_leads_with_state_rejects_metric_report_frame_and_token_list() -> None:
    """A metrics-report lead or a raw metric-token list fails the strengthened gate (VOICE-R7)."""
    # The exact report frames the spec forbids as a LEAD (COACH-R7 / VOICE-R7):
    assert not leads_with_state("Here is your current training-load picture from the data:")
    assert not leads_with_state("Here are your latest metrics:")
    assert not leads_with_state("<p>What I can tell you is your latest training-load picture:</p>")
    # A raw metric-token list lead (>= 2 token-words) must NOT pass on word-count alone:
    assert not leads_with_state("ctl: 6.7, atl: 30.2, tsb: -28.")
    assert not leads_with_state("Metrics: ctl 6.7, atl 30.2.")
    # A normal warm state read still passes — even one naming an athlete-native word + number:
    assert leads_with_state("Your fitness is trending up after a steady block.")
    assert leads_with_state("You're carrying more fatigue than usual right now.")


def test_count_foregrounded_numbers_ignores_markup_digits() -> None:
    """Number-density counts prose numbers only, not digits inside stripped markup (VOICE-R7)."""
    body = '<p class="lead7">You rode 3 times and gained 2.5 points of fitness.</p>'
    assert count_foregrounded_numbers(body) == 2


def test_count_foregrounded_numbers_counts_sentence_final_number() -> None:
    """A number ending a sentence is counted (the cap gate must not undercount it, VOICE-R7)."""
    # Pre-fix bug: the trailing "." made "62" uncounted, leaving the number-cap gate too lenient.
    assert count_foregrounded_numbers("Your fitness is 62.") == 1
    assert count_foregrounded_numbers("fitness 6.7, fatigue 30.2, form -28.") == 3
    # A decimal / version-ish run is still not split into spurious bare integers:
    assert count_foregrounded_numbers("It was 3.14 not 3.") == 2


async def test_answer_question_scrubs_tokens_and_repairs_report_lead() -> None:
    """The DELIVERED answer leads with state + carries no raw token, even from a report draft.

    Drives the real :func:`answer_question` projection over a recorded terminal state that
    replays the OLD violating output (report-frame lead + raw ctl/atl/tsb tokens); the
    presentation enforcement (VOICE-R2/COACH-R7) must repair the lead and translate every token,
    while the grounded citations (GROUND-R5/R7) stay verbatim canonical.
    """
    draft = (
        "Here is your current training-load picture from the data: your fitness (ctl) is 6.7, "
        "your fatigue (atl) is 30.2, and your form (tsb) is -28."
    )
    terminal: AgentState = {
        "status": RunStatus.COMPLETED,
        "idempotency_key": "thread-report",
        "grounded_text": draft,
        "grounded_html": f"<p>{draft}</p>",
        "observations": [],
        "citations": [
            {"record_id": "ctl-1", "metric": "ctl", "value": 6.7, "as_of": "2026-06-08"},
            {"record_id": "atl-1", "metric": "atl", "value": 30.2, "as_of": "2026-06-08"},
            {"record_id": "tsb-1", "metric": "tsb", "value": -28.0, "as_of": "2026-06-08"},
        ],
    }
    policy = VoicePresentation.from_aliases(
        {"fitness": "ctl", "fatigue": "atl", "form": "tsb", "freshness": "tsb"}
    )
    answer = await answer_question(
        FakeGraph(terminal),
        "athlete-1",
        "How much load over six weeks?",
        locale="en",
        response_length="standard",
        presentation=policy,
    )
    assert leads_with_state(answer.answer_text)
    lowered = answer.answer_text.lower()
    for token in ("ctl", "atl", "tsb"):
        assert not re.search(rf"(?<![a-z]){token}(?![a-z])", lowered), token
    # Grounding untouched: the canonical citations remain verbatim (GROUND-R7).
    assert answer.citations == (
        Citation(record_id="ctl-1", metric="ctl", value=6.7, as_of="2026-06-08"),
        Citation(record_id="atl-1", metric="atl", value=30.2, as_of="2026-06-08"),
        Citation(record_id="tsb-1", metric="tsb", value=-28.0, as_of="2026-06-08"),
    )


def test_number_density_cap_defaults_per_length() -> None:
    """The number ceiling rises with verbosity: short <= standard <= detailed (VOICE-R8)."""
    assert number_cap("short") == 2
    assert number_cap("standard") == 3
    assert number_cap("detailed") == 4


def test_first_sentence_strips_markup_and_splits() -> None:
    """first_sentence returns the leading sentence with markup and excess whitespace removed."""
    html = "<p>Strong week — three solid rides. Watch the ramp next week.</p>"
    assert first_sentence(html) == "Strong week — three solid rides."


def test_fake_graph_satisfies_coachgraph_protocol() -> None:
    """The FakeGraph fixture is structurally a CoachGraph (the seam the deliverables drive)."""
    assert isinstance(FakeGraph(_digest_terminal()), CoachGraph)


def test_response_length_must_be_a_known_value() -> None:
    """A defaulted/known response_length drives the run without error (VOICE-R8 closed set)."""
    # standard is the persisted default; verbosity never changes truth/identity/scope.
    assert number_cap("standard") == 3
    with pytest.raises(KeyError):
        number_cap("verbose")  # type: ignore[arg-type]
