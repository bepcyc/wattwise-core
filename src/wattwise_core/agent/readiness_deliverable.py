"""The readiness/form coach deliverable: a typed verdict + state-first narration.

This is the focused sibling of :mod:`wattwise_core.agent.deliverables` that owns the
readiness/form deliverable and NOTHING else (QA-EVAL-R2.4 / COACH-R7 / STRUCT-R1). It
reuses the SHARED voice/projection layer from the LEAF :mod:`wattwise_core.agent.voice`
module (the :class:`~wattwise_core.agent.voice.Citation` shape, the deterministic
number-density / leads-with-state checks, and the number-cap enforcement) — it imports
those FROM ``voice``, NOT from ``deliverables``. That makes this module independently
importable (no ``deliverables`` <-> ``readiness_deliverable`` cycle): it depends only
DOWNWARD on ``voice``, while ``deliverables`` re-exports the readiness names this module
owns so every public import path stays stable.

Grounded numbers resolve against canonical analytics by metric name: the form number
grounds against ``"form"`` (the athlete-facing verbatim alias of canonical TSB,
capabilities.MetricName.FORM) and the HRV number against ``"hrv_rmssd_ms"`` (GROUND-R7).
The grounder (engine.ClaimGrounder) owns the name resolution; the deliverable only
projects what survives.

Readiness is a typed STATE (``go | maintain | ease | rest``), NEVER a number
(QA-EVAL-R2.4 / COACH-R7): the DELIVERED verdict is ALWAYS the deterministic
:func:`~wattwise_core.analytics.readiness.assess_readiness` verdict (canonical wins,
mirroring grounding's GROUND-R3 substitution), so it is metric-consistent by
construction; the model only proposes a warm state sentence, which a code gate checks
and falls back from when it disagrees (COACH-R3 / EVAL-R5, fail-closed).

Cited requirements: COACH-R3, COACH-R7, QA-EVAL-R2.4, STRUCT-R1, EVAL-R5,
GROUND-R5/-R6/-R7, VOICE-R7/-R9.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from wattwise_core.agent.contracts import GroundingResult, RunStatus
from wattwise_core.agent.observations import build_observations
from wattwise_core.agent.projection import project_observations
from wattwise_core.agent.voice import (
    Citation,
    Observation,
    ResponseLength,
    _demote_numbers,
    _enforce_number_cap,
    _project_citations,
    count_foregrounded_numbers,
    first_sentence,
    leads_with_state,
    number_cap,
)
from wattwise_core.analytics.readiness import (
    ReadinessAssessment,
    assess_readiness,
    readiness_consistent,
)
from wattwise_core.domain.enums import ReadinessVerdict


@dataclass(frozen=True, slots=True)
class Readiness:
    """The readiness/form coach deliverable: a typed VERDICT + state-first narration.

    Readiness is a typed STATE (``go | maintain | ease | rest``), NEVER a number
    (QA-EVAL-R2.4 / COACH-R7): there is deliberately NO numeric ``readiness`` score on
    this contract. ``verdict`` is the DELIVERED verdict — ALWAYS the deterministic
    :func:`~wattwise_core.analytics.readiness.assess_readiness` verdict (canonical wins,
    mirroring grounding's GROUND-R3 substitution), so it is metric-consistent by
    construction; it is ``None`` iff form is unavailable and the deliverable abstains
    truthfully (GROUND-R6).

    ``summary_text`` LEADS with a warm, number-light state sentence (COACH-R7); the form
    number is demoted to on-demand grounded backing surfaced only via ``citations`` (the
    grounded canonical form/HRV, GROUND-R5/R7). ``coverage`` is the typed input
    used/unavailable map (from the oracle's ``inputs_used``/``inputs_unavailable``) plus
    any consistency-override caveat; ``suggested_followups`` offers a jargon-free
    reveal-the-numbers prompt (VOICE-R9).
    """

    verdict: ReadinessVerdict | None
    status: RunStatus
    as_of: str | None
    summary_html: str
    summary_text: str
    observations: tuple[Observation, ...] = ()
    citations: tuple[Citation, ...] = ()
    coverage: Mapping[str, Any] | None = None
    suggested_followups: tuple[str, ...] = ()


class _ReadinessNarration(BaseModel):
    """Provider-enforced readiness narration (STRUCT-R1): a state sentence + a verdict.

    The model emits ONLY this closed structure (``extra:forbid``): a warm, number-light
    ``summary_text`` (the state-first lead, COACH-R7) and its proposed ``verdict``. The
    proposed ``verdict`` is NEVER trusted as authoritative — a deterministic gate checks
    it against the metrics (``readiness_consistent``) and the DELIVERED verdict is always
    the canonical :func:`assess_readiness` verdict (COACH-R3 / EVAL-R5, fail-closed).
    """

    model_config = {"extra": "forbid"}
    summary_text: str = ""
    verdict: ReadinessVerdict = ReadinessVerdict.MAINTAIN


@runtime_checkable
class ReadinessGrounder(Protocol):
    """The grounding seam the readiness deliverable drives (GROUND-R1/R2/R7).

    The production :class:`~wattwise_core.agent.engine.ClaimGrounder` implements this:
    given a draft + the canonical-evidence athlete scope, it model-extracts candidate
    claims and CODE-verifies each against canonical analytics, returning a
    :class:`~wattwise_core.agent.contracts.GroundingResult` whose ``scrubbed_text`` has
    every unverifiable number removed and whose survivors carry canonical citations. The
    deliverable reaches grounding ONLY through this seam (ARCH-R21), never the model.
    """

    async def ground(
        self, *, athlete_id: str, draft: str, retrieved: Mapping[str, Any]
    ) -> GroundingResult: ...


class StructuredNarrationError(RuntimeError):
    """The narration model produced no usable structured output (fail-closed marker).

    The narrator closure raises this when the provider cannot yield a schema-valid
    narration; :func:`_run_narration` then falls back to the deterministic per-verdict
    state sentence rather than surfacing a model failure to the athlete (never a guessed
    verdict, never a fabricated number).
    """


#: The model seam the readiness narration uses for structured output (STRUCT-R1). Kept as
#: a narrow callable so the deliverable imports no concrete model and the test injects a
#: ``FakeModel.structured``-backed closure directly. The closure raises
#: :class:`StructuredNarrationError` on a provider failure so narration fails closed.
StructuredNarrator = Callable[[str], Awaitable[_ReadinessNarration]]

# Per-verdict deterministic state sentence (COACH-R7 fallback). Used when no model is
# wired OR when the model narration fails the state-first voice gate: a warm, jargon-free,
# number-LESS lead keyed off the canonical verdict, so the delivered lead is ALWAYS a
# state phrase even if the model misbehaves (fail-closed voice, mirrors GROUND-R3).
_VERDICT_STATE_SENTENCE: Mapping[ReadinessVerdict, str] = {
    ReadinessVerdict.GO: "You're fresh and ready for a hard day.",
    ReadinessVerdict.MAINTAIN: "You're in a steady place — keep things as planned.",
    ReadinessVerdict.EASE: "You're carrying some fatigue, so ease off a little today.",
    ReadinessVerdict.REST: "You're deep in fatigue right now, so today is for rest.",
}

#: The truthful abstain lead when form itself is unavailable (GROUND-R6): no verdict, no
#: number, an honest state sentence rather than a guessed readiness call.
_ABSTAIN_SENTENCE = "There isn't enough recent data to read your readiness yet."

#: The honest HRV-unavailable clause appended to the state sentence when the verdict came
#: from form alone (GROUND-R7: say HRV is missing rather than emit a placeholder). PUBLIC so
#: the eval voice grader (a sibling pack in :mod:`wattwise_core.eval.suites`) can import the
#: EXACT prod text to match a live narration against, rather than hand-maintaining a regex
#: that drifts from this wording (FIX 7). ``_HRV_UNAVAILABLE_CLAUSE`` is kept as a private
#: alias so existing internal references stay valid.
HRV_UNAVAILABLE_CLAUSE = "I don't have a recent HRV reading, so this is from your form."
_HRV_UNAVAILABLE_CLAUSE = HRV_UNAVAILABLE_CLAUSE

#: Any decimal digit. COACH-R7 wants the FIRST sentence number-LIGHT, so the state-first gate
#: rejects a model lead whose first sentence carries ANY digit (not merely a leading one) and
#: falls back to the deterministic, digit-free per-verdict state sentence.
_DIGIT_RE = re.compile(r"[0-9]")


async def readiness_assessment(
    athlete_id: str,
    *,
    form: float | None,
    as_of: str | None,
    hrv_rmssd: float | None,
    hrv_baseline: float | None,
    narrate: StructuredNarrator | None,
    grounder: ReadinessGrounder | None,
    response_length: ResponseLength = "standard",
) -> Readiness:
    """Assemble the readiness/form deliverable from canonical inputs (QA-EVAL-R2.4).

    Inputs are gathered DETERMINISTICALLY by the caller (the readiness JTBD is fixed — it
    does NOT route through the retrieval planner): ``form`` is the latest canonical TSB,
    ``as_of`` its date, ``hrv_rmssd``/``hrv_baseline`` the latest HRV; any unavailable
    input is ``None`` (fail-closed).

    Flow: (1) run the deterministic oracle (:func:`assess_readiness`). (2) If it abstains
    (form unavailable) return a truthful ABSTAIN :class:`Readiness` — no verdict, no
    number, an honest state sentence (GROUND-R6) — with NO model call. (3) Otherwise ask
    the model for a structured narration, run the CODE consistency gate
    (:func:`readiness_consistent`) — the DELIVERED verdict is ALWAYS the oracle's, and a
    mismatch records an override caveat (COACH-R3 / EVAL-R5, fail-closed) — ground the
    narration so numbers are verbatim canonical (GROUND-R7), enforce the state-first /
    number-cap / no-"readiness score" voice gates (COACH-R7 / VOICE-R7), and project.
    """
    assessment = assess_readiness(form=form, hrv_rmssd=hrv_rmssd, hrv_baseline=hrv_baseline)
    verdict = assessment.verdict
    if verdict is None:
        return _abstain_readiness(assessment, as_of)
    return await _narrate_readiness(
        athlete_id,
        verdict=verdict,
        assessment=assessment,
        as_of=as_of,
        hrv_rmssd=hrv_rmssd,
        hrv_baseline=hrv_baseline,
        narrate=narrate,
        grounder=grounder,
        response_length=response_length,
    )


def _abstain_readiness(assessment: ReadinessAssessment, as_of: str | None) -> Readiness:
    """Build the truthful abstain deliverable when form is unavailable (GROUND-R6).

    No verdict, no number, an honest state sentence; the coverage map records the missing
    input from the oracle so the API can render the degradation in coach voice.
    """
    return Readiness(
        verdict=None,
        status=RunStatus.DEGRADED,
        as_of=as_of,
        summary_html=f"<p>{_ABSTAIN_SENTENCE}</p>",
        summary_text=_ABSTAIN_SENTENCE,
        observations=(),
        citations=(),
        coverage=_coverage_map(assessment, override=None),
        suggested_followups=(),
    )


async def _narrate_readiness(
    athlete_id: str,
    *,
    verdict: ReadinessVerdict,
    assessment: ReadinessAssessment,
    as_of: str | None,
    hrv_rmssd: float | None,
    hrv_baseline: float | None,
    narrate: StructuredNarrator | None,
    grounder: ReadinessGrounder | None,
    response_length: ResponseLength,
) -> Readiness:
    """Narrate, gate the verdict, ground the numbers, and project (the assessed path).

    ``verdict`` is the oracle's non-None verdict (the caller handled the abstain case). The
    DELIVERED verdict is ALWAYS this canonical one; a model proposal that disagrees only
    records an override caveat (COACH-R3 / EVAL-R5, fail-closed).
    """
    narration = await _run_narration(narrate, assessment, as_of)
    override = narration is not None and not readiness_consistent(
        narration.verdict, form=assessment.form, hrv_rmssd=hrv_rmssd, hrv_baseline=hrv_baseline
    )
    # On a verdict override the model's lead may describe the WRONG state (e.g. "go" prose
    # under a canonical "rest"); fail closed to the deterministic state sentence so the
    # delivered narration is coherent with the canonical verdict (COACH-R3, mirrors GROUND-R3).
    lead_narration = None if override else narration
    draft = _state_first_draft(lead_narration, verdict, assessment.inputs_unavailable)
    text, html, citations, observations = await _ground_readiness(
        athlete_id, draft, grounder, response_length
    )
    hrv_missing = "hrv" in assessment.inputs_unavailable
    return Readiness(
        verdict=verdict,
        status=RunStatus.DEGRADED if hrv_missing else RunStatus.COMPLETED,
        as_of=as_of,
        summary_html=html,
        summary_text=text,
        observations=observations,
        citations=citations,
        coverage=_coverage_map(assessment, override=override),
        suggested_followups=_readiness_followups(citations),
    )


def _readiness_followups(citations: Sequence[Citation]) -> tuple[str, ...]:
    """No continuation chip: readiness is STATELESS this phase (API-R41).

    A "reveal the numbers" chip implies a durable multi-turn thread this phase does NOT
    maintain — durable readiness threads are a deferred sub-epic, and the response carries no
    ``thread_id`` to target a follow-up against, so the chip would be unactionable (API-R41).
    The grounded form/HRV numbers are ALREADY surfaced inline via ``citations``, so nothing is
    lost by omitting it. ``citations`` is accepted for signature parity with the other
    deliverables' follow-up generators but is intentionally unused.
    """
    return ()


async def _run_narration(
    narrate: StructuredNarrator | None,
    assessment: ReadinessAssessment,
    as_of: str | None,
) -> _ReadinessNarration | None:
    """Obtain the model narration, or ``None`` when no model is wired / it errors.

    A ``None`` narration falls the deliverable back to the deterministic per-verdict state
    sentence — never a fabricated number, never a guessed verdict (fail-closed).
    """
    if narrate is None:
        return None
    try:
        return await narrate(_narration_context(assessment, as_of))
    except (StructuredNarrationError, ValueError):
        return None


def _narration_context(assessment: ReadinessAssessment, as_of: str | None) -> str:
    """The trusted context handed to the narration model (INJECT-R1 user region).

    Carries the canonical verdict + form/HRV values + which inputs were unavailable, so the
    model writes a warm state sentence around the TRUE state. It is told to lead with a
    number-light state phrase and never to call this a "readiness score".
    """
    hrv = f"{assessment.hrv_rmssd:g}" if assessment.hrv_rmssd is not None else "unavailable"
    form = f"{assessment.form:g}" if assessment.form is not None else "unavailable"
    return (
        f"verdict: {assessment.verdict}\n"
        f"form_tsb: {form}\nas_of: {as_of}\n"
        f"hrv_rmssd_ms: {hrv}\n"
        f"inputs_unavailable: {list(assessment.inputs_unavailable)}\n"
        "Write one warm, plain-language state sentence leading the summary; keep numbers "
        "out of the first sentence; never say 'readiness score'."
    )


def _state_first_draft(
    narration: _ReadinessNarration | None,
    verdict: ReadinessVerdict,
    inputs_unavailable: Sequence[str],
) -> str:
    """The draft to ground: the model lead if it passes the voice gates, else the fallback.

    Fail-closed voice (COACH-R7 / VOICE-R7): the model lead is used ONLY if it leads with a
    state phrase, its FIRST sentence carries NO digit (COACH-R7 wants a number-light lead, so
    a digit anywhere in sentence 1 — not just a leading one — fails the gate), AND it carries
    no "readiness score" substring; otherwise the deterministic, digit-free per-verdict state
    sentence is used. When the verdict came from form alone (HRV unavailable) the honest
    HRV-missing clause is appended (GROUND-R7).
    """
    lead = narration.summary_text.strip() if narration is not None else ""
    if (
        not lead
        or not leads_with_state(lead)
        or _has_digit(first_sentence(lead))
        or _mentions_readiness_score(lead)
    ):
        lead = _VERDICT_STATE_SENTENCE[verdict]
    if "hrv" in inputs_unavailable:
        return f"{lead} {_HRV_UNAVAILABLE_CLAUSE}"
    return lead


def _has_digit(text: str) -> bool:
    """True iff ``text`` contains ANY decimal digit (COACH-R7 number-light first sentence)."""
    return _DIGIT_RE.search(text) is not None


def _mentions_readiness_score(text: str) -> bool:
    """True iff the text uses a forbidden numeric 'readiness score' framing (COACH-R7)."""
    return "readiness score" in text.lower()


async def _ground_readiness(
    athlete_id: str,
    draft: str,
    grounder: ReadinessGrounder | None,
    response_length: ResponseLength,
) -> tuple[str, str, tuple[Citation, ...], tuple[Observation, ...]]:
    """Ground the narration and return ``(text, html, citations, observations)`` (GROUND-R5/R7).

    Numbers in the draft are verified verbatim against canonical analytics and surface ONLY
    as grounded citations; an unverifiable number is scrubbed (GROUND-R3). With no grounder
    wired the draft is number-light by construction (the state lead), so it is held to the
    number cap directly. The HTML wraps the grounded text in a paragraph for the API to
    sanitize. Each grounded, citable survivor ALSO projects to a STABLE-id observation
    (COACH-R8: every deliverable's distinct observations carry a stable id) — the
    drill/reveal-numbers handle behind which the demoted form/HRV numbers live (COACH-R7).
    """
    cap = number_cap(response_length)
    if grounder is None:
        text = _demote_numbers(draft, cap) if count_foregrounded_numbers(draft) > cap else draft
        return text, f"<p>{text}</p>", (), ()
    result = await grounder.ground(athlete_id=athlete_id, draft=draft, retrieved={})
    text = result.scrubbed_text
    text, _ = _enforce_number_cap(text, text, cap)
    citations = _readiness_citations(result)
    observations = project_observations(build_observations(result.survivors))
    return text, f"<p>{text}</p>", citations, observations


def _readiness_citations(result: GroundingResult) -> tuple[Citation, ...]:
    """Project the surviving grounded form/HRV numbers into citations (GROUND-R5)."""
    raw = [c.citation for c in result.survivors if c.citation is not None]
    return _project_citations(raw)


def _coverage_map(assessment: ReadinessAssessment, *, override: bool | None) -> Mapping[str, Any]:
    """The typed coverage map from the oracle's inputs + any consistency override caveat.

    ``inputs_used``/``inputs_unavailable`` come straight from the deterministic oracle
    (truthful, never guessed). ``override`` records that the model proposed a verdict the
    metrics did not support, so the canonical verdict was substituted (COACH-R3 / EVAL-R5)
    — the audit trail for the fail-closed decision, mirroring grounding's GROUND-R3 caveat.
    """
    coverage: dict[str, Any] = {
        "inputs_used": list(assessment.inputs_used),
        "inputs_unavailable": list(assessment.inputs_unavailable),
        "rationale": assessment.rationale,
    }
    if override:
        coverage["verdict_override"] = "model_inconsistent_with_metrics"
    return coverage


__all__ = [
    "HRV_UNAVAILABLE_CLAUSE",
    "Readiness",
    "ReadinessGrounder",
    "StructuredNarrationError",
    "StructuredNarrator",
    "readiness_assessment",
]
