"""Additional CI-gated eval suites driving the PRODUCTION engine (EVAL-R2a/-R3/-R5/-R7).

These suites exercise the shipped code paths the core grounding/abstention/injection
suites do not:

* ``termination`` (EVAL-R7 / REFLECT-R4) — two fixtures driving the PRODUCTION
  :func:`~wattwise_core.agent.graph.build_graph`: a perpetually-insufficient-coverage run
  that must terminate at the ``reflection_count`` bound, and a perpetually-failing-to-ground
  run that must terminate at the ``redraft_count`` bound — both ``degraded``, never an
  unbounded loop, never an error/budget_exceeded.
* ``intent_plan`` (EVAL-R3 / PLAN-R*) — a labelled dataset scoring the planner's emitted
  capability requests for precision + recall (>= 0.9 gate).
* ``multilingual`` (EVAL-R7a / LANG-R*) — the SAME grounded fixture rendered EN/DE/RU must
  carry identical numbers/citations with no untranslated internal token.
* ``judge`` (EVAL-R5) — an LLM-as-judge rubric scored via a provider-enforced structured
  output (recorded offline), never certifying grounding/abstention/injection/status.

Everything here is deterministic and network-free (TIER-R1, QA-EVAL-R9).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from wattwise_core.agent.contracts import (
    AgentState,
    Claim,
    ClaimKind,
    GroundDecision,
    GroundedClaim,
    GroundingResult,
    GroundVerdict,
    ReflectDecision,
    ReflectVerdict,
    RetrievalRequest,
    RunStatus,
)
from wattwise_core.agent.graph import MAX_REDRAFTS, MAX_REFLECTIONS, AgentServices, build_graph
from wattwise_core.agent.model import FakeModel
from wattwise_core.agent.readiness_deliverable import HRV_UNAVAILABLE_CLAUSE
from wattwise_core.analytics.readiness import readiness_consistent
from wattwise_core.domain.enums import ReadinessVerdict
from wattwise_core.eval.grading import (
    READINESS_MAX_NUMBERS,
    JudgeGrade,
    ReadinessGrade,
    TerminationGrade,
)
from wattwise_core.eval.intent_plan_suite import grade_intent_plan  # re-export (EVAL-R3)
from wattwise_core.eval.prose_checks import MAX_FK_GRADE, detect_language, flesch_kincaid_grade

_DATASETS_DIR = Path(__file__).parent / "datasets"
_INTERNAL_TOKEN = re.compile(r"(ctl|atl|tsb|rmssd|coverage_gaps|__truncated__)", re.IGNORECASE)
# A foregrounded numeric value in athlete-facing prose (mirrors the deliverables
# number-density regex): plain signed integers/decimals, standalone so words/dates are
# not miscounted. Used by the readiness voice-liveness count (COACH-R7 / QA-EVAL-R11).
_SUMMARY_NUMBER_RE = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w.])")
# A leading STATE sentence must contain NO digit at all (the bounded form number is
# demoted, never the headline — COACH-R7 / QA-EVAL-R2.12).
_FIRST_SENTENCE_RE = re.compile(r"^[^.!?]*")
# A numeric "readiness score" is forbidden: readiness is a typed STATE, not a number
# (SCHEMA-R3). Catch "readiness score", "readiness: <n>", "readiness <n>", "readiness=<n>".
_READINESS_SCORE_RE = re.compile(r"readiness\s*(?:score|[:=]\s*[+-]?\d|\s+[+-]?\d)", re.IGNORECASE)
# An HRV-unavailable summary must SAY HRV/heart-rate-variability is unknown (GROUND-R7).
_HRV_MENTION_RE = re.compile(r"\bhrv\b|heart[- ]rate[- ]variability", re.IGNORECASE)
# The HRV-absent phrasing the grader accepts. FIX 7: the REAL prod clause
# (``HRV_UNAVAILABLE_CLAUSE`` = "I don't have a recent HRV reading, so this is from your
# form.") was NOT matched by the old pattern, so the check only ever passed on hand-written
# fixtures. The PRIMARY match is the exact canonical clause (substring, in
# ``_states_hrv_unavailable``); this regex is the fallback for hand-fixture phrasings and
# carries only genuine ABSENCE tokens. We deliberately do NOT include the bare token
# "from your form": it false-positives on prose that mentions HRV POSITIVELY (e.g.
# "Your HRV is strong, momentum comes from your form") — absence must be stated, not implied.
_HRV_ABSENT_RE = re.compile(
    r"\b(?:unavailable|not available|wasn't available|was not available|"
    r"wasn't recorded|was not recorded|no hrv|unknown|not guessing|"
    r"leans on form|don't have a recent hrv|do not have a recent hrv|"
    r"no recent hrv|without (?:a |an )?(?:recent )?hrv)\b",
    re.IGNORECASE,
)


def _load(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(
        (_DATASETS_DIR / f"{name}.json").read_text(encoding="utf-8")
    )
    return loaded


# --- termination suite (EVAL-R7) -------------------------------------------------------


class _StubPlanner:
    async def plan(
        self, *, request_text: str | None, gaps: Any, already: Any
    ) -> list[RetrievalRequest]:
        return [RetrievalRequest(capability="weekly_load", params={})]


class _StubGateway:
    async def gather(self, *, athlete_id: str, requests: Any) -> dict[str, Any]:
        return {"rec": {"value": 1.0, "relevance": 1.0}}


class _GapCoverage:
    def __init__(self, gaps: set[str]) -> None:
        self._gaps = gaps

    def assess(self, *, request_text: str | None, retrieved: Any) -> set[str]:
        return set(self._gaps)


class _ScriptedGrounder:
    def __init__(self, decision: GroundDecision) -> None:
        self._decision = decision

    async def ground(self, *, athlete_id: str, draft: str, retrieved: Any) -> GroundingResult:
        claim = Claim(kind=ClaimKind.NUMBER, text="1", value=1.0, metric="ctl")
        survivor = GroundedClaim(claim=claim, verdict=GroundVerdict.GROUNDED, citation={"m": "ctl"})
        return GroundingResult(decision=self._decision, claims=(survivor,), scrubbed_text=draft)


class _ReflectModel(FakeModel):
    """A model whose structured reflect verdict always asks to REPLAN (drive the bound)."""

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M:
        if schema is ReflectDecision:
            return ReflectDecision(verdict=ReflectVerdict.REPLAN)  # type: ignore[return-value]
        raise NotImplementedError(schema.__name__)


def _termination_services(*, gaps: set[str], decision: GroundDecision) -> AgentServices:
    return AgentServices(
        planner=_StubPlanner(),
        gateway=_StubGateway(),
        coverage=_GapCoverage(gaps),
        grounder=_ScriptedGrounder(decision),
    )


async def _run_termination_case(case: dict[str, Any]) -> tuple[str, bool, str]:
    """Drive the production graph to its bound; return (id, bounded_ok, reason)."""
    bound = str(case["bound"])
    gaps = {"missing"} if bound == "reflection_count" else set()
    decision = GroundDecision.REGENERATE if bound == "redraft_count" else GroundDecision.PROCEED
    svc = _termination_services(gaps=gaps, decision=decision)
    graph = build_graph(_ReflectModel(), svc, InMemorySaver())
    state: AgentState = {
        "athlete_id": "athlete-term",
        "trigger": "user_turn",
        "request_text": "q",
        "locale": "en",
        "idempotency_key": case["id"],
    }
    out = await graph.ainvoke(state, config={"configurable": {"thread_id": case["id"]}})
    status_ok = out.get("status") is RunStatus.DEGRADED
    bound_hit = (
        out.get("reflection_count", 0) == MAX_REFLECTIONS
        if bound == "reflection_count"
        else out.get("redraft_count", 0) == MAX_REDRAFTS
    )
    ok = status_ok and bound_hit
    reason = (
        ""
        if ok
        else (
            f"status={out.get('status')} reflect={out.get('reflection_count')} "
            f"redraft={out.get('redraft_count')}"
        )
    )
    return case["id"], ok, reason


async def grade_termination() -> TerminationGrade:
    """Run the EVAL-R7 termination fixtures through the production graph."""
    cases = _load("termination")["cases"]
    failures: list[str] = []
    bounded = 0
    for case in cases:
        cid, ok, reason = await _run_termination_case(case)
        if ok:
            bounded += 1
        else:
            failures.append(f"{cid}: {reason}")
    return TerminationGrade(len(cases), bounded, tuple(failures))


# --- readiness / form suite (QA-EVAL-R2.4) ---------------------------------------------


def _first_sentence(text: str) -> str:
    """The leading athlete-facing sentence (up to the first . ! or ?)."""
    match = _FIRST_SENTENCE_RE.match(text.strip())
    return match.group(0) if match else text.strip()


def _state_leads(summary: str) -> bool:
    """True iff the FIRST sentence is number-light: it carries NO digit (COACH-R7).

    The bounded form number must be demoted out of the headline, so a leading sentence
    with any digit fails (QA-EVAL-R2.12 / VOICE — readiness leads with STATE, not a
    number).
    """
    return not any(ch.isdigit() for ch in _first_sentence(summary))


def _number_density_ok(summary: str) -> bool:
    """True iff the summary foregrounds <= READINESS_MAX_NUMBERS numeric values."""
    return len(_SUMMARY_NUMBER_RE.findall(summary)) <= READINESS_MAX_NUMBERS


def _no_readiness_score(summary: str) -> bool:
    """True iff the summary carries NO numeric 'readiness score' (SCHEMA-R3)."""
    return _READINESS_SCORE_RE.search(summary) is None


def _states_hrv_unavailable(summary: str) -> bool:
    """True iff the summary explicitly says HRV was unavailable/unknown (GROUND-R7).

    FIX 7: a summary carrying the EXACT canonical prod clause
    (:data:`~wattwise_core.agent.readiness_deliverable.HRV_UNAVAILABLE_CLAUSE`) is
    recognised directly, so the grader matches a LIVE narration and not only hand-written
    fixtures; otherwise it falls back to the HRV-mention + HRV-absent token pair (the
    absent-token pattern now includes that clause's key tokens too).
    """
    if HRV_UNAVAILABLE_CLAUSE.lower() in summary.lower():
        return True
    mentions_hrv = _HRV_MENTION_RE.search(summary) is not None
    says_absent = _HRV_ABSENT_RE.search(summary) is not None
    return mentions_hrv and says_absent


def _states_insufficient_data(summary: str) -> bool:
    """True iff an abstain summary truthfully says it cannot assess (GROUND-R6)."""
    lowered = summary.lower()
    return any(
        phrase in lowered
        for phrase in (
            "not enough data",
            "don't have enough",
            "insufficient",
            "can't give you",
            "can't call",
            "cannot assess",
            "hasn't been computed",
            "not been computed",
        )
    )


def _voice_failures(case: dict[str, Any], summary: str) -> list[str]:
    """Deterministic voice-liveness checks on ONE summary (COACH-R7 / QA-EVAL-R11)."""
    cid = case["id"]
    out: list[str] = []
    if not _state_leads(summary):
        out.append(f"{cid}: summary is number-led (first sentence carries a digit)")
    if not _number_density_ok(summary):
        out.append(f"{cid}: summary foregrounds more than {READINESS_MAX_NUMBERS} numbers")
    if not _no_readiness_score(summary):
        out.append(f"{cid}: summary carries a numeric 'readiness score'")
    if case.get("expects_hrv_unavailable_statement") and not _states_hrv_unavailable(summary):
        out.append(f"{cid}: HRV was unavailable but the summary does not say so")
    return out


def _is_abstain(case: dict[str, Any]) -> bool:
    """True iff the case is an abstain case — i.e. ``form`` is null (FIX 4).

    Abstain means "we could not read form". The classification is FORM-driven ONLY: a
    case WITH a form is NON-abstain even if its delivered verdict is null (that is then a
    failure, not a free pass), and a case with a null form is abstain regardless of the
    delivered verdict (a delivered verdict on a form-null case is itself a failure).
    """
    return case.get("form") is None


def _consistency_failure(case: dict[str, Any]) -> str | None:
    """Certify the delivered verdict against the deterministic band (QA-EVAL-R2.4).

    Classification is FORM-driven (FIX 4): a case is abstain IFF ``form`` is null.

    * Abstain (``form`` null): the delivered verdict MUST be null AND the summary MUST
      truthfully state insufficient data; a delivered verdict on a form-null case fails.
    * Non-abstain (``form`` present): the delivered verdict MUST be non-null AND pass the
      :func:`readiness_consistent` certificate (the code, not the LLM, decides — EVAL-R5).
      A present-form case with a NULL delivered verdict is a FAILURE ("form present but no
      verdict delivered"), never a silent abstain.

    Returns a reason string on failure, or ``None`` when the case is consistent.
    """
    cid = case["id"]
    delivered = case.get("delivered_verdict")
    form = case.get("form")
    # Abstain IFF form is null (FORM-driven, FIX 4). The explicit ``form is None`` check
    # (equivalent to ``_is_abstain(case)``) also narrows ``form`` to non-None below.
    if form is None:
        if delivered is not None:
            return f"{cid}: abstain case (form null) delivered a verdict ({delivered!r})"
        if not _states_insufficient_data(str(case.get("summary_text", ""))):
            return f"{cid}: abstain summary does not state insufficient data"
        return None
    if delivered is None:
        return f"{cid}: form present but no verdict delivered"
    consistent = readiness_consistent(
        ReadinessVerdict(str(delivered)),
        form=float(form),
        hrv_rmssd=case.get("hrv_rmssd"),
        hrv_baseline=case.get("hrv_baseline"),
    )
    if not consistent:
        return f"{cid}: delivered verdict {delivered!r} is inconsistent with the metrics"
    return None


def grade_readiness() -> ReadinessGrade:
    """Grade the readiness/form fixtures deterministically (QA-EVAL-R2.4 / COACH-R7).

    Classification is FORM-driven (FIX 4): a case is abstain IFF ``form`` is null. For
    each NON-abstain case (``form`` present) the delivered verdict MUST be non-null and is
    certified against the deterministic band via :func:`readiness_consistent` (deep-negative
    form is never a hard "go"; the code decides, not the LLM — EVAL-R5) — a present-form
    case with a null delivered verdict is a FAILURE, never a silent abstain. Abstain cases
    (``form`` null) must deliver NO verdict and a summary that says it cannot assess. EVERY
    case's summary is checked for voice-liveness: a number-light STATE-first sentence, number
    density within the cap, no numeric "readiness score", and an explicit HRV-unavailable
    statement where the inputs were absent (GROUND-R7).
    """
    cases = _load("readiness")["cases"]
    failures: list[str] = []
    non_abstain = 0
    consistent = 0
    voice_ok = 0
    for case in cases:
        is_abstain = _is_abstain(case)  # FORM-driven only (FIX 4): abstain IFF form null.
        if not is_abstain:
            non_abstain += 1
        reason = _consistency_failure(case)
        if reason is None:
            if not is_abstain:
                consistent += 1
        else:
            failures.append(reason)
        voice_problems = _voice_failures(case, str(case.get("summary_text", "")))
        if voice_problems:
            failures.extend(voice_problems)
        else:
            voice_ok += 1
    return ReadinessGrade(len(cases), non_abstain, consistent, voice_ok, tuple(failures))


# --- multilingual rendering suite (EVAL-R7a) -------------------------------------------


def grade_multilingual() -> tuple[int, tuple[str, ...]]:
    """Assert EN/DE/RU renders carry identical numbers/citations + no internal token.

    QA-EVAL-R2.8 adds two case kinds beyond the EN/DE/RU parity render: a mid-conversation
    language ``switch`` (the SAME grounded numbers + citation persist across the switch) and
    an unsupported-language ``fallback`` (fall back to English AND carry a human-readable
    notice). Both reuse the number/citation-parity core; ``fallback`` additionally asserts
    the fallback locale rendered and the notice tokens are present. Every render is ALSO
    language-detected programmatically (:func:`detect_language`): an answer authored in the
    wrong language fails even when its numbers are right (QA-EVAL-R2.8 (a)).
    """
    cases = _load("multilingual")["cases"]
    failures: list[str] = []
    for case in cases:
        failures.extend(_multilingual_parity_failures(case))
        if str(case.get("kind", "")) == "fallback":
            failures.extend(_multilingual_fallback_failures(case))
    return len(cases), tuple(failures)


def _multilingual_parity_failures(case: dict[str, Any]) -> list[str]:
    """Number/citation parity + jargon-free checks shared by every multilingual case."""
    failures: list[str] = []
    renders = case["renders"]
    numbers = {lang: sorted(re.findall(r"-?\d+(?:\.\d+)?", text)) for lang, text in renders.items()}
    first = next(iter(numbers.values()))
    if any(nums != first for nums in numbers.values()):
        failures.append(f"{case['id']}: numbers differ across languages {numbers}")
    for lang, text in renders.items():
        # Internal jargon tokens (ctl/rmssd/coverage_gaps) must not leak untranslated into a
        # localized render (EVAL-R7a jargon-free); the en render may name a metric in prose.
        if _INTERNAL_TOKEN.search(text) and lang != "en":
            failures.append(f"{case['id']}/{lang}: leaked internal token")
        # QA-EVAL-R2.8 (a): the answer must be AUTHORED in the selected language — checked
        # programmatically (QA-EVAL-R3), never assumed from the render key.
        detected = detect_language(text)
        if detected != lang:
            failures.append(f"{case['id']}/{lang}: output language detected as {detected!r}")
    if sorted(case["expected_citations"]) != sorted(case.get("citations", [])):
        failures.append(f"{case['id']}: citations changed across languages")
    return failures


def _multilingual_fallback_failures(case: dict[str, Any]) -> list[str]:
    """Assert an unsupported-language case fell back to English + carries a notice (R2.8)."""
    failures: list[str] = []
    cid = case["id"]
    fallback = str(case.get("fallback_locale", "en"))
    text = case["renders"].get(fallback)
    if text is None:
        failures.append(f"{cid}: unsupported locale did not render the {fallback!r} fallback")
        return failures
    for token in case.get("notice_tokens", []):
        if str(token).lower() not in text.lower():
            failures.append(f"{cid}: fallback render is missing the human-readable notice")
            break
    return failures


# --- LLM-as-judge rubric suite (EVAL-R5) -----------------------------------------------


class JudgeVerdict(BaseModel):
    """Provider-enforced structured judge verdict over the qualitative rubric (EVAL-R5)."""

    coherence: int
    tone: int
    coach_voice: int
    actionability: int
    clarity: int


_JUDGE_DIMS = ("coherence", "tone", "coach_voice", "actionability", "clarity")
_JUDGE_MIN = 3  # 1..5 rubric; below 3 fails the case


async def grade_judge() -> JudgeGrade:
    """Score deliverables with an LLM-as-judge structured rubric (recorded offline, EVAL-R5).

    The judge runs in recorded-response mode: a :class:`FakeModel` returns the dataset's
    recorded :class:`JudgeVerdict` for each case (no network). The judge scores tone/voice/
    clarity ONLY; it NEVER certifies grounding/abstention/injection/status.
    """
    cases = _load("judge")["cases"]
    failures: list[str] = []
    passed = 0
    lowest = 5.0
    for case in cases:
        recorded = JudgeVerdict(**case["recorded_scores"])
        model = FakeModel(scripted={JudgeVerdict.__name__: recorded})
        verdict = await model.structured(system="judge", data=case["body"], schema=JudgeVerdict)
        scores = [getattr(verdict, dim) for dim in _JUDGE_DIMS]
        lowest = min(lowest, *scores)
        # Deterministic reading-level dimension (QUAL-R13(h)/(i)): the athlete-facing
        # body must stay at or under the plain-language ceiling - graded by code, not
        # by the LLM judge, so a jargon-dense regression trips the gate mechanically.
        grade_level = flesch_kincaid_grade(str(case["body"]))
        reading_ok = grade_level <= MAX_FK_GRADE
        if all(s >= _JUDGE_MIN for s in scores) and reading_ok:
            passed += 1
        elif not reading_ok:
            failures.append(
                f"{case['id']}: reading level {grade_level:.1f} exceeds the "
                f"plain-language ceiling ({MAX_FK_GRADE:.0f}th grade)"
            )
        else:
            failures.append(f"{case['id']}: a rubric dimension scored below {_JUDGE_MIN}")
    return JudgeGrade(len(cases), passed, lowest, tuple(failures))


__all__ = [
    "JudgeVerdict",
    "grade_intent_plan",
    "grade_judge",
    "grade_multilingual",
    "grade_readiness",
    "grade_termination",
]
