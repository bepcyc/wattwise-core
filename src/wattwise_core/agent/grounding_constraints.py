"""Deterministic constraint gate — the always-on contradiction FLOOR (GROUND-R13/R14).

The pure, synchronous, deterministic layer of the constraint-aware grounding design (ADR 0008
§1/§2/§10): after the value gate (GROUND-R1..R9) and the optional entailment gate (GROUND-R11),
a prescription is checked against the athlete's ACTIVE constraints for a CONTRADICTION. This is
NOT the 3-way NLI verifier ADR 0008 §1 proposes (that model layer is a documented opt-in seam,
default OFF and not implemented here) — it is the deterministic FLOOR ADR 0008 §10.1 calls for:
a high-confidence, structured/lexical activity-term match that catches the OBVIOUS contradictions
(a "no running" constraint vs a "run 5x4 min" prescription) without a model, and is deliberately
CONSERVATIVE so it does not over-block (an "easy swim" against "no running" is NEUTRAL — it must
still publish; the necessity-of-contradiction-not-support case of ADR 0008 §1). Paraphrase is the
deferred NLI layer; this floor only flags a clear lexical activity-term contradiction.

Severity drives the outcome, mirroring ACSM's absolute/relative contraindication split (ADR 0008
§2) and the ``_downgrade_for_sweep`` discipline in :mod:`grounding`:

* **HARD (absolute)** → a VETO: the violating sentence span is scrubbed from the published text and
  the aggregate decision is forced off ``proceed`` (REGENERATE when prose survives, ABSTAIN when
  nothing does) — identical handling to a contradicted NUMBER (GROUND-R9), so a contraindicated
  plan is never published and a human is never asked to approve one.
* **SOFT (relative)** → a CAUTION: the prescription is NOT scrubbed (a silent blanket veto would
  re-introduce the inverse #77 harm — over-refusal); instead the contradiction is surfaced as a
  structured caution note the caller threads to the athlete (the shared-decision StARRT stance).

Everything here is a pure function of its text inputs (GRAPH-R4): no model, no IO, no awaiting, so
the same inputs always yield the same result. The wiring that extracts the active constraints from
graph state and records the observability counters lives in the grounder (the gate stays pure).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from wattwise_core.agent.contracts import GroundDecision, GroundingResult
from wattwise_core.agent.memory import ConstraintSeverity


class ConstraintMode(StrEnum):
    """Deterministic constraint-floor rollout mode (CFG-R1a, ADR 0008 §7).

    ``off`` skips the gate; ``shadow`` DETECTS would-be vetoes/cautions and reports them for the
    observability counters but applies NOTHING (no scrub, no caution surfaced — the rollout step);
    ``enforce`` applies HARD vetoes and SOFT cautions. The deterministic floor is precise, so the
    OSS default is ``enforce`` (ADR 0008 §7). The bound enum fails the boot closed on a bad value.
    """

    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


# Sentence splitter shared with the entailment gate's discipline (sentence-terminator OR newline):
# the gate scrubs/flags at the SENTENCE granularity so a single contraindicated prescription is
# removed without taking the surrounding grounded prose with it.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# --- the deterministic multilingual FLOOR lexicon (ADR 0008 §10.1, deliberately minimal) ---
#
# This is the DETERMINISTIC-FLOOR lexicon, NOT the NLI model: it is intentionally SMALL and
# CONSERVATIVE (English / German / Russian, matching the repo's supported languages, LANG-R1) so it
# flags only OBVIOUS, high-confidence activity-term contradictions. Paraphrase, synonymy, and
# cross-lingual entailment are the deferred GROUND-R13 NLI layer — not attempted here. An operator
# extends the floor by editing this map (it is the floor's whole vocabulary); the model layer is the
# real generalization path.

#: Negation cue words that, when preceding an activity term in a constraint, mark that activity as
#: FORBIDDEN ("no running", "avoid running", "keine Intervalle", "не бегать"). Word-bounded,
#: case-folded at match time. Russian negation ("нет"/"не"/"без") + German ("kein…"/"vermeiden") +
#: English ("no"/"avoid"/"don't"/"cannot"/"stop"/"without") covers the explicit-limit
#: phrasings the capture path records.
_NEGATION_CUES: frozenset[str] = frozenset(
    {
        # English
        "no",
        "not",
        "avoid",
        "avoiding",
        "stop",
        "cannot",
        "cant",
        "dont",
        "without",
        "skip",
        "never",
        # German
        "kein",
        "keine",
        "keinen",
        "keiner",
        "nicht",
        "vermeiden",
        "ohne",
        # Russian
        "нет",
        "не",
        "без",
        "избегать",
    }
)

#: Activity terms the floor recognizes, grouped so a constraint phrased in one surface form and a
#: prescription phrased in another (incl. cross-lingual, ADR 0008 §8) still match: every member of a
#: group maps to the SAME canonical activity token. Multilingual by design (EN/DE/RU). Minimal and
#: clearly the deterministic floor — synonymy beyond these groups is the deferred NLI layer.
_ACTIVITY_GROUPS: tuple[tuple[str, ...], ...] = (
    ("run", "running", "runs", "ran", "laufen", "lauf", "бег", "бегать", "побегать"),  # noqa: RUF001 - multilingual (RU) activity lexicon
    ("interval", "intervals", "intervalle", "intervall", "интервал", "интервалы"),
    ("sprint", "sprints", "sprinting", "спринт", "спринты"),
    ("jump", "jumps", "jumping", "plyometric", "plyometrics", "springen", "прыжки", "прыжок"),
    ("swim", "swimming", "swims", "schwimmen", "плавание", "плавать"),
    ("ride", "riding", "cycling", "bike", "biking", "radfahren", "велосипед", "езда"),
    ("lift", "lifting", "weights", "deadlift", "deadlifts", "squat", "squats", "heben", "тяга"),
    ("hill", "hills", "climb", "climbs", "climbing", "berg", "berge", "холм", "холмы"),
)

#: activity-surface-term -> its canonical activity token (the membership test the gate uses).
_ACTIVITY_TERM_TO_TOKEN: dict[str, str] = {
    term.casefold(): group[0] for group in _ACTIVITY_GROUPS for term in group
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class CautionNote:
    """A surfaced SOFT-constraint caution (GROUND-R14): the prescription stays, flagged.

    ``constraint`` is the active SOFT constraint's content (the athlete's own words) and
    ``prescription`` the violating sentence the gate detected. The caller threads this to the
    athlete as a shared-decision prompt ("I see you noted X — is Y still off the table?"), never a
    silent scrub (ADR 0008 §2): a silent blanket veto would re-introduce the inverse #77 harm.
    """

    constraint: str
    prescription: str
    activity: str


@dataclass(frozen=True, slots=True)
class ConstraintGateResult:
    """Typed outcome of the deterministic constraint gate the grounder integrates (GROUND-R13/R14).

    ``result`` is the (possibly veto-scrubbed + decision-downgraded) :class:`GroundingResult`;
    ``hard_violations`` / ``cautions`` are the structured records of what the gate did (for the
    observability counters and the athlete-facing caution channel). When the gate found nothing the
    ``result`` is the input unchanged and both lists are empty.
    """

    result: GroundingResult
    hard_violations: tuple[str, ...] = ()
    cautions: tuple[CautionNote, ...] = ()


@dataclass(frozen=True, slots=True)
class ActiveConstraint:
    """One active constraint as the pure gate sees it: the athlete's words + its severity.

    The pure gate's input projection (ADR 0008 §3): the constraint ``content`` (the athlete's own
    stated limit) and its :class:`~wattwise_core.agent.memory.ConstraintSeverity`. The wiring builds
    these from the recalled active-constraint set threaded through graph state; the gate never reads
    a store (it stays pure, GRAPH-R4).
    """

    content: str
    severity: ConstraintSeverity = ConstraintSeverity.SOFT
    #: The pre-computed forbidden-activity token set, derived from ``content`` via the lexicon.
    forbidden: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_content(cls, content: str, severity: ConstraintSeverity) -> ActiveConstraint:
        """Build an active constraint, deriving its forbidden-activity token set (GROUND-R13)."""
        return cls(content=content, severity=severity, forbidden=forbidden_activities(content))


def forbidden_activities(constraint_text: str) -> frozenset[str]:
    """Extract the forbidden-activity canonical tokens from a constraint's text (GROUND-R13 floor).

    The structured/lexical extraction of the deterministic floor (ADR 0008 §10.1): an activity term
    is FORBIDDEN when a negation cue word appears NEAR it (within a small word window, or in
    the same short constraint sentence) — "no running", "avoid intervals", "keine Intervalle",
    "не бегать". Returns the set of CANONICAL activity tokens (so a constraint and a prescription
    phrased differently — incl. cross-lingual — still match). CONSERVATIVE by construction: an
    activity term with NO nearby negation cue yields nothing (a constraint like "I enjoy running"
    forbids nothing), and a term outside the small lexicon is ignored (paraphrase is the NLI
    layer). Pure and deterministic.
    """
    words = [m.group(0).casefold() for m in _WORD_RE.finditer(constraint_text)]
    forbidden: set[str] = set()
    for index, word in enumerate(words):
        token = _ACTIVITY_TERM_TO_TOKEN.get(word)
        if token is None:
            continue
        # A negation cue within a small window before/after the activity term marks it forbidden.
        window = words[max(0, index - _NEGATION_WINDOW) : index + _NEGATION_WINDOW + 1]
        if any(cue in _NEGATION_CUES for cue in window):
            forbidden.add(token)
    return frozenset(forbidden)


#: How many words on either side of an activity term a negation cue may sit and still forbid it.
#: Small + fixed: the floor catches "no hard running", "running is not allowed" — not a negation a
#: clause away (that ambiguity is for the NLI layer).
_NEGATION_WINDOW = 4


def _sentence_activities(sentence: str) -> set[str]:
    """The canonical activity tokens a prescriptive sentence mentions (GROUND-R13)."""
    activities: set[str] = set()
    for match in _WORD_RE.finditer(sentence):
        token = _ACTIVITY_TERM_TO_TOKEN.get(match.group(0).casefold())
        if token is not None:
            activities.add(token)
    return activities


def apply_constraint_gate(
    result: GroundingResult,
    constraints: Sequence[ActiveConstraint],
    *,
    prescriptive_sentences: Sequence[str] | None = None,
) -> ConstraintGateResult:
    """Gate a grounded draft against active constraints (GROUND-R13/R14, the deterministic floor).

    For each PRESCRIPTIVE sentence of the published text, detect a CONTRADICTION with an active
    constraint by canonical activity-term overlap (a forbidden activity the sentence prescribes):

    * a HARD-severity match is a VETO — the violating sentence is scrubbed from the published text
      and the decision is forced off ``proceed`` (mirroring ``_downgrade_for_sweep``): REGENERATE
      when grounded prose survives, ABSTAIN when nothing does;
    * a SOFT-severity match is a CAUTION — the sentence STAYS (no scrub) and is recorded as a
      :class:`CautionNote` the caller surfaces to the athlete.

    ``prescriptive_sentences`` optionally restricts the scan to sentences the upstream extractor
    flagged prescriptive; when ``None`` every published sentence is scanned (the conservative
    default — the floor only ever fires on an activity-term contradiction regardless, so scanning a
    non-prescriptive sentence that happens to forbid an activity is acceptable fail-closed for HARD
    and merely surfaces a caution for SOFT). Pure and deterministic — no model, no IO (GRAPH-R4).
    """
    text = result.scrubbed_text
    if not text.strip() or not constraints:
        return ConstraintGateResult(result=result)
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    scannable = (
        {s.strip() for s in prescriptive_sentences} if prescriptive_sentences is not None else None
    )
    hard_violations: list[str] = []
    cautions: list[CautionNote] = []
    veto_sentences: list[str] = []
    for sentence in sentences:
        if scannable is not None and sentence not in scannable:
            continue
        activities = _sentence_activities(sentence)
        if not activities:
            continue
        for constraint in constraints:
            hit = activities & constraint.forbidden
            if not hit:
                continue
            activity = sorted(hit)[0]
            if constraint.severity is ConstraintSeverity.HARD:
                hard_violations.append(sentence)
                veto_sentences.append(sentence)
            else:
                cautions.append(
                    CautionNote(
                        constraint=constraint.content, prescription=sentence, activity=activity
                    )
                )
    if not veto_sentences:
        return ConstraintGateResult(result=result, hard_violations=(), cautions=tuple(cautions))
    scrubbed = _remove_sentences(text, veto_sentences)
    decision = _downgrade_for_veto(result.decision, scrubbed)
    gated = GroundingResult(decision=decision, claims=result.claims, scrubbed_text=scrubbed)
    return ConstraintGateResult(
        result=gated, hard_violations=tuple(hard_violations), cautions=tuple(cautions)
    )


def _remove_sentences(text: str, vetoed: Sequence[str]) -> str:
    """Remove each vetoed sentence from ``text`` (first occurrence; whitespace tidied).

    Mirrors the entailment gate's ``_remove_sentences``: the violating prescription is removed from
    the PUBLISHED text (the artifact that would ship); on any veto the run re-drafts, so this
    round's claims are discarded with the draft (the per-claim verdicts are not rewritten here).
    """
    edited = text
    for sentence in vetoed:
        idx = edited.find(sentence)
        if idx == -1:
            continue
        edited = edited[:idx] + edited[idx + len(sentence) :]
    edited = re.sub(r"\s{2,}", " ", edited)
    return re.sub(r"\s+([.,;:!?])", r"\1", edited).strip()


def _downgrade_for_veto(decision: GroundDecision, remaining_text: str) -> GroundDecision:
    """Force a non-``proceed`` decision after a HARD veto (fail-closed, mirrors GROUND-R9).

    A HARD contradiction is handled like a contradicted NUMBER (ADR 0008 §2): never published, the
    run re-drafts. ``proceed`` downgrades to ``regenerate`` when grounded prose survives the scrub,
    to ``abstain`` when nothing publishable remains (GROUND-R6). An already-recovering/abstaining
    decision is kept (the veto only ever strengthens the gate).
    """
    if not remaining_text.strip():
        return GroundDecision.ABSTAIN
    if decision is GroundDecision.PROCEED:
        return GroundDecision.REGENERATE
    return decision


class ConstraintGate:
    """Mode-aware wrapper over the pure constraint gate (GROUND-R13/R14, ADR 0008 §7).

    Carries the rollout :class:`ConstraintMode`. ``apply`` runs the pure
    :func:`apply_constraint_gate` to DETECT contradictions, then — in ``shadow`` — returns the
    DETECTION (the ``hard_violations`` / ``cautions`` the caller counts) but the ORIGINAL,
    unmodified result, applying neither the scrub/downgrade nor the surfaced cautions; in
    ``enforce`` it returns the gated result with the veto applied. ``off`` is never constructed
    (``from_settings`` returns ``None``). The wrapper holds no state beyond the mode, so it is
    safe to share across runs.
    """

    def __init__(self, mode: ConstraintMode = ConstraintMode.ENFORCE) -> None:
        self.mode = mode

    def apply(
        self,
        result: GroundingResult,
        constraints: Sequence[ActiveConstraint],
        *,
        prescriptive_sentences: Sequence[str] | None = None,
    ) -> ConstraintGateResult:
        """Detect (always) and — in ``enforce`` — apply the constraint gate (ADR 0008 §7)."""
        detected = apply_constraint_gate(
            result, constraints, prescriptive_sentences=prescriptive_sentences
        )
        if self.mode is ConstraintMode.ENFORCE:
            return detected
        # SHADOW: report the detection on the counters, but ship the UNMODIFIED result (the rollout
        # step before an operator promotes the floor to enforce; cautions/vetoes are not applied).
        return ConstraintGateResult(
            result=result,
            hard_violations=detected.hard_violations,
            cautions=detected.cautions,
        )


def gate_from_settings(settings: Any) -> ConstraintGate | None:
    """Build the deterministic constraint gate from ``[agent.constraints]`` (CFG-R1a, ADR 0008 §7).

    ``None`` when ``mode = "off"`` (the gate does not run). Otherwise a :class:`ConstraintGate` in
    the configured mode — ``shadow`` (detect + record, apply nothing) or ``enforce`` (the OSS
    default: HARD vetoes, SOFT cautions applied). A bad mode value fails the boot closed via the
    :class:`ConstraintMode` enum (CFG-R1a). NOTE: ``agent__constraints__enabled`` gates ONLY the
    deferred NLI model layer (ADR 0008 §1, not implemented here) — the DETERMINISTIC floor this gate
    runs is governed solely by ``mode``, so it enforces by default independent of ``enabled``.
    """
    mode = ConstraintMode(settings.agent__constraints__mode)
    if mode is ConstraintMode.OFF:
        return None
    return ConstraintGate(mode=mode)


__all__ = [
    "ActiveConstraint",
    "CautionNote",
    "ConstraintGate",
    "ConstraintGateResult",
    "ConstraintMode",
    "apply_constraint_gate",
    "forbidden_activities",
    "gate_from_settings",
]
