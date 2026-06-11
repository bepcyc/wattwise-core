"""Decorrelated sentence-entailment gate over the grounded draft (issue #10 Phase 2).

The proposed GROUND-R11 layer of the binding-faithful grounding design: after the
deterministic value gate (GROUND-R1..R9) and the binding guards (GROUND-R10), a SECOND
verifier — one that shares no weights with the drafting model — checks that each published
sentence is ENTAILED by the code-rendered canonical fact sheet. This operationalizes the
AIS criterion (Rashkin et al., arXiv:2112.12870 — a citation is valid iff the cited source
entails the sentence) with a MiniCheck-class grounded fact-checking model (Tang, Laban &
Durrett, EMNLP 2024, arXiv:2404.10774). It is what catches the binding errors rules cannot
enumerate — paraphrase, direction/trend words ("climbed", "stable"), multi-fact composition
— including the ``COMPLEMENTARY`` numberless-trend free pass.

Fail-closed semantics, in both directions that matter:

* a NON-ENTAILED sentence is removed from the published text and the decision is forced
  off ``proceed`` (re-draft, or abstain when nothing publishable remains) — the verifier
  can only VETO; it can never make an unverified value sayable;
* an UNAVAILABLE verifier (not installed / load failure) degrades to the deterministic
  layers ONLY — the result is returned unchanged and the caller records the degradation
  on the observability surface (never silently fail-open, never a crash of the run).

The per-claim verdicts of the deterministic gate are NOT rewritten here: a veto edits the
PUBLISHED text and the aggregate decision (the artifacts that ship); on any veto the run
re-drafts, so this round's citations are discarded with the draft. Thresholds are loaded
config (CFG-R1a), calibratable via :mod:`wattwise_core.agent.grounding_conformal`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from wattwise_core.agent.contracts import (
    ClaimKind,
    GroundDecision,
    GroundingResult,
    GroundVerdict,
)
from wattwise_core.agent.grounding_conformal import (
    CalibrationProvenance,
    conformal_thresholds,
    load_calibration,
    prompt_sha256,
)
from wattwise_core.agent.grounding_sweep import NUMBER_RE

# The verifier ADAPTER import is light (the heavy ML stack loads lazily inside the
# adapter on first use), so the seam wiring below can reference it unconditionally.
from wattwise_core.agent.verifier_minicheck import MiniCheckVerifier

#: Bound on per-deliverable verifier calls (the VOICE-R7 number cap keeps real deliverables
#: far below it; the bound is a cost/latency guard, not a correctness gate — sentences past
#: it stay governed by the deterministic layers).
_DEFAULT_MAX_CHECKS = 16

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@runtime_checkable
class EntailmentVerifier(Protocol):
    """A grounded fact-checking model behind a typed seam (proposed GROUND-R11).

    ``support`` returns the probability in ``[0, 1]`` that ``sentence`` is supported by
    (entailed by) ``facts``. Implementations MUST be independent of the drafting model
    (different weights, different objective) — decorrelation is the point: a verifier that
    shares the generator's blind spots re-certifies its hallucinations. The production
    adapter is :class:`~wattwise_core.agent.verifier_minicheck.MiniCheckVerifier`.
    """

    async def support(self, *, sentence: str, facts: str) -> float: ...


@dataclass(frozen=True, slots=True)
class EntailmentThresholds:
    """Per-claim-class publication thresholds (CFG-R1a loaded; conformally calibratable).

    ``number`` gates digit-bearing sentences; ``statement`` gates numberless trend/state
    sentences (the former ``COMPLEMENTARY`` free pass). Group-conditional thresholds follow
    the conformal claim-filtering literature (issue #10 Phase 3): one global threshold
    under-covers the hardest class.
    """

    number: float
    statement: float

    def for_sentence(self, sentence: str) -> float:
        """The threshold for one sentence, by its checkable-content class."""
        return self.number if NUMBER_RE.search(sentence) else self.statement


@dataclass(frozen=True, slots=True)
class EntailmentReport:
    """What the gate did, for the caller's observability recording (AGT-OBS-R4).

    ``unavailable`` is True when the verifier could not run at all (the run degraded to
    the deterministic layers; the caller records it — proposal: never silently open).
    """

    checked: int = 0
    vetoed: tuple[str, ...] = ()
    unavailable: bool = False


class EntailmentGate:
    """Sentence-level entailment gating over a :class:`GroundingResult` (GROUND-R11)."""

    def __init__(
        self,
        verifier: EntailmentVerifier,
        thresholds: EntailmentThresholds,
        *,
        max_checks: int = _DEFAULT_MAX_CHECKS,
    ) -> None:
        self._verifier = verifier
        self._thresholds = thresholds
        self._max_checks = max(1, max_checks)

    async def apply(
        self, result: GroundingResult, *, facts: str
    ) -> tuple[GroundingResult, EntailmentReport]:
        """Veto every published sentence the fact sheet does not entail (fail-closed).

        Checks the CHECKABLE sentences of the already-scrubbed text: every digit-bearing
        sentence (each surviving figure is canonical after the value gate — the question
        left is whether the SENTENCE means what the record says) and every published
        complementary statement (numberless trend/state claims). A sentence scoring below
        its class threshold is removed and the decision is forced off ``proceed``. A
        verifier failure returns the result UNCHANGED with ``unavailable=True`` — the
        deterministic layers remain the floor and the caller records the degradation.
        """
        targets = self._targets(result)
        if not targets:
            return result, EntailmentReport()
        vetoed: list[str] = []
        try:
            for sentence in targets:
                probability = await self._verifier.support(sentence=sentence, facts=facts)
                if probability < self._thresholds.for_sentence(sentence):
                    vetoed.append(sentence)
        except Exception:
            # ANY verifier fault (missing optional dependency, load/runtime failure) degrades
            # the run to the deterministic layers — recorded by the caller, never fail-open
            # and never a crash of the athlete's turn (issue #10: verifier absence is a
            # caveated degradation, not an outage).
            return result, EntailmentReport(unavailable=True)
        if not vetoed:
            return result, EntailmentReport(checked=len(targets))
        text = _remove_sentences(result.scrubbed_text, vetoed)
        decision = _downgrade(result.decision, text)
        gated = GroundingResult(decision=decision, claims=result.claims, scrubbed_text=text)
        return gated, EntailmentReport(checked=len(targets), vetoed=tuple(vetoed))

    def _targets(self, result: GroundingResult) -> tuple[str, ...]:
        """The checkable sentences of the published text, bounded by ``max_checks``.

        Digit-bearing sentences are always checkable; numberless sentences are checked when
        the deterministic gate published a complementary STATEMENT claim (the trend/state
        free pass this layer closes). The bound truncates deterministically — unchecked
        overflow stays under the deterministic layers rather than being vetoed blind.
        """
        has_statement = any(
            gc.verdict is GroundVerdict.COMPLEMENTARY and gc.claim.kind is ClaimKind.STATEMENT
            for gc in result.claims
        )
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(result.scrubbed_text)]
        targets = [
            sentence
            for sentence in sentences
            if sentence and (NUMBER_RE.search(sentence) or has_statement)
        ]
        return tuple(targets[: self._max_checks])


def _remove_sentences(text: str, vetoed: list[str]) -> str:
    """Remove each vetoed sentence from ``text`` (first occurrence; whitespace tidied)."""
    edited = text
    for sentence in vetoed:
        idx = edited.find(sentence)
        if idx == -1:
            continue
        edited = edited[:idx] + edited[idx + len(sentence) :]
    edited = re.sub(r"\s{2,}", " ", edited)
    return re.sub(r"\s+([.,;:!?])", r"\1", edited).strip()


def _downgrade(decision: GroundDecision, remaining_text: str) -> GroundDecision:
    """The post-veto aggregate decision (never ``proceed``; fail-closed).

    A veto means a verified-value sentence failed the MEANING check: re-draft when prose
    survives, abstain when nothing does (GROUND-R6). An already-recovering decision is
    kept — the veto only ever strengthens the gate.
    """
    if not remaining_text.strip():
        return GroundDecision.ABSTAIN
    if decision is GroundDecision.PROCEED:
        return GroundDecision.REGENERATE
    return decision


def gate_from_settings(settings: Any) -> EntailmentGate | None:
    """Build the configured GROUND-R11 gate from ``[agent.entailment]`` (CFG-R1a).

    ``None`` when the gate is disabled (the OSS default: the deterministic layers carry
    the guarantee; the verifier is an operator opt-in with its own model download). With
    a ``calibration_path`` the per-class thresholds come from the split-conformal artifact
    (proposed GROUND-R12) — a missing/malformed artifact FAILS THE BOOT closed
    (:class:`~wattwise_core.agent.grounding_conformal.CalibrationError`), never a silent
    fallback to uncalibrated thresholds, and the artifact's PROVENANCE stamp must match
    the configured verifier checkpoint + claim-extraction prompt (+ the optional dataset
    pin) — a stale artifact fails the boot the same way (the QA-EVAL-R12 cassette-pin
    rule: a calibration recorded under another model/prompt does not transfer). The
    verifier import stays lazy: building the gate is cheap; the checkpoint loads on first
    use and an unloadable verifier degrades each run to the deterministic layers,
    recorded (never fail-open).
    """
    if not settings.agent__entailment__enabled:
        return None
    number = settings.agent__entailment__threshold_number
    statement = settings.agent__entailment__threshold_statement
    calibration_path = settings.agent__entailment__calibration_path
    if calibration_path:
        records = load_calibration(
            Path(calibration_path),
            expected=CalibrationProvenance(
                model_id=settings.agent__entailment__model_id,
                claim_prompt_sha256=prompt_sha256(settings.agent__coach__prompts["claim_system"]),
                dataset_version=settings.agent__entailment__calibration_dataset_version,
            ),
        )
        calibrated = conformal_thresholds(
            records, settings.agent__entailment__alpha, groups=("number", "statement")
        )
        number, statement = calibrated["number"], calibrated["statement"]
    verifier = MiniCheckVerifier(
        settings.agent__entailment__model_id, device=settings.agent__entailment__device
    )
    return EntailmentGate(
        verifier,
        EntailmentThresholds(number=number, statement=statement),
        max_checks=settings.agent__entailment__max_checks,
    )


__all__ = [
    "EntailmentGate",
    "EntailmentReport",
    "EntailmentThresholds",
    "EntailmentVerifier",
    "gate_from_settings",
]
