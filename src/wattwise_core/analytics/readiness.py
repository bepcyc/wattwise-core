"""Deterministic readiness/form assessment oracle (QA-EVAL-R2.4).

"Readiness" is a typed VERDICT/state, not a number (SCHEMA-R3 / COACH-R1 #2):
:class:`~wattwise_core.domain.enums.ReadinessVerdict` ``go | maintain | ease | rest``.
There is deliberately NO numeric ``readiness`` metric. "Form" IS a number — the
canonical TSB (``CTL(d-1) - ATL(d-1)``, doc 40 PMC-R1) — computed elsewhere and
passed in here as the ``form`` input; this module never computes CTL/ATL/TSB.

This module is the DETERMINISTIC oracle behind the binding QA-EVAL-R2.4 invariant:
the verdict direction MUST be consistent with the metrics (deep-negative form ⇒ NOT
a hard "go" day), and that consistency is decided by code, not the LLM (COACH-R3 /
EVAL-R5). :func:`readiness_consistent` is the runtime consistency gate and the eval
grader's certificate.

Missing-input handling is honest, never guessed (COACH-R1 #2 / GROUND-R7): when HRV
is unavailable it is recorded in ``inputs_unavailable`` and the verdict is taken from
form alone; when ``form`` itself is unavailable (``None`` / not-seeded / non-finite),
readiness cannot be assessed and the verdict is ``None`` so the caller abstains
truthfully (GROUND-R6).

Pure module (ANL-R2/R30): no DB, no I/O, no wall-clock, no RNG; no imports from
``agent/`` or ``persistence/``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from wattwise_core.analytics.constants import (
    READINESS_FATIGUE_FLOOR,
    READINESS_FRESH_FORM,
    READINESS_HRV_SUPPRESSION_FRAC,
    READINESS_NEUTRAL_FLOOR,
)
from wattwise_core.domain.enums import ReadinessVerdict

__all__ = [
    "ReadinessAssessment",
    "assess_readiness",
    "readiness_consistent",
]

# Aggressiveness ladder GO (most aggressive) -> REST (least), used for the one-step
# fail-safe HRV nudge and for monotonicity reasoning. Index 0 == GO, last == REST.
# Explicit ordered tuple + index clamp — NEVER arithmetic on enum values.
_AGGRESSIVENESS_ORDER: tuple[ReadinessVerdict, ...] = (
    ReadinessVerdict.GO,
    ReadinessVerdict.MAINTAIN,
    ReadinessVerdict.EASE,
    ReadinessVerdict.REST,
)


@dataclass(frozen=True, slots=True)
class ReadinessAssessment:
    """One deterministic readiness assessment (the oracle output, QA-EVAL-R2.4).

    ``verdict`` is ``None`` iff there was insufficient data to assess (``form``
    unavailable). ``form`` is the canonical TSB used as the citable backing number
    (never a "readiness score"). ``rationale`` is a short deterministic snake_case
    TAG (NOT prose): one of ``fresh``, ``neutral``, ``productive_fatigue``,
    ``deep_fatigue``, ``hrv_suppressed``, ``form_unavailable``.
    """

    verdict: ReadinessVerdict | None
    form: float | None
    hrv_rmssd: float | None
    inputs_used: tuple[str, ...]
    inputs_unavailable: tuple[str, ...]
    rationale: str


def _is_finite(x: float | None) -> bool:
    """True iff ``x`` is a present, finite real (fail-closed on None/NaN/Inf)."""
    return x is not None and math.isfinite(x)


def _base_verdict(form: float) -> tuple[ReadinessVerdict, str]:
    """Map a finite ``form`` (TSB) to its band verdict + rationale tag (QA-EVAL-R2.4).

    Bands (defaults; configurable in spirit — see constants):
        form > READINESS_FRESH_FORM            => GO   ("fresh")
        NEUTRAL_FLOOR <= form <= FRESH_FORM     => MAINTAIN ("neutral")
        FATIGUE_FLOOR <= form < NEUTRAL_FLOOR   => EASE ("productive_fatigue")
        form < FATIGUE_FLOOR                    => REST ("deep_fatigue")
    """
    if form > READINESS_FRESH_FORM:
        return ReadinessVerdict.GO, "fresh"
    if form >= READINESS_NEUTRAL_FLOOR:
        return ReadinessVerdict.MAINTAIN, "neutral"
    if form >= READINESS_FATIGUE_FLOOR:
        return ReadinessVerdict.EASE, "productive_fatigue"
    return ReadinessVerdict.REST, "deep_fatigue"


def assess_readiness(
    *,
    form: float | None,
    hrv_rmssd: float | None = None,
    hrv_baseline: float | None = None,
) -> ReadinessAssessment:
    """Deterministically assess readiness from canonical inputs (QA-EVAL-R2.4).

    Parameters
    ----------
    form:
        The canonical TSB (``CTL(d-1) - ATL(d-1)``, PMC-R1), computed elsewhere.
        ``None`` (or non-finite) means form is unavailable -> the verdict is
        ``None`` and the caller abstains truthfully (GROUND-R6/R7).
    hrv_rmssd, hrv_baseline:
        The measured HRV (RMSSD) and the athlete's HRV baseline. The HRV nudge fires
        only when BOTH are present, finite, and ``hrv_baseline > 0``; otherwise HRV
        is recorded in ``inputs_unavailable`` and the verdict comes from form alone.
        HRV can ONLY nudge toward MORE caution (never toward GO) — fail-safe.
    """
    hrv_usable = _is_finite_hrv(hrv_rmssd, hrv_baseline)

    # 1) Form unavailable (None / NaN / Inf) -> cannot assess; fail-closed (GROUND-R6/R7).
    if form is None or not math.isfinite(form):
        hrv_unavail: tuple[str, ...] = () if hrv_usable else ("hrv",)
        return ReadinessAssessment(
            verdict=None,
            form=None,
            hrv_rmssd=hrv_rmssd,
            inputs_used=(),
            inputs_unavailable=("form", *hrv_unavail),
            rationale="form_unavailable",
        )

    # 2) Base verdict from the form bands.
    verdict, rationale = _base_verdict(form)

    # 3) HRV nudge — only ever toward MORE caution (fail-safe, COACH-R1 #2 / GROUND-R7).
    inputs_used: tuple[str, ...]
    inputs_unavailable: tuple[str, ...]
    if hrv_usable and hrv_rmssd is not None and hrv_baseline is not None:
        inputs_used = ("form", "hrv")
        inputs_unavailable = ()
        if hrv_rmssd < hrv_baseline * (1.0 - READINESS_HRV_SUPPRESSION_FRAC):
            verdict = _nudge_one_step_more_cautious(verdict)
            rationale = "hrv_suppressed"
    else:
        inputs_used = ("form",)
        inputs_unavailable = ("hrv",)

    return ReadinessAssessment(
        verdict=verdict,
        form=form,
        hrv_rmssd=hrv_rmssd,
        inputs_used=inputs_used,
        inputs_unavailable=inputs_unavailable,
        rationale=rationale,
    )


def _is_finite_hrv(hrv_rmssd: float | None, hrv_baseline: float | None) -> bool:
    """True iff HRV inputs are usable: both present, finite, and baseline > 0."""
    return (
        _is_finite(hrv_rmssd)
        and _is_finite(hrv_baseline)
        and hrv_baseline is not None
        and hrv_baseline > 0.0
    )


def _nudge_one_step_more_cautious(verdict: ReadinessVerdict) -> ReadinessVerdict:
    """Move one step toward REST along GO->MAINTAIN->EASE->REST, clamped at REST.

    Implemented via an explicit ordered tuple + index clamp (no enum arithmetic).
    """
    idx = _AGGRESSIVENESS_ORDER.index(verdict)
    return _AGGRESSIVENESS_ORDER[min(idx + 1, len(_AGGRESSIVENESS_ORDER) - 1)]


def readiness_consistent(
    verdict: ReadinessVerdict,
    *,
    form: float | None,
    hrv_rmssd: float | None = None,
    hrv_baseline: float | None = None,
) -> bool:
    """True iff ``verdict`` equals the deterministic band verdict (EVAL-R5 / COACH-R3).

    Used by the runtime consistency gate and the eval grader. When the deterministic
    verdict is ``None`` (form unavailable) this returns ``False``: no verdict can be
    certified against a missing form (GROUND-R6).
    """
    assessment = assess_readiness(form=form, hrv_rmssd=hrv_rmssd, hrv_baseline=hrv_baseline)
    return assessment.verdict is not None and verdict == assessment.verdict
