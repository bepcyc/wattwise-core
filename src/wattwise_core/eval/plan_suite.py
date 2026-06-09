"""Multi-day-PLAN coach-quality eval suite (QA-EVAL-R2.5 / COACH-R2 / COACH-R3).

A deterministic, network-free grader for the ``plan`` dataset: each case is a concrete
multi-day plan (a per-day sequence of workout prescriptions) the engine would surface,
and the grader certifies three properties — the CODE deciding, never the LLM (EVAL-R5),
exactly as :func:`wattwise_core.eval.suites.grade_readiness` does for readiness:

* **GROUNDING (COACH-R2).** Every prescribed NUMBER (a power/HR target) and every
  prescribed workout NAME is run through the SHIPPED grounder
  (:func:`wattwise_core.agent.grounding.ground`, GROUND-R8 — not a re-implementation)
  against the case's canonical evidence (the athlete's CP/zones/threshold metrics and the
  canonical workout-name library). A planted ungrounded prescription (an invented number
  or a workout name that resolves to no library item) MUST be scrubbed, and ZERO
  non-canonical figures may survive (GROUND-R7 / EVAL-R4). A plan MUST NOT contain a
  workout that cannot be resolved to a canonical, schedulable entity (COACH-R2).
* **PROGRESSION (QA-EVAL-R2.5).** The plan's cumulative projected load is computed via the
  CANONICAL PMC EWMA (:func:`wattwise_core.analytics.pmc.pmc`) from the athlete's
  carried-forward pre-plan ``(CTL, ATL)`` seed and the per-day prescribed TSS; the
  resulting weekly CTL ramp MUST stay within the case's STATED progression bound. The
  bound caps how fast fitness may RISE, so a taper (a negative ramp) is always within
  bound; only an over-steep climb violates it.
* **CONSISTENCY (COACH-R3).** The plan's PEAK day-load intent MUST be consistent with the
  stated readiness verdict: a low-readiness verdict (``ease``/``rest``) MUST NOT co-occur
  with a high-load (peak-tier) prescription. Mirrors the readiness aggressiveness ladder.

The grade is a single dataclass with three 100%-gates and a ``failures`` tuple, mirroring
:class:`wattwise_core.eval.grading.ReadinessGrade`. Negative cases in the dataset
(``negative_cases``) drive the grader's teeth: each is asserted to FAIL its named
certificate, proving the gate is non-vacuous (a real defect cannot pass).

Cited requirements: QA-EVAL-R2.5, COACH-R2, COACH-R3, GROUND-R7, GROUND-R8, EVAL-R4,
EVAL-R5, OUTCOME-R5; EVAL-R1 / TIER-R1 (offline, deterministic, no network).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wattwise_core.agent.contracts import Claim, ClaimKind, GroundVerdict
from wattwise_core.agent.grounding import ground
from wattwise_core.analytics.pmc import PmcSeed, pmc
from wattwise_core.analytics.result import Computed
from wattwise_core.domain.enums import PlanDayIntent, ReadinessVerdict

_DATASETS_DIR = Path(__file__).parent / "datasets"
_DAYS_PER_WEEK = 7.0
# Numeric match tolerance for a prescribed power/HR figure vs the canonical metric. Looser
# than the grounder's own band on purpose: this only guards the "no non-canonical number
# survived" assertion; the SHIPPED grounder applies its own tighter tolerance internally.
_VALUE_TOL = 0.01

# Day-load tier ladder (0 lowest .. 3 highest peak), mirroring the GO->REST aggressiveness
# order in :mod:`wattwise_core.analytics.readiness`. A HIGH-load (peak) day is tier 3.
_INTENT_LOAD_TIER: dict[PlanDayIntent, int] = {
    PlanDayIntent.REST: 0,
    PlanDayIntent.RECOVERY: 0,
    PlanDayIntent.EASY: 1,
    PlanDayIntent.MODERATE: 2,
    PlanDayIntent.THRESHOLD: 2,
    PlanDayIntent.HARD: 3,
    PlanDayIntent.VO2: 3,
    PlanDayIntent.SPRINT: 3,
    PlanDayIntent.RACE: 3,
}
# The readiness verdict caps the plan's PEAK day-load tier (COACH-R3): a low-readiness
# verdict forbids a high-load (peak-tier) day, and ``rest`` additionally forbids tier-2.
_VERDICT_LOAD_CEILING: dict[ReadinessVerdict, int] = {
    ReadinessVerdict.GO: 3,
    ReadinessVerdict.MAINTAIN: 3,
    ReadinessVerdict.EASE: 2,
    ReadinessVerdict.REST: 1,
}


@dataclass(frozen=True, slots=True)
class PlanGrade:
    """Outcome of grading the multi-day-PLAN suite (QA-EVAL-R2.5 / COACH-R2 / COACH-R3).

    Three deterministic 100% gates over the positive cases: ``grounded`` (every
    prescription grounds / planted-ungrounded scrubbed, zero non-canonical survivors),
    ``within_bound`` (weekly CTL ramp within the stated bound), and ``consistent`` (peak
    day-load consistent with the readiness verdict). ``failures`` records every defect.
    """

    total: int
    grounded: int
    within_bound: int
    consistent: int
    failures: tuple[str, ...] = ()

    @property
    def grounding_rate(self) -> float:
        """Fraction of cases whose every prescription grounded fail-closed (1.0 if none)."""
        return 1.0 if self.total == 0 else self.grounded / self.total

    @property
    def progression_rate(self) -> float:
        """Fraction of cases whose weekly CTL ramp stays within the stated bound."""
        return 1.0 if self.total == 0 else self.within_bound / self.total

    @property
    def consistency_rate(self) -> float:
        """Fraction of cases whose peak load is consistent with the verdict (1.0 if none)."""
        return 1.0 if self.total == 0 else self.consistent / self.total

    @property
    def passed(self) -> bool:
        """Gate: all three rates are 100% AND zero recorded failures (QA-EVAL-R2.5).

        Like :class:`~wattwise_core.eval.grading.ReadinessGrade`, the rates alone can mask a
        defect, so the gate additionally requires ``failures == ()`` — any recorded failure
        of any certificate fails CI.
        """
        return (
            self.grounding_rate >= 1.0
            and self.progression_rate >= 1.0
            and self.consistency_rate >= 1.0
            and self.failures == ()
        )


def _load(name: str = "plan") -> dict[str, Any]:
    """Load a versioned checked-in dataset by stem (QA-EVAL-R1, no network)."""
    loaded: dict[str, Any] = json.loads(
        (_DATASETS_DIR / f"{name}.json").read_text(encoding="utf-8")
    )
    return loaded


class _PlanEvidence:
    """Eval-side ``GroundingEvidence`` + ``NameLibrary`` driving the SHIPPED grounder.

    Exposes the synchronous ``metric_snapshot`` accessor the production grounder resolves
    numbers against, a first-party URL allow-list, and the canonical workout-name library
    (so a NAME claim grounds only when it resolves to a real library item — COACH-R2).
    """

    def __init__(
        self,
        metrics: dict[str, float],
        allowed_urls: set[str],
        names: dict[str, str],
    ) -> None:
        self._metrics = metrics
        self._allowed = allowed_urls
        self._names = names

    def metric_snapshot(self, metric: str, as_of: str | None) -> float | None:
        return self._metrics.get(metric)

    async def metric_value(self, metric: str, as_of: str | None) -> float | None:
        return self._metrics.get(metric)

    def url_allowed(self, url: str) -> bool:
        return url in self._allowed

    def canonical_name(self, name: str) -> str | None:
        return self._names.get(name)


def _prescription_claims(case: dict[str, Any]) -> tuple[Claim, ...]:
    """Lift every per-day prescription in the plan into a typed :class:`Claim`."""
    claims: list[Claim] = []
    for day in case["days"]:
        for p in day.get("prescriptions", []):
            claims.append(
                Claim(
                    kind=ClaimKind(str(p["kind"])),
                    text=str(p["text"]),
                    metric=p.get("metric"),
                    value=float(p["value"]) if p.get("value") is not None else None,
                    ref=p.get("ref"),
                )
            )
    return tuple(claims)


def _scrub_key(claim: Claim, metrics: dict[str, float]) -> str:
    """Dataset-aligned scrub key (a contradicted number tags ``metric@value``)."""
    if claim.kind is ClaimKind.NUMBER and claim.metric is not None:
        return f"{claim.metric}@{claim.value}" if claim.metric in metrics else claim.metric
    return claim.text


def _grounding_failure(case: dict[str, Any]) -> str | None:
    """Drive the plan's prescriptions through the SHIPPED grounder (COACH-R2 / GROUND-R8).

    Returns a reason string when a planted-ungrounded prescription was NOT scrubbed or a
    non-canonical figure survived; ``None`` when the plan grounds fail-closed.
    """
    cid = case["id"]
    ev_raw = case["evidence"]
    metrics = {str(k): float(v) for k, v in ev_raw.get("metrics", {}).items()}
    allowed = {str(u) for u in ev_raw.get("allowed_urls", [])}
    names = {str(k): str(v) for k, v in ev_raw.get("names", {}).items()}
    evidence = _PlanEvidence(metrics, allowed, names)
    claims = _prescription_claims(case)
    result = ground("plan", claims, evidence, allowed)
    scrubbed = {
        _scrub_key(gc.claim, metrics)
        for gc in result.claims
        if gc.verdict in (GroundVerdict.UNGROUNDED, GroundVerdict.CONTRADICTED)
    }
    survived_non_canonical = [
        gc.claim.metric
        for gc in result.claims
        if gc.verdict is GroundVerdict.GROUNDED
        and gc.claim.kind is ClaimKind.NUMBER
        and abs((gc.claim.value or 0.0) - metrics.get(gc.claim.metric or "", 1e18)) > _VALUE_TOL
    ]
    if survived_non_canonical:
        return f"{cid}: non-canonical prescription survived grounding {survived_non_canonical}"
    expected = {str(s) for s in case["expected"].get("scrubbed_prescriptions", [])}
    missing = expected - scrubbed
    if missing:
        return f"{cid}: planted-ungrounded prescription not scrubbed {sorted(missing)}"
    return None


def _weekly_ctl_ramp(case: dict[str, Any]) -> float:
    """Compute the plan's weekly CTL ramp via the CANONICAL PMC EWMA (QA-EVAL-R2.5).

    Seeds the chart from the athlete's carried-forward ``(CTL, ATL)`` and runs the
    per-day prescribed TSS through :func:`wattwise_core.analytics.pmc.pmc`; the ramp is
    ``CTL(last) - CTL(seed)`` normalized to a 7-day week so plans of any length compare to
    the same per-week bound. The code (the canonical analytic), not the LLM, decides this.
    """
    seed_raw = case["seed"]
    seed = PmcSeed(ctl_prev=float(seed_raw["ctl_prev"]), atl_prev=float(seed_raw["atl_prev"]))
    loads = [float(d["tss"]) for d in case["days"]]
    results = pmc(loads, seed=seed)
    last = results[-1]
    if not isinstance(last, Computed):
        # An unavailable PMC tail cannot certify a ramp; treat as over-bound (fail-closed).
        return float("inf")
    end_ctl = last.value.ctl
    ramp_total = end_ctl - seed.ctl_prev
    return ramp_total * (_DAYS_PER_WEEK / len(loads))


def _progression_failure(case: dict[str, Any]) -> str | None:
    """Certify the plan's weekly CTL ramp against the stated bound (QA-EVAL-R2.5)."""
    cid = case["id"]
    bound = float(case["progression_bound"]["max_ctl_ramp_per_week"])
    ramp = _weekly_ctl_ramp(case)
    if ramp > bound:
        return f"{cid}: weekly CTL ramp {ramp:.2f} exceeds stated bound {bound:.2f}"
    return None


def _peak_load_tier(case: dict[str, Any]) -> int:
    """The plan's highest single-day load tier (0..3) over its day intents."""
    return max(_INTENT_LOAD_TIER[PlanDayIntent(str(d["intent"]))] for d in case["days"])


def _consistency_failure(case: dict[str, Any]) -> str | None:
    """Certify the plan's peak day-load against the readiness verdict (COACH-R3).

    A low-readiness verdict (``ease``/``rest``) prescribed alongside a peak-tier day is
    the forbidden co-occurrence; the deterministic ceiling decides, not the LLM (EVAL-R5).
    """
    cid = case["id"]
    verdict = ReadinessVerdict(str(case["readiness_verdict"]))
    peak = _peak_load_tier(case)
    ceiling = _VERDICT_LOAD_CEILING[verdict]
    if peak > ceiling:
        return (
            f"{cid}: readiness verdict {verdict.value!r} (load ceiling {ceiling}) co-occurs "
            f"with a higher-load day (peak tier {peak}) — COACH-R3 inconsistency"
        )
    return None


def grade_plan() -> PlanGrade:
    """Grade the multi-day-PLAN fixtures deterministically (QA-EVAL-R2.5 / COACH-R2/R3).

    For each POSITIVE case the grader runs all three certificates — grounding (through the
    SHIPPED grounder), progression (through the canonical PMC EWMA), and verdict<->load
    consistency — and records a failure for each that does not hold. Every certificate is a
    100% gate; the negative-case teeth are exercised by the suite's own tests, not here.
    """
    cases = _load()["cases"]
    failures: list[str] = []
    grounded = within_bound = consistent = 0
    for case in cases:
        g = _grounding_failure(case)
        if g is None:
            grounded += 1
        else:
            failures.append(g)
        p = _progression_failure(case)
        if p is None:
            within_bound += 1
        else:
            failures.append(p)
        c = _consistency_failure(case)
        if c is None:
            consistent += 1
        else:
            failures.append(c)
    return PlanGrade(len(cases), grounded, within_bound, consistent, tuple(failures))


__all__ = [
    "PlanGrade",
    "grade_plan",
]
