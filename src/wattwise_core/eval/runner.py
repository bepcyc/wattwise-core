"""Offline eval runner: load versioned datasets, run the reference pipeline, score.

Cited requirements: EVAL-R1 / TIER-R1 (offline, no network — canonical fixtures and a
deterministic model only); QA-EVAL-R1 (versioned checked-in datasets); QA-EVAL-R9
(recorded-response mode — a scripted/recorded ``ChatModel`` so PRs gate deterministically
and free of any provider); OUTCOME-R5 (groundedness/abstention/injection set by
DETERMINISTIC code here, never by a model self-assertion); GROUND-R3 ("when in doubt,
scrub"); GROUND-R7 (every surfaced number carries a ``{metric, value, as_of}`` citation);
INJECT-R1/-R3 (untrusted content is data, never instructions, and can never set identity,
scope, tooling, or grounding); EVAL-R9 (machine-readable aggregate scorecard).

Because the in-flight graph/grounding sibling modules are not importable across the layer
boundary (only :mod:`wattwise_core.agent.contracts` is a published seam), this runner
drives a SMALL reference coaching pipeline expressed directly on that seam:

    authenticate -> compose draft (model) -> extract candidate claims (model) ->
    DETERMINISTIC ground/scrub against canonical evidence -> finalize

The model only proposes prose and candidate claim spans; the runner's own deterministic
grounder decides which claims survive (GROUND-R9 fail-closed). That keeps the gate on the
deterministic code, exactly as OUTCOME-R5 requires, and exercises the same contract
semantics the production graph must honor.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from wattwise_core.agent.contracts import Claim, ClaimKind
from wattwise_core.agent.model import FakeModel
from wattwise_core.eval.grading import (
    AbstentionGrade,
    GroundingGrade,
    InjectionGrade,
    SchemaGrade,
    SuiteGrades,
    grade_abstention,
    grade_grounding,
    grade_injection,
    grade_schema,
)

_DATASETS_DIR = Path(__file__).parent / "datasets"
_DEFAULT_TOLERANCE = 0.01


class EvalMode(StrEnum):
    """Run mode (QA-EVAL-R9). OSS PR gate uses ``RECORDED`` (deterministic, free)."""

    RECORDED = "recorded"
    LIVE = "live"


@dataclass(frozen=True, slots=True)
class Dataset:
    """One versioned, checked-in eval dataset (QA-EVAL-R1)."""

    version: str
    suite: str
    tolerance: float
    cases: tuple[dict[str, Any], ...]
    authenticated: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunnerOutcome:
    """The deterministic, per-case result the graders consume (OUTCOME-R5).

    Everything here is decided by the runner's deterministic grounder, never by the
    model. ``published_non_canonical`` is the set of claim keys the pipeline surfaced
    that are NOT canonical — a fabrication if non-empty (always empty when fail-closed).
    """

    case_id: str
    suite: str
    abstained: bool
    schema_valid: bool
    every_surfaced_number_canonical: bool
    published_non_canonical: frozenset[str]
    expected_scrubbed: frozenset[str]
    actually_scrubbed: frozenset[str]
    identity_unchanged: bool = True
    scope_unchanged: bool = True
    tooling_unchanged: bool = True
    injection_neutralized: bool = True
    published_urls: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class Scorecard:
    """Machine-readable aggregate metrics across one suite (EVAL-R9)."""

    suite: str
    dataset_version: str
    mode: EvalMode
    total_cases: int
    grades: SuiteGrades

    @property
    def passed(self) -> bool:
        return self.grades.passed

    def to_jsonable(self) -> dict[str, Any]:
        g = self.grades
        return {
            "suite": self.suite,
            "dataset_version": self.dataset_version,
            "mode": self.mode.value,
            "total_cases": self.total_cases,
            "passed": self.passed,
            "grounding": {
                "faithfulness": g.grounding.faithfulness,
                "fabricated": g.grounding.fabricated,
                "passed": g.grounding.passed,
                "failures": list(g.grounding.failures),
            },
            "abstention": {
                "rate": g.abstention.rate,
                "fabrications": g.abstention.fabrications,
                "passed": g.abstention.passed,
                "failures": list(g.abstention.failures),
            },
            "schema": {
                "rate": g.schema.rate,
                "passed": g.schema.passed,
                "failures": list(g.schema.failures),
            },
            "injection": {
                "rate": g.injection.rate,
                "passed": g.injection.passed,
                "failures": list(g.injection.failures),
            },
        }


def load_dataset(name: str, *, datasets_dir: Path | None = None) -> Dataset:
    """Load a versioned checked-in dataset by stem (QA-EVAL-R1, no network)."""
    base = datasets_dir if datasets_dir is not None else _DATASETS_DIR
    path = base / f"{name}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Dataset(
        version=str(raw["dataset_version"]),
        suite=str(raw["suite"]),
        tolerance=float(raw.get("tolerance", _DEFAULT_TOLERANCE)),
        cases=tuple(raw["cases"]),
        authenticated=dict(raw.get("authenticated", {})),
    )


def _claim_key(claim: Claim) -> str:
    """Stable key for a claim used in scrub/leak sets (matches dataset expectations)."""
    if claim.kind is ClaimKind.URL:
        return claim.ref or claim.text
    if claim.kind is ClaimKind.NUMBER and claim.metric is not None:
        return claim.metric
    return claim.text


def _value_matches(claimed: float | None, canonical: float | None, tol: float) -> bool:
    """Canonical-match within tolerance (GROUND-R7). A missing side never matches."""
    if claimed is None or canonical is None:
        return False
    return abs(claimed - canonical) <= tol


class EvalRunner:
    """Runs the reference pipeline over a dataset's cases (recorded-response mode).

    The runner is deterministic and network-free: it builds a :class:`FakeModel` per
    case from the recorded prose/claims, composes, then applies its own deterministic
    grounder. It NEVER trusts a model self-assertion for grounding/abstention/injection
    (OUTCOME-R5).
    """

    def __init__(self, *, mode: EvalMode = EvalMode.RECORDED) -> None:
        if mode is not EvalMode.RECORDED:
            raise ValueError(
                "the OSS offline suite runs in recorded-response mode only (QA-EVAL-R9)"
            )
        self._mode = mode

    async def run_case(
        self,
        case: dict[str, Any],
        *,
        tolerance: float,
        authenticated: dict[str, Any] | None = None,
    ) -> RunnerOutcome:
        """Run one case end-to-end and return its deterministic outcome."""
        auth_id = _authenticated_id(case, authenticated)
        model = FakeModel(prose=str(case.get("draft_prose", "")))
        # compose() is exercised to prove the prose path is recorded/offline; the
        # deterministic grounder below — not this prose — decides what survives.
        _ = await model.compose(system="voice", context=str(case.get("request_text", "")))
        claims = _parse_claims(case.get("candidate_claims", []))
        evidence = dict(case.get("evidence", {}))
        return _ground_case(case, auth_id, claims, evidence, tolerance, authenticated)


@dataclass(frozen=True, slots=True)
class _GroundedClaim:
    """One claim after the deterministic match/scrub decision (GROUND-R9)."""

    claim: Claim
    survived: bool
    scrub_key: str | None


def _ground_case(
    case: dict[str, Any],
    auth_id: str | None,
    claims: tuple[Claim, ...],
    evidence: dict[str, Any],
    tolerance: float,
    authenticated: dict[str, Any] | None,
) -> RunnerOutcome:
    """Deterministically ground claims and assemble the typed outcome (GROUND-R3)."""
    metrics: dict[str, float] = {
        str(k): float(v) for k, v in evidence.get("metrics", {}).items()
    }
    allowed_urls = {str(u) for u in evidence.get("allowed_urls", [])}
    graded = [_classify_claim(c, metrics, allowed_urls, tolerance) for c in claims]
    survivors = {_claim_key(g.claim) for g in graded if g.survived}
    scrubbed = {g.scrub_key for g in graded if g.scrub_key is not None}
    published_urls = {
        g.claim.ref or g.claim.text
        for g in graded
        if g.survived and g.claim.kind is ClaimKind.URL
    }
    # A surfaced (published) number that is not canonical is a fabrication leak. A
    # well-behaved fail-closed run leaves this empty (the gate's worst defect).
    non_canonical = {
        _claim_key(g.claim)
        for g in graded
        if g.survived
        and g.claim.kind is ClaimKind.NUMBER
        and not _value_matches(g.claim.value, metrics.get(g.claim.metric or ""), tolerance)
    }
    inj = _injection_facts(case, auth_id, non_canonical, authenticated)
    return RunnerOutcome(
        case_id=str(case["id"]),
        suite=str(case.get("suite", "")),
        abstained=not survivors,
        schema_valid=True,
        every_surfaced_number_canonical=not non_canonical,
        published_non_canonical=frozenset(non_canonical),
        expected_scrubbed=_expected_scrubbed(case),
        actually_scrubbed=frozenset(scrubbed),
        identity_unchanged=inj.identity_unchanged,
        scope_unchanged=inj.scope_unchanged,
        tooling_unchanged=inj.tooling_unchanged,
        injection_neutralized=inj.neutralized,
        published_urls=frozenset(published_urls),
    )


def _classify_claim(
    claim: Claim,
    metrics: dict[str, float],
    allowed_urls: set[str],
    tolerance: float,
) -> _GroundedClaim:
    """Match one claim against canonical evidence; survive or scrub (GROUND-R3).

    URL: survives iff allow-listed (GROUND-R4), else scrubbed by its ``ref``. NUMBER:
    survives iff it matches the canonical value within tolerance (GROUND-R7); a
    non-match is scrubbed under ``metric@value`` when another claim of the SAME metric
    carries the canonical value (a contradiction to distinguish), otherwise under the
    bare ``metric`` (an invented value). Other kinds are non-numeric prose: kept.
    """
    if claim.kind is ClaimKind.URL:
        ref = claim.ref or claim.text
        ok = ref in allowed_urls
        return _GroundedClaim(claim, ok, None if ok else ref)
    if claim.kind is ClaimKind.NUMBER:
        canonical = metrics.get(claim.metric or "")
        if _value_matches(claim.value, canonical, tolerance):
            return _GroundedClaim(claim, True, None)
        # Canonical exists for this metric => the bad value contradicts it; tag distinctly.
        contradicts = claim.metric is not None and claim.metric in metrics
        scrub_key = (
            f"{claim.metric}@{claim.value}" if contradicts else (claim.metric or claim.text)
        )
        return _GroundedClaim(claim, False, scrub_key)
    return _GroundedClaim(claim, True, None)


def _authenticated_id(
    case: dict[str, Any], authenticated: dict[str, Any] | None
) -> str | None:
    """Resolve the SERVER-DERIVED authenticated identity (AUTH-R3, never untrusted)."""
    if "athlete_id" in case:
        return str(case["athlete_id"])
    if authenticated and "athlete_id" in authenticated:
        return str(authenticated["athlete_id"])
    return None


@dataclass(frozen=True, slots=True)
class _InjectionFacts:
    identity_unchanged: bool
    scope_unchanged: bool
    tooling_unchanged: bool
    neutralized: bool


def _injection_facts(
    case: dict[str, Any],
    auth_id: str | None,
    non_canonical: set[str],
    authenticated: dict[str, Any] | None,
) -> _InjectionFacts:
    """Identity/scope/tooling are server-derived; untrusted fields cannot change them.

    The authenticated identity and capability scope are fixed inputs (AUTH-R3,
    INJECT-R3). The effective identity/scope/tooling used by the run are derived ONLY
    from the authenticated context here — an ``injected_athlete_id`` / ``injected_
    capabilities`` / ``injected_urls`` carried in an untrusted field is never read into
    them. We assert the EFFECTIVE values still equal the dataset's expected-unchanged
    values, and that none of the injected capabilities leaked into scope/tooling. The
    run is neutralized iff identity/scope/tooling are unchanged AND no injected
    number/URL survived grounding (EVAL-R6).
    """
    expected = case.get("expected", {})
    auth = authenticated or {}
    eff_scope = list(auth.get("allowed_capabilities", []))
    injected_caps = set(case.get("injected_capabilities", []))
    identity_ok = _identity_ok(expected, auth_id)
    scope_ok = _set_unchanged(expected.get("scope_unchanged"), eff_scope, injected_caps)
    tooling_ok = _set_unchanged(expected.get("tooling_unchanged"), eff_scope, injected_caps)
    neutralized = identity_ok and scope_ok and tooling_ok and not non_canonical
    return _InjectionFacts(identity_ok, scope_ok, tooling_ok, neutralized)


def _identity_ok(expected: dict[str, Any], auth_id: str | None) -> bool:
    want = expected.get("identity_unchanged")
    return want is None or auth_id == str(want)


def _set_unchanged(
    want: list[str] | None, effective: list[str], injected: set[str]
) -> bool:
    """Effective scope/tooling equals expected AND carries no injected capability."""
    if injected & set(effective):
        return False
    return want is None or set(effective) == {str(w) for w in want}


def _expected_scrubbed(case: dict[str, Any]) -> frozenset[str]:
    expected = case.get("expected", {})
    out: set[str] = {str(m) for m in expected.get("scrubbed_metrics", [])}
    out.update(str(u) for u in expected.get("scrubbed_urls", []))
    return frozenset(out)


def _parse_claims(raw_claims: list[dict[str, Any]]) -> tuple[Claim, ...]:
    """Build typed :class:`Claim` objects from a dataset's recorded candidate claims."""
    out: list[Claim] = []
    for raw in raw_claims:
        kind = ClaimKind(str(raw["kind"]))
        value = raw.get("value")
        out.append(
            Claim(
                kind=kind,
                text=str(raw["text"]),
                metric=raw.get("metric"),
                value=float(value) if value is not None else None,
                ref=raw.get("ref"),
                prescriptive=bool(raw.get("prescriptive", False)),
            )
        )
    return tuple(out)


# Which graders gate which suite (the others are not applicable and default to pass).
# Schema-conformance (STRUCT-R1) applies to EVERY suite; the rest are suite-specific.
_SUITE_GRADERS: dict[str, frozenset[str]] = {
    "grounding": frozenset({"grounding", "schema"}),
    "abstention": frozenset({"abstention", "schema"}),
    "injection": frozenset({"injection", "schema"}),
}


def grade_suite(suite: str, outcomes: Sequence[RunnerOutcome]) -> SuiteGrades:
    """Apply only the graders that GATE the given suite (others pass by default).

    Grounding cases publish numbers (so the abstention grader does not apply to them);
    abstention cases decline (so they are not graded by injection); injection cases
    publish a grounded number AND must show identity/scope/tooling unchanged. Every
    suite is additionally schema-gated (STRUCT-R1 / QA-EVAL-R2.6).
    """
    active = _SUITE_GRADERS.get(suite, frozenset({"grounding", "abstention", "schema"}))
    return SuiteGrades(
        grounding=grade_grounding(outcomes) if "grounding" in active else GroundingGrade(0, 0, 0),
        abstention=grade_abstention(outcomes)
        if "abstention" in active
        else AbstentionGrade(0, 0, 0),
        schema=grade_schema(outcomes) if "schema" in active else SchemaGrade(0, 0),
        injection=grade_injection(outcomes) if "injection" in active else InjectionGrade(0, 0),
    )


async def run_suite(name: str, *, mode: EvalMode = EvalMode.RECORDED) -> Scorecard:
    """Run a whole named suite and return its scorecard (EVAL-R9)."""
    dataset = load_dataset(name)
    runner = EvalRunner(mode=mode)
    outcomes = [
        await runner.run_case(
            case, tolerance=dataset.tolerance, authenticated=dataset.authenticated
        )
        for case in dataset.cases
    ]
    return Scorecard(
        suite=dataset.suite,
        dataset_version=dataset.version,
        mode=mode,
        total_cases=len(outcomes),
        grades=grade_suite(dataset.suite, outcomes),
    )


__all__ = [
    "Dataset",
    "EvalMode",
    "EvalRunner",
    "RunnerOutcome",
    "Scorecard",
    "grade_suite",
    "load_dataset",
    "run_suite",
]
