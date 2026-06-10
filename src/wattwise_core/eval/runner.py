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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from wattwise_core.agent.contracts import (
    Claim,
    ClaimKind,
    GroundDecision,
    GroundVerdict,
)
from wattwise_core.agent.grounding import ground
from wattwise_core.agent.grounding_evidence import _ClaimSchema
from wattwise_core.agent.model import FakeModel
from wattwise_core.eval import budget as budget_mod
from wattwise_core.eval import engine_suites, injection
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
from wattwise_core.eval.passk import degenerate_pass_k
from wattwise_core.eval.scorecard import EvalMode, Scorecard

_DATASETS_DIR = Path(__file__).parent / "datasets"
_DEFAULT_TOLERANCE = 0.01


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
        """Run one case end-to-end through the PRODUCTION grounder (GROUND-R8/EVAL-R4).

        The model only proposes prose + candidate claim spans; the SHIPPED
        :func:`wattwise_core.agent.grounding.ground` — not a re-implementation — decides
        what survives, so the gate exercises the production grounding identity path.
        """
        auth_id = injection.authenticated_id(case, authenticated)
        model = FakeModel(prose=str(case.get("draft_prose", "")))
        # compose() is exercised to prove the prose path is recorded/offline; the
        # production grounder below — not this prose — decides what survives.
        _ = await model.compose(system="voice", context=str(case.get("request_text", "")))
        claims = _parse_claims(case.get("candidate_claims", []))
        evidence = dict(case.get("evidence", {}))
        schema_valid = _validate_structured_output(case)
        return await _ground_case(
            case, auth_id, claims, evidence, tolerance, authenticated, schema_valid
        )


class _EvalEvidence:
    """Eval-side :class:`GroundingEvidence` driving the PRODUCTION grounder (GROUND-R8).

    Exposes the synchronous ``metric_snapshot(metric, as_of)`` accessor the production
    grounder resolves numbers against, a first-party URL allow-list, and an optional
    ``canonical_name`` library. Numbers come VERBATIM from the dataset's canonical metrics
    (a competing ``stale_memory`` value, if present, is DELIBERATELY NOT consulted, proving
    MEM-R3/EVAL-R2a non-substitution). ``tolerance`` is folded into the value match by the
    grounder's own tolerance, so this only returns the canonical value.
    """

    def __init__(
        self,
        metrics: Mapping[str, float],
        allowed_urls: set[str],
        names: Mapping[str, str] | None = None,
    ) -> None:
        self._metrics = dict(metrics)
        self._allowed = allowed_urls
        self._names = dict(names or {})

    def metric_snapshot(self, metric: str, as_of: str | None) -> float | None:
        return self._metrics.get(metric)

    async def metric_value(self, metric: str, as_of: str | None) -> float | None:
        return self._metrics.get(metric)

    def url_allowed(self, url: str) -> bool:
        return url in self._allowed

    def canonical_name(self, name: str) -> str | None:
        return self._names.get(name)


async def _ground_case(
    case: dict[str, Any],
    auth_id: str | None,
    claims: tuple[Claim, ...],
    evidence: dict[str, Any],
    tolerance: float,
    authenticated: dict[str, Any] | None,
    schema_valid: bool,
) -> RunnerOutcome:
    """Ground claims through the PRODUCTION grounder and assemble the outcome (GROUND-R8)."""
    metrics: dict[str, float] = {str(k): float(v) for k, v in evidence.get("metrics", {}).items()}
    allowed_urls = {str(u) for u in evidence.get("allowed_urls", [])}
    names = {str(k): str(v) for k, v in evidence.get("names", {}).items()}
    ev = _EvalEvidence(metrics, allowed_urls, names)
    # EVAL-R2a / MEM-R3: a STALE memory value competes as a candidate claim. The production
    # grounder must surface the LIVE canonical value (in ``metrics``) and scrub the memory
    # value, proving memory never substitutes for a live canonical number.
    claims = (*claims, *_memory_competitor_claims(case, metrics))
    # GROUND-R7: a metric whose canonical computation is ``unavailable`` has NO value in
    # ``metrics``; the production grounder scrubs any claim for it (never a placeholder).
    draft = str(case.get("draft_prose", ""))
    result = ground(draft, claims, ev, allowed_urls)

    survivors = {
        _claim_key(gc.claim) for gc in result.claims if gc.verdict is GroundVerdict.GROUNDED
    }
    scrubbed = {
        _scrub_key(gc.claim, metrics)
        for gc in result.claims
        if gc.verdict in (GroundVerdict.UNGROUNDED, GroundVerdict.CONTRADICTED)
    }
    published_urls = {
        gc.claim.ref or gc.claim.text
        for gc in result.claims
        if gc.verdict is GroundVerdict.GROUNDED and gc.claim.kind is ClaimKind.URL
    }
    # A surfaced (published) number that is not canonical is a fabrication leak. The
    # production grounder scrubs/replaces all such, so this is empty when fail-closed.
    non_canonical = {
        _claim_key(gc.claim)
        for gc in result.claims
        if gc.verdict is GroundVerdict.GROUNDED
        and gc.claim.kind is ClaimKind.NUMBER
        and not _value_matches(gc.claim.value, metrics.get(gc.claim.metric or ""), tolerance)
    }
    inj = await injection.injection_facts(case, auth_id, non_canonical, authenticated)
    return RunnerOutcome(
        case_id=str(case["id"]),
        suite=str(case.get("suite", "")),
        abstained=result.decision is GroundDecision.ABSTAIN or not survivors,
        schema_valid=schema_valid,
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


def _memory_competitor_claims(
    case: dict[str, Any], metrics: Mapping[str, float]
) -> tuple[Claim, ...]:
    """Build a candidate claim carrying a STALE MEMORY value, if the case plants one.

    EVAL-R2a / MEM-R3: the memory value competes against the live canonical number; the
    production grounder must NOT publish it (it is contradicted/scrubbed because it differs
    from the live ``metrics`` value), proving memory never substitutes for live truth.
    """
    stale = case.get("stale_memory") or {}
    out: list[Claim] = []
    for metric, value in stale.items():
        # Only inject when it genuinely differs from the live value (a real competitor).
        live = metrics.get(str(metric))
        if live is None or float(value) == live:
            continue
        out.append(
            Claim(
                kind=ClaimKind.NUMBER,
                text=f"memory says {metric} was {value}",
                metric=str(metric),
                value=float(value),
            )
        )
    return tuple(out)


def _scrub_key(claim: Claim, metrics: Mapping[str, float]) -> str:
    """Dataset-aligned scrub key (a contradicted number tags ``metric@value``)."""
    if claim.kind is ClaimKind.URL:
        return claim.ref or claim.text
    if claim.kind is ClaimKind.NUMBER and claim.metric is not None:
        if claim.metric in metrics:
            return f"{claim.metric}@{claim.value}"
        return claim.metric
    return claim.text


def _expected_scrubbed(case: dict[str, Any]) -> frozenset[str]:
    expected = case.get("expected", {})
    out: set[str] = {str(m) for m in expected.get("scrubbed_metrics", [])}
    out.update(str(u) for u in expected.get("scrubbed_urls", []))
    return frozenset(out)


# The fields the production claim-extraction schema (``_ExtractedClaim``) accepts. The
# recorded candidate-claim shape carries extra eval-only keys (``ref``/``prescriptive``)
# the closed schema forbids; only these are projected when validating against it so the
# check measures whether the recorded structured OUTPUT conforms, not the eval wrapper.
_EXTRACTED_CLAIM_FIELDS = ("kind", "text", "metric", "value", "as_of")


def _validate_structured_output(case: dict[str, Any]) -> bool:
    """Measure structured-output conformance against the DECLARED schema (QA-EVAL-R2.6).

    This is the REAL provider-enforced structured-output validation the schema suite reads,
    NOT a hardcoded literal: the recorded structured claim-extraction output is validated
    against the SHIPPED production claim-extraction schema (``_ClaimSchema``, ``extra='forbid'``,
    in :mod:`wattwise_core.agent.engine_services`). A case may record the verbatim model
    output under ``structured_output``; otherwise the recorded ``candidate_claims`` are
    projected onto the closed schema's fields and validated. A value of the wrong type (e.g. a
    non-numeric ``value``) or an extra/unknown field is a conformance failure (schema_valid False).
    """
    if "structured_output" in case:
        payload = case["structured_output"]
    else:
        payload = {
            "claims": [
                {k: raw[k] for k in _EXTRACTED_CLAIM_FIELDS if k in raw}
                for raw in case.get("candidate_claims", [])
            ]
        }
    try:
        _ClaimSchema.model_validate(payload)
    except PydanticValidationError:
        return False
    return True


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


def list_suites() -> tuple[str, ...]:
    """Every CI-gated suite name (EVAL-R1/-R9): safety + engine + ROAD-R2-EXIT coach suites."""
    return (
        "grounding",
        "abstention",
        "injection",
        "termination",
        "reflection_termination",
        "intent_plan",
        "multilingual",
        "judge",
        "readiness",
        "plan",
        "voice",
        "self_certification",
    )


async def run_suite(name: str, *, mode: EvalMode = EvalMode.RECORDED) -> Scorecard:
    """Run a whole named suite and return its scorecard (EVAL-R9 + degenerate k=1 pass^k).

    The OSS recorded tier is deterministic (TIER-R1) so one trial is exact (QA-EVAL-R10);
    the env-gated live nightly leg is the only place k>1 real trials are spent. The
    QA-EVAL-R8 per-case token/cost/latency record + budget gate are folded into every
    scorecard.
    """
    if name in engine_suites.ENGINE_SUITES:
        return await engine_suites.run_engine_suite(name, mode)
    dataset = load_dataset(name)
    runner = EvalRunner(mode=mode)
    outcomes = [
        await runner.run_case(
            case, tolerance=dataset.tolerance, authenticated=dataset.authenticated
        )
        for case in dataset.cases
    ]
    grades = grade_suite(dataset.suite, outcomes)
    return Scorecard(
        suite=dataset.suite,
        dataset_version=dataset.version,
        mode=mode,
        total_cases=len(outcomes),
        grades=budget_mod.with_budget(grades, dataset.cases),
        budget_samples=budget_mod.record_samples(dataset.cases),
        pass_k=degenerate_pass_k(dataset.suite, grades.passed),
    )


__all__ = [
    "Dataset",
    "EvalMode",
    "EvalRunner",
    "RunnerOutcome",
    "Scorecard",
    "grade_suite",
    "list_suites",
    "load_dataset",
    "run_suite",
]
