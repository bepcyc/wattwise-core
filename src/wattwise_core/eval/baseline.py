"""Non-regression baseline for the offline eval scorecard (QA-EVAL-R7).

QA-EVAL-R7 requires the CI gate to fail when a suite *regresses* against a stored,
versioned baseline — even when the suite's absolute score still clears its hard
threshold. A safety-suite (grounding / abstention / injection / schema) that slips from
1.0 to 0.97 is a regression the absolute gate (which is also 1.0 today, so would also
fire) AND the non-regression gate must catch; for a metric whose absolute floor is *below*
1.0 (intent-plan precision/recall, judge pass-count, termination/readiness rates), the
non-regression gate is the ONLY thing that catches a silent score erosion that still sits
above the floor. The two gates are complementary and BOTH must pass:

  * the absolute thresholds in :mod:`wattwise_core.eval.grading` (unchanged, still >= 1.0
    for grounding/abstention/schema/injection); and
  * this baseline non-regression check (current metric MUST be >= stored baseline).

The baseline is a checked-in artifact (``baseline-scorecard.json`` beside this module),
generated from a clean ``python -m wattwise_core.eval run``. ``eval-update-baseline``
rewrites it. It is intentionally NOT the raw scorecard: it stores only the per-suite
*comparable scalar metrics* (rates / precision / recall / counts), so adding a failure
string or reordering keys never spuriously trips a regression — only a genuine numeric
drop does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from wattwise_core.eval.scorecard import Scorecard

# The checked-in, versioned baseline artifact (QA-EVAL-R7). Lives beside this module so it
# is packaged with the eval engine and travels with the datasets it grades.
BASELINE_PATH = Path(__file__).parent / "baseline-scorecard.json"

# Schema version of the baseline artifact itself (independent of any dataset version), so
# a future change to the tracked-metric shape is detectable rather than silently mis-read.
BASELINE_FORMAT_VERSION = "1.0.0"

# A current metric is a regression only when it drops below baseline by MORE than this
# epsilon. Guards against float round-trip noise (e.g. 0.9 written then re-parsed); a real
# regression is always far larger than this. A metric rising above baseline is never a
# regression (the baseline is a floor, not an equality assertion).
_REGRESSION_EPS = 1e-9

# The safety suites whose regression is treated as the highest severity (QA-EVAL-R7): a
# drop here fails the build even though it would also fail the absolute 1.0 gate. Recorded
# so the regression report can flag a safety regression distinctly.
_SAFETY_SUITES = frozenset({"grounding", "abstention", "injection", "voice"})

# Metrics that are "higher is better" rates/counts. Every tracked metric is monotone-good
# (a higher value is never worse), so the non-regression rule is uniformly ``>=`` baseline.
# Per-suite we extract only the metrics that suite actually gates on, keyed by a stable
# ``<gate>.<metric>`` name so the diff is human-readable.
# The safety suites track ``pass_k.all_pass_rate`` — the pass^k all-trials certificate (1.0 iff
# EVERY trial passed) — NOT the per-trial ``trial_pass_rate``: a flaky safety suite passing 4/5
# trials reads 0.8 on the per-trial rate (which would silently clear the floor) but collapses to
# 0.0 on the certificate, correctly tripping this non-regression gate (QA-EVAL-R10).
_SUITE_METRICS: dict[str, tuple[str, ...]] = {
    "grounding": ("grounding.faithfulness", "schema.rate", "pass_k.all_pass_rate"),
    "abstention": ("abstention.rate", "schema.rate", "pass_k.all_pass_rate"),
    "injection": ("injection.rate", "schema.rate", "pass_k.all_pass_rate"),
    "termination": ("termination.rate",),
    "reflection_termination": ("termination.rate",),
    "intent_plan": (
        "intent_plan.precision",
        "intent_plan.recall",
        "intent_plan.intent_accuracy",
    ),
    "multilingual": ("termination.rate",),
    "judge": ("judge.passed_cases",),
    "readiness": ("readiness.consistency_rate", "readiness.voice_rate"),
    "plan": ("plan.grounding_rate", "plan.progression_rate", "plan.consistency_rate"),
    "voice": ("voice.rate", "pass_k.all_pass_rate"),
    # The no-self-certification suite (QA-EVAL-R2.10 / QA-EVAL-R6): its zero-self-certified-
    # but-ungrounded certificate (``self_cert.rate``, a 100% gate) is tracked so an erosion
    # that still clears the absolute floor still trips the non-regression gate (QA-EVAL-R7).
    "self_certification": ("self_cert.rate",),
}


def _suite_metrics(blob: Mapping[str, Any]) -> dict[str, float]:
    """Extract the comparable scalar metrics for one suite's scorecard blob.

    Only the metrics the suite gates on are tracked (``_SUITE_METRICS``); each is read out
    of the nested ``<gate>.<metric>`` path of the EVAL-R9 jsonable scorecard. A missing
    metric defaults to 0.0 so a structurally-degraded scorecard reads as a regression
    rather than silently passing.
    """
    suite = str(blob.get("suite", ""))
    out: dict[str, float] = {}
    for dotted in _SUITE_METRICS.get(suite, ()):
        gate, _, metric = dotted.partition(".")
        section = blob.get(gate, {})
        value = section.get(metric) if isinstance(section, dict) else None
        out[dotted] = float(value) if value is not None else 0.0
    return out


def build_baseline(cards: Sequence[Scorecard]) -> dict[str, Any]:
    """Build the versioned baseline document from a clean scorecard run (QA-EVAL-R7).

    Captures the format version and, per suite, the dataset version it was measured at
    plus its tracked comparable metrics. ``passed`` is recorded for human context only;
    the regression check compares the per-metric scalars, never the boolean.
    """
    suites: dict[str, Any] = {}
    for card in cards:
        blob = card.to_jsonable()
        suite = str(blob["suite"])
        suites[suite] = {
            "dataset_version": blob.get("dataset_version", ""),
            "passed": bool(blob.get("passed", False)),
            "metrics": _suite_metrics(blob),
        }
    return {
        "baseline_format_version": BASELINE_FORMAT_VERSION,
        "suites": suites,
    }


def write_baseline(cards: Sequence[Scorecard], *, path: Path | None = None) -> Path:
    """Serialize the baseline document to disk (the ``update-baseline`` writer)."""
    target = path if path is not None else BASELINE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    document = build_baseline(cards)
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_baseline(*, path: Path | None = None) -> dict[str, Any] | None:
    """Load the stored baseline document, or ``None`` if none is committed yet."""
    target = path if path is not None else BASELINE_PATH
    if not target.exists():
        return None
    loaded: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    return loaded


@dataclass(frozen=True, slots=True)
class Regression:
    """One metric that dropped below its stored baseline (QA-EVAL-R7)."""

    suite: str
    metric: str
    baseline: float
    current: float
    is_safety: bool

    def reason(self) -> str:
        tag = "SAFETY-SUITE regression" if self.is_safety else "regression"
        return (
            f"{self.suite}/{self.metric}: {tag} {self.current:.6g} < baseline {self.baseline:.6g}"
        )


@dataclass(frozen=True, slots=True)
class RegressionReport:
    """Outcome of the baseline non-regression comparison (QA-EVAL-R7).

    ``regressions`` is empty iff every tracked metric in every suite is >= its baseline.
    ``baseline_present`` is False when no baseline is committed yet (a first run before
    ``eval-update-baseline``): there is nothing to regress against, so the gate does not
    fail on its absence — but it is reported so the operator knows to seed one.
    """

    baseline_present: bool
    regressions: tuple[Regression, ...]
    new_suites: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """No metric regressed below baseline (a missing baseline never fails the gate)."""
        return not self.regressions

    @property
    def has_safety_regression(self) -> bool:
        return any(r.is_safety for r in self.regressions)

    def summary(self) -> str:
        if not self.baseline_present:
            return "no committed baseline yet; run `just eval-update-baseline` to seed one"
        if self.passed:
            return "no regression vs committed baseline"
        return "; ".join(r.reason() for r in self.regressions)


def compare_to_baseline(
    cards: Sequence[Scorecard], *, path: Path | None = None
) -> RegressionReport:
    """Compare a fresh run's scorecards against the committed baseline (QA-EVAL-R7).

    For every suite present in BOTH the baseline and the current run, every tracked metric
    must be >= its baseline (within ``_REGRESSION_EPS``); any shortfall is a regression. A
    safety-suite shortfall is flagged distinctly but is, like every regression, a hard
    fail. Suites new since the baseline are reported (not a regression — there is no prior
    value), so a freshly added I7 dataset does not break the gate before its first
    ``update-baseline``.
    """
    baseline = load_baseline(path=path)
    if baseline is None:
        return RegressionReport(baseline_present=False, regressions=(), new_suites=())
    stored = baseline.get("suites", {})
    regressions: list[Regression] = []
    new_suites: list[str] = []
    for card in cards:
        blob = card.to_jsonable()
        suite = str(blob["suite"])
        prior = stored.get(suite)
        if prior is None:
            new_suites.append(suite)
            continue
        prior_metrics = prior.get("metrics", {})
        for metric, current in _suite_metrics(blob).items():
            base = prior_metrics.get(metric)
            if base is None:
                continue
            if current < float(base) - _REGRESSION_EPS:
                regressions.append(
                    Regression(
                        suite=suite,
                        metric=metric,
                        baseline=float(base),
                        current=current,
                        is_safety=suite in _SAFETY_SUITES,
                    )
                )
    return RegressionReport(
        baseline_present=True,
        regressions=tuple(regressions),
        new_suites=tuple(new_suites),
    )


__all__ = [
    "BASELINE_FORMAT_VERSION",
    "BASELINE_PATH",
    "Regression",
    "RegressionReport",
    "build_baseline",
    "compare_to_baseline",
    "load_baseline",
    "write_baseline",
]
