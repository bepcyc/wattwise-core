"""Per-case token/cost/latency recording + the cost & latency budget gate (QA-EVAL-R8).

QA-EVAL-R8 mandates the eval harness RECORD per-case token usage, cost, and latency, and
FAIL the gate if the median cost-per-task or the p95 latency exceeds DECLARED budgets —
protecting the unit-economics constraint that agent work stay cheap.

The budgets are LOADED config content (CFG-R1a), never a code hardcode:
:meth:`CostLatencyBudget.from_settings` reads ``agent__eval__median_cost_usd`` /
``agent__eval__p95_latency_ms`` / ``agent__eval__cost_per_1k_tokens_usd`` from the resolved
settings. A run records one :class:`BudgetSample` per case (the recorded provider token
usage and latency the cassette captured, priced deterministically — QA-EVAL-R9); the
aggregate is graded with :func:`grade_budget` and folded into the scorecard.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from wattwise_core.config import load_eval_budget

if TYPE_CHECKING:
    from wattwise_core.eval.grading import SuiteGrades


@dataclass(frozen=True, slots=True)
class CostLatencyBudget:
    """The declared median-cost / p95-latency budget the gate enforces (QA-EVAL-R8).

    Values are LOADED config (CFG-R1a): :meth:`from_settings` resolves them from
    ``[agent.eval]``. No value is constructed here without an explicit caller-supplied
    number, so a budget never silently defaults to a code constant.
    """

    median_cost_usd: float
    p95_latency_ms: float
    cost_per_1k_tokens_usd: float

    @classmethod
    def load(cls) -> CostLatencyBudget:
        """Build the budget from the layered [agent.eval] config (CFG-R1a / QA-EVAL-R8).

        Reads ONLY the eval-budget keys via :func:`~wattwise_core.config.load_eval_budget`,
        so the network-free, secret-free offline eval tier (TIER-R1) resolves its cost/latency
        budgets without instantiating the secret-validated full settings. A budget key absent
        from every config layer fails closed.
        """
        values = load_eval_budget()
        return cls(
            median_cost_usd=values["agent__eval__median_cost_usd"],
            p95_latency_ms=values["agent__eval__p95_latency_ms"],
            cost_per_1k_tokens_usd=values["agent__eval__cost_per_1k_tokens_usd"],
        )

    def cost_for(self, total_tokens: int) -> float:
        """Price recorded token usage deterministically (no network, QA-EVAL-R9)."""
        return (total_tokens / 1000.0) * self.cost_per_1k_tokens_usd


@dataclass(frozen=True, slots=True)
class BudgetSample:
    """One case's recorded token usage, cost, and latency (QA-EVAL-R8 per-case record)."""

    case_id: str
    total_tokens: int
    cost_usd: float
    latency_ms: float

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
        }


def sample_from_case(case: Mapping[str, Any], budget: CostLatencyBudget) -> BudgetSample:
    """Record a case's token/cost/latency from its recorded usage (QA-EVAL-R8 / QA-EVAL-R9).

    The cassette captures the provider token usage and wall-clock latency of the real run
    under ``recorded_usage`` (``prompt_tokens`` + ``completion_tokens`` + ``latency_ms``);
    cost is derived deterministically by pricing the recorded tokens. A case that records no
    usage contributes a zero sample (the honest record for an unmeasured offline case) — the
    LIVE leg (QA-EVAL-R9) is where real per-case usage is captured.
    """
    usage = dict(case.get("recorded_usage", {}))
    prompt = int(usage.get("prompt_tokens", 0))
    completion = int(usage.get("completion_tokens", 0))
    total = prompt + completion
    latency = float(usage.get("latency_ms", 0.0))
    cost = float(usage["cost_usd"]) if "cost_usd" in usage else budget.cost_for(total)
    return BudgetSample(str(case.get("id", "")), total, cost, latency)


@dataclass(frozen=True, slots=True)
class BudgetGrade:
    """Outcome of grading the cost & latency budget (QA-EVAL-R8).

    ``median_cost_usd`` and ``p95_latency_ms`` are the measured aggregates across the run's
    per-case samples; the gate passes iff BOTH stay within their declared budgets. An empty
    run passes (no work, no cost) — but the harness records a sample per case so a real run
    always has measurements.
    """

    total: int
    total_tokens: int
    median_cost_usd: float
    p95_latency_ms: float
    budget: CostLatencyBudget
    failures: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """Gate: median cost-per-task AND p95 latency within the declared budgets."""
        return (
            self.median_cost_usd <= self.budget.median_cost_usd
            and self.p95_latency_ms <= self.budget.p95_latency_ms
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "total_tokens": self.total_tokens,
            "median_cost_usd": self.median_cost_usd,
            "p95_latency_ms": self.p95_latency_ms,
            "budget_median_cost_usd": self.budget.median_cost_usd,
            "budget_p95_latency_ms": self.budget.p95_latency_ms,
            "passed": self.passed,
            "failures": list(self.failures),
        }


def _median(values: Sequence[float]) -> float:
    """The median of a non-empty sequence (0.0 when empty)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _p95(values: Sequence[float]) -> float:
    """The p95 (nearest-rank) of a non-empty sequence (0.0 when empty)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    # Nearest-rank p95: the smallest value at or above the 95th percentile position.
    rank = max(0, min(len(ordered) - 1, _ceil_index(0.95, len(ordered))))
    return ordered[rank]


def _ceil_index(percentile: float, n: int) -> int:
    """Nearest-rank index for ``percentile`` over ``n`` items (1-based rank, 0-based index)."""
    return math.ceil(percentile * n) - 1


def grade_budget(
    samples: Sequence[Mapping[str, Any] | BudgetSample], budget: CostLatencyBudget
) -> BudgetGrade:
    """Grade per-case cost/latency against the declared budget (QA-EVAL-R8).

    ``samples`` carry per-case ``cost_usd`` and ``latency_ms`` (a :class:`BudgetSample` or
    an equivalent mapping). The gate fails when the MEDIAN cost-per-task exceeds the budget
    or the P95 latency exceeds the budget; the failure list names which bound was breached.
    """
    costs: list[float] = []
    latencies: list[float] = []
    total_tokens = 0
    for raw in samples:
        cost = float(raw.cost_usd if isinstance(raw, BudgetSample) else raw["cost_usd"])
        latency = float(raw.latency_ms if isinstance(raw, BudgetSample) else raw["latency_ms"])
        tokens = int(raw.total_tokens if isinstance(raw, BudgetSample) else raw["total_tokens"])
        costs.append(cost)
        latencies.append(latency)
        total_tokens += tokens
    median_cost = _median(costs)
    p95_latency = _p95(latencies)
    failures: list[str] = []
    if median_cost > budget.median_cost_usd:
        failures.append(
            f"median cost-per-task {median_cost} exceeds budget {budget.median_cost_usd}"
        )
    if p95_latency > budget.p95_latency_ms:
        failures.append(
            f"p95 latency {p95_latency}ms exceeds budget {budget.p95_latency_ms}ms"
        )
    return BudgetGrade(
        total=len(costs),
        total_tokens=total_tokens,
        median_cost_usd=median_cost,
        p95_latency_ms=p95_latency,
        budget=budget,
        failures=tuple(failures),
    )


def record_samples(cases: Sequence[Mapping[str, Any]]) -> tuple[BudgetSample, ...]:
    """Record one per-case token/cost/latency sample for a run (QA-EVAL-R8).

    The declared budget is LOADED from config (CFG-R1a / TIER-R1 — secret-free) so the
    recorded run can price its token usage deterministically without a network call.
    """
    budget = CostLatencyBudget.load()
    return tuple(sample_from_case(case, budget) for case in cases)


def with_budget(grades: SuiteGrades, cases: Sequence[Mapping[str, Any]]) -> SuiteGrades:
    """Fold the QA-EVAL-R8 cost/latency budget grade into the suite grades.

    The harness records per-case token/cost/latency from each case's ``recorded_usage`` (the
    cassette's captured provider usage, priced deterministically — QA-EVAL-R9) and the
    declared median-cost / p95-latency budgets are LOADED from config (CFG-R1a). The budget
    grade gates the suite: a median cost-per-task or p95 latency above the declared budget
    fails it (QA-EVAL-R8).
    """
    budget = CostLatencyBudget.load()
    samples = [sample_from_case(case, budget) for case in cases]
    return replace(grades, budget=grade_budget(samples, budget))


__all__ = [
    "BudgetGrade",
    "BudgetSample",
    "CostLatencyBudget",
    "grade_budget",
    "record_samples",
    "sample_from_case",
    "with_budget",
]
