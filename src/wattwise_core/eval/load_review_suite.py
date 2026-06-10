"""Weekly load-review suite (QA-EVAL-R2.3): CTL/ATL trend + form + notable sessions.

Each case fixes a canonical training history (per-day loads) and the load review the
agent DELIVERED for the final week: the stated end-of-week CTL/ATL/form, the stated
CTL trend over the review week, the named notable sessions, and the athlete-facing
summary. Grading is deterministic and programmatic (QA-EVAL-R3): the oracle recomputes
the PMC over the SAME canonical loads with the SHIPPED production
:func:`wattwise_core.analytics.pmc.pmc` and asserts

* the stated CTL/ATL/form equal the recomputed values within tolerance (numeric
  consistency with canonical analytics — never the model's own arithmetic);
* the stated trend direction matches the recomputed CTL movement across the week;
* every named notable session is one of the week's genuinely heaviest days, and the
  single heaviest day is named (a review that misses the week's biggest session, or
  invents one, fails);
* every number surfaced in the summary text is canonical — present in the recomputed
  metrics or the week's loads within tolerance (no fabricated numbers, QA-EVAL-R2.1
  applied to the digest prose).

Network-free and deterministic (TIER-R1, QA-EVAL-R9); the gate floor lives in
QA-EVAL-R6 (coach suites).
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wattwise_core.analytics.pmc import pmc
from wattwise_core.analytics.result import Computed

_DATASETS_DIR = Path(__file__).parent / "datasets"

# Numeric-consistency tolerance for STATED metric values vs the recomputed canonical
# PMC (one decimal of display rounding on CTL/ATL/form-scale numbers).
_VALUE_TOL = 0.06
# CTL movement smaller than this over the review week reads as "steady".
_TREND_EPS = 0.5
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class LoadReviewGrade:
    """Outcome of grading the weekly load-review suite (QA-EVAL-R2.3).

    One deterministic 100% gate: a case is *consistent* iff its stated CTL/ATL/form
    match the recomputed canonical PMC, the trend direction matches, the notable
    sessions are the genuinely heaviest days, and the summary surfaces only canonical
    numbers. ``failures`` records every defect so the rate alone can never mask one.
    """

    total: int
    consistent: int
    failures: tuple[str, ...] = ()

    @property
    def consistency_rate(self) -> float:
        """Fraction of cases numerically consistent with canonical analytics."""
        return 1.0 if self.total == 0 else self.consistent / self.total

    @property
    def passed(self) -> bool:
        """Gate: 100% consistency AND zero recorded failures (QA-EVAL-R6 coach floor)."""
        return self.consistency_rate >= 1.0 and self.failures == ()


def _load_cases() -> list[dict[str, Any]]:
    raw = json.loads((_DATASETS_DIR / "load_review.json").read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = raw["cases"]
    return cases


def _recompute(case: dict[str, Any]) -> tuple[dict[str, float], dict[_dt.date, float]]:
    """Recompute end-of-week CTL/ATL/form + the week-ago CTL from the canonical loads."""
    loads = {_dt.date.fromisoformat(day): float(load) for day, load in case["daily_loads"].items()}
    series = pmc(loads)
    computed = [d for d in series if isinstance(d, Computed)]
    last = computed[-1].value
    week_ago = computed[-8].value if len(computed) >= 8 else computed[0].value
    metrics = {
        "ctl": last.ctl,
        "atl": last.atl,
        "form": last.tsb,
        "ctl_week_ago": week_ago.ctl,
    }
    return metrics, loads


def _trend_of(ctl_end: float, ctl_week_ago: float) -> str:
    if ctl_end > ctl_week_ago + _TREND_EPS:
        return "rising"
    if ctl_end < ctl_week_ago - _TREND_EPS:
        return "falling"
    return "steady"


def _notable_failures(case: dict[str, Any], loads: dict[_dt.date, float]) -> list[str]:
    """The notable sessions must BE the heaviest review-week days (no misses/inventions)."""
    cid = case["id"]
    review_days = sorted(loads)[-7:]
    by_load = sorted(
        (d for d in review_days if loads[d] > 0.0), key=lambda d: loads[d], reverse=True
    )
    stated = [_dt.date.fromisoformat(d) for d in case["review"]["notable_sessions"]]
    failures: list[str] = []
    if not by_load:
        if stated:
            failures.append(f"{cid}: notable sessions named in a zero-load week")
        return failures
    top = set(by_load[: max(len(stated), 1)])
    if by_load[0] not in stated:
        failures.append(f"{cid}: the week's heaviest session {by_load[0]} is not named")
    for day in stated:
        if day not in top:
            failures.append(f"{cid}: named notable session {day} is not a heaviest day")
    return failures


def _summary_number_failures(
    case: dict[str, Any], metrics: dict[str, float], loads: dict[_dt.date, float]
) -> list[str]:
    """Every number surfaced in the summary must be canonical within tolerance."""
    canonical = [
        *metrics.values(),
        *(abs(v) for v in metrics.values()),
        *loads.values(),
        float(sum(loads[d] for d in sorted(loads)[-7:])),
    ]
    failures: list[str] = []
    for token in _NUMBER.findall(case["summary_text"]):
        value = float(token)
        if not any(abs(value - c) <= _VALUE_TOL for c in canonical):
            failures.append(f"{case['id']}: summary number {token} is not canonical")
    return failures


def _case_failures(case: dict[str, Any]) -> list[str]:
    cid = case["id"]
    metrics, loads = _recompute(case)
    review = case["review"]
    failures: list[str] = []
    for key in ("ctl", "atl", "form"):
        stated = float(review[key])
        if abs(stated - metrics[key]) > _VALUE_TOL:
            failures.append(f"{cid}: stated {key} {stated} != canonical {metrics[key]:.3f}")
    expected_trend = _trend_of(metrics["ctl"], metrics["ctl_week_ago"])
    if str(review["trend"]) != expected_trend:
        failures.append(f"{cid}: stated trend {review['trend']!r} != canonical {expected_trend!r}")
    failures.extend(_notable_failures(case, loads))
    failures.extend(_summary_number_failures(case, metrics, loads))
    return failures


def grade_load_review() -> LoadReviewGrade:
    """Grade the weekly load-review fixtures deterministically (QA-EVAL-R2.3).

    The oracle is the SHIPPED PMC implementation over the case's canonical loads —
    the same numbers the grounding gate would certify — so a review whose trend, form,
    notable sessions, or surfaced numbers drift from canonical analytics fails.
    """
    cases = _load_cases()
    failures: list[str] = []
    consistent = 0
    for case in cases:
        case_failures = _case_failures(case)
        if case_failures:
            failures.extend(case_failures)
        else:
            consistent += 1
    return LoadReviewGrade(len(cases), consistent, tuple(failures))


__all__ = ["LoadReviewGrade", "grade_load_review"]
