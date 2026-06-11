"""Live-mode eval run: INFRA_ERROR classification + the max-infra-rate gate (QA-EVAL-R12(b)).

A live run (QA-EVAL-R9: the nightly/release-candidate leg against the real provider)
MUST distinguish INFRASTRUCTURE failures — provider unavailability, network timeouts,
rate-limit exhaustion — from QUALITY regressions:

* an infrastructure failure lands under the distinct ``INFRA_ERROR`` status — never
  ``FAIL`` and never silently counted as a pass;
* the run carries a configured maximum infrastructure-error rate
  (``[agent.eval].max_infra_error_rate``, CFG-R1a); exceeding it ALERTS and BLOCKS
  promotion until a clean run is obtained — an infra blip never masks a real
  regression, and a perpetually failing run is never dismissed as flake;
* a quality failure (a suite genuinely below threshold) stays a ``FAIL`` and alerts as
  a regression.

The live leg is env-gated (``WATTWISE_LLM_API_KEY``) and NEVER part of the offline
``-n auto`` gate (TIER-R1); CI-R4 schedules it nightly.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum

from wattwise_core.config.settings import load_eval_budget
from wattwise_core.eval.scorecard import EvalMode, Scorecard


async def _runner_run_suite(suite: str) -> Scorecard:
    """Indirection over the runner entry (kept here so mocking one seam is enough)."""
    # The runner package imports this module's grades; a top-level import is a cycle.
    from wattwise_core.eval import runner as _runner  # noqa: PLC0415

    return await _runner.run_suite(suite, mode=EvalMode.LIVE)


#: HTTP statuses that mark a provider/infrastructure failure (rate-limit + 5xx).
_INFRA_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class LiveStatus(StrEnum):
    """Per-suite live-run status (QA-EVAL-R12(b)): infra is NEVER conflated with FAIL."""

    PASS = "pass"  # noqa: S105 - a status token, not a credential
    FAIL = "fail"
    INFRA_ERROR = "infra_error"


def classify_infra(exc: BaseException) -> bool:
    """``True`` iff the exception is an INFRASTRUCTURE failure, not a quality signal.

    Provider unavailability, network timeouts, connection drops, and rate-limit
    exhaustion are infrastructure; assertion/grading errors are quality. Detection is
    typed-first (``TimeoutError``/``ConnectionError``/``OSError``) with an HTTP
    status-code fallback for provider-client exceptions carrying ``status_code``/
    ``response.status_code`` in the infra set (429/5xx/408).
    """
    if isinstance(exc, TimeoutError | ConnectionError | asyncio.TimeoutError):
        return True
    if isinstance(exc, OSError):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    return status in _INFRA_HTTP_STATUSES


@dataclass(frozen=True, slots=True)
class LiveSuiteResult:
    """One suite's live outcome: its status plus the scorecard when the run completed."""

    suite: str
    status: LiveStatus
    detail: str = ""
    scorecard: Scorecard | None = None


@dataclass(frozen=True, slots=True)
class LiveRunReport:
    """Aggregate live-run verdict (QA-EVAL-R12(b)): quality and infra gated SEPARATELY."""

    results: tuple[LiveSuiteResult, ...]
    max_infra_error_rate: float

    @classmethod
    def from_results(cls, results: tuple[LiveSuiteResult, ...]) -> LiveRunReport:
        """Build the report under the CONFIGURED max infra rate (CFG-R1a, never a literal)."""
        values = load_eval_budget()
        return cls(results, float(values["agent__eval__max_infra_error_rate"]))

    @property
    def infra_error_rate(self) -> float:
        """Fraction of suites that landed INFRA_ERROR (0.0 for an empty run)."""
        if not self.results:
            return 0.0
        infra = sum(1 for r in self.results if r.status is LiveStatus.INFRA_ERROR)
        return infra / len(self.results)

    @property
    def quality_failed(self) -> tuple[str, ...]:
        """Suites with a genuine quality FAIL (alert as regression, QA-EVAL-R7)."""
        return tuple(r.suite for r in self.results if r.status is LiveStatus.FAIL)

    @property
    def infra_blocked(self) -> bool:
        """``True`` when the infra rate exceeds the configured max: alert + block promotion."""
        return self.infra_error_rate > self.max_infra_error_rate

    @property
    def clean(self) -> bool:
        """A clean live run: zero quality failures AND zero infrastructure errors.

        Promotion / baseline advancement requires a CLEAN run (QA-EVAL-R12(c)); a
        within-budget infra blip still bars baseline advancement, it only avoids the
        blocking alert.
        """
        return not self.quality_failed and self.infra_error_rate == 0.0

    def alert_lines(self) -> tuple[str, ...]:
        """Human-readable alert lines (CI-R4: a live regression/infra breach must alert)."""
        lines: list[str] = []
        if self.quality_failed:
            lines.append("ALERT quality regression in live eval: " + ", ".join(self.quality_failed))
        if self.infra_blocked:
            lines.append(
                f"ALERT live-eval INFRA_ERROR rate {self.infra_error_rate:.2f} exceeds "
                f"max {self.max_infra_error_rate:.2f}: promotion BLOCKED until a clean run"
            )
        return tuple(lines)


#: Failure-text tokens that mark an infrastructure failure in the live smoke's junit
#: output (the env-gated ``llm`` pytest tier driven by ``--mode=live``).
_INFRA_TEXT_TOKENS = (
    "timeout",
    "timed out",
    "connection",
    "connect error",
    "rate limit",
    "rate-limit",
    "429",
    "502",
    "503",
    "504",
    "unavailable",
    "temporarily",
)


def classify_infra_text(text: str) -> bool:
    """Classify a live-smoke failure MESSAGE as infrastructure vs quality (R12(b)).

    The live smoke runs under pytest, which surfaces provider failures as failure text
    rather than raisable exceptions; the same infra taxonomy (unavailability, timeout,
    rate-limit) is detected from the message.
    """
    lowered = text.lower()
    return any(token in lowered for token in _INFRA_TEXT_TOKENS)


async def run_live_suite(suite: str) -> LiveSuiteResult:
    """Run one suite in LIVE mode, classifying any infrastructure failure (R12(b)).

    Wraps the suite execution so a provider/network failure lands as ``INFRA_ERROR``
    (with the exception detail) instead of FAIL; a completed run maps the scorecard's
    pass/fail verdict onto PASS/FAIL.
    """
    try:
        card = await _runner_run_suite(suite)
    except Exception as exc:
        if classify_infra(exc):
            return LiveSuiteResult(suite, LiveStatus.INFRA_ERROR, detail=repr(exc))
        return LiveSuiteResult(suite, LiveStatus.FAIL, detail=repr(exc))
    status = LiveStatus.PASS if card.passed else LiveStatus.FAIL
    return LiveSuiteResult(suite, status, scorecard=card)


__all__ = [
    "LiveRunReport",
    "LiveStatus",
    "LiveSuiteResult",
    "classify_infra",
    "classify_infra_text",
    "run_live_suite",
]
