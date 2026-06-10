"""The data-quality / coverage DIAGNOSIS deliverable: a grounded, fail-closed narration (API-R15).

The focused sibling of :mod:`wattwise_core.agent.deliverables` (QUAL-R9 size split) that owns the
data-quality / coverage diagnosis deliverable and NOTHING else (API-R15). Unlike the free-form
answer, the readiness verdict, or the multi-day plan, a DIAGNOSIS is purely DETERMINISTIC: it
narrates which canonical analytic inputs are PRESENT, MISSING, or STALE for an athlete by probing
the SAME canonical analytics service the rest of the engine grounds against — it makes NO model
call and routes through NO retrieval planner, so there is nothing for a model to fabricate
(GROUND-R7, OUTCOME-R5). Every reported input status is read VERBATIM off the analytics
``Computed`` / ``Unavailable`` envelope (doc 40 §1): a probe that fails closed to ``Unavailable``
is reported missing/stale with its typed reason, never silently treated as present and never
invented as a confident number.

The deliverable is therefore the coverage narration the agent uses to explain WHY a downstream
answer would degrade — "your power-curve has no recent rides, so I can't speak to your sprint" —
grounded only in what canonical analytics actually computed. It surfaces NO athlete-facing numbers
(VOICE-R7): each input is a typed ``present|missing|stale`` status with a jargon-free label and,
for a degraded input, the analytics reason — never a metric value (the coverage caveat shape
OUTCOME-R4 already uses for a degraded answer).

Cited requirements: API-R15, GROUND-R7, OUTCOME-R3/-R4/-R5, ANL-R3/-R4, COACH-R7, VOICE-R2/-R7.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from wattwise_core.agent.contracts import RunStatus
from wattwise_core.analytics.constants import READINESS_MIN_FITNESS_CTL
from wattwise_core.analytics.result import MetricResult, is_computed
from wattwise_core.analytics.service import AnalyticsService

# The trailing window the diagnosis probes for canonical coverage. A canonical input present
# only OUTSIDE this window is reported STALE (present historically but not recently enough to
# back a current answer); a 42-day look-back matches the PMC chronic-load constant the rest of
# the engine reasons over, so "recent enough" is consistent across deliverables.
DIAGNOSIS_WINDOW_DAYS = 42


class InputStatus(StrEnum):
    """The closed coverage status of one canonical input (API-R15).

    ``PRESENT`` — canonical analytics computed the input within the recent window.
    ``STALE`` — the input is computable historically but not within the recent window.
    ``MISSING`` — canonical analytics fails closed (``Unavailable``) for the input.
    """

    PRESENT = "present"
    STALE = "stale"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class InputCoverage:
    """One canonical input's typed coverage line in a diagnosis (API-R15, OUTCOME-R4).

    ``key`` is a stable machine id the client branches on (``training_load`` / ``power_curve``
    / ``critical_power`` / ``hrv`` / ``fitness_signature``); ``label`` is the jargon-free
    athlete-native name (VOICE-R2). ``status`` is the closed :class:`InputStatus`; ``reason``
    is the typed analytics ``Unavailable`` reason for a ``missing`` input (doc 40 §1) or ``None``
    when the input is present/stale. There is deliberately NO numeric field — a diagnosis reports
    coverage, never a canonical value (VOICE-R7 / GROUND-R7).
    """

    key: str
    label: str
    status: InputStatus
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class AgentDiagnosis:
    """A grounded, fail-closed data-quality / coverage diagnosis (API-R15).

    Projected DETERMINISTICALLY from the canonical analytics envelope (no model call): the
    per-input coverage lines plus a status-discriminated outcome. ``status`` is ``completed``
    when at least one canonical input is present and ``degraded`` when none is (the athlete has
    no usable canonical coverage at all, so any downstream answer would abstain — OUTCOME-R3).
    ``as_of`` is the ISO date the probe windowed against. The diagnosis surfaces NO athlete-facing
    numbers; ``coverage_caveat`` mirrors the OUTCOME-R4 typed note the API renders in coach voice.
    """

    status: RunStatus
    athlete_id: str
    as_of: str
    inputs: tuple[InputCoverage, ...] = ()
    coverage_caveat: dict[str, Any] | None = None


def _range_status(results: Sequence[MetricResult[Any]]) -> InputStatus:
    """Classify a windowed series of metric results into present/missing (fail-closed).

    A date-range probe (PMC / power-curve) computes one result per qualifying input in the
    window; if ANY is ``Computed`` the input is PRESENT, else the whole window failed closed
    and the input is MISSING. Staleness for a range probe is handled by the window itself — a
    series with no computed day in the recent window simply reports missing.
    """
    return InputStatus.PRESENT if any(is_computed(r) for r in results) else InputStatus.MISSING


def _scalar_status(result: MetricResult[Any]) -> InputStatus:
    """Classify a single metric result into present/missing (fail-closed, ANL-R4)."""
    return InputStatus.PRESENT if is_computed(result) else InputStatus.MISSING


def _reason_of(result: MetricResult[Any]) -> str | None:
    """The typed ``Unavailable`` reason for a missing input, else ``None`` (doc 40 §1)."""
    return None if is_computed(result) else result.reason.value


def _has_real_fitness(series: Sequence[MetricResult[Any]]) -> bool:
    """True iff a computed PMC day carries a real chronic base, not just a cold-start seed.

    The canonical PMC seeds an HONEST ``(0,0)`` origin for a brand-new athlete (PMC-R3/R5), so an
    athlete with ZERO real training still gets ``Computed`` days with ctl≈atl≈tsb≈0. Treating that
    as "training-load coverage" would over-report a present input on no data, so — mirroring the
    readiness cold-start guard (GROUND-R6) — a series whose every computed day's ctl is below
    :data:`READINESS_MIN_FITNESS_CTL` is NOT real coverage. A finite ctl at or above the epsilon on
    any computed day is the real chronic-base signal the diagnosis reports as present.
    """
    for day in series:
        if is_computed(day):
            ctl = day.value.ctl
            if ctl is not None and math.isfinite(ctl) and ctl >= READINESS_MIN_FITNESS_CTL:
                return True
    return False


async def _training_load_coverage(
    svc: AnalyticsService, athlete_id: str, start: _dt.date, today: _dt.date
) -> InputCoverage:
    """Probe canonical training-load (PMC) coverage over the recent window (API-R15).

    Reads the SAME canonical PMC series the readiness/answer flows ground against; a window with a
    real chronic base (a computed day whose ctl clears the cold-start epsilon, :func:`_has_real_
    fitness`) is PRESENT, else MISSING — a brand-new athlete's honest ``(0,0)`` cold-start seed is
    NOT over-reported as coverage (GROUND-R6). Never fabricates a load number — only its presence
    is reported (GROUND-R7).
    """
    series = await svc.pmc(athlete_id, start, today)
    status = InputStatus.PRESENT if _has_real_fitness(series) else InputStatus.MISSING
    return InputCoverage(
        key="training_load",
        label="Training load",
        status=status,
        reason=None if status is InputStatus.PRESENT else "insufficient_data",
    )


async def _power_curve_coverage(
    svc: AnalyticsService, athlete_id: str, start: _dt.date, today: _dt.date
) -> InputCoverage:
    """Probe canonical power-curve coverage over the recent window (API-R15)."""
    curve = await svc.power_curve(athlete_id, start, today)
    status = _range_status(tuple(curve.values()))
    return InputCoverage(key="power_curve", label="Power profile", status=status)


async def _critical_power_coverage(
    svc: AnalyticsService, athlete_id: str, start: _dt.date, today: _dt.date
) -> InputCoverage:
    """Probe canonical critical-power coverage over the recent window (API-R15)."""
    result = await svc.critical_power(athlete_id, start, today)
    return InputCoverage(
        key="critical_power",
        label="Critical power",
        status=_scalar_status(result),
        reason=_reason_of(result),
    )


async def _hrv_coverage(svc: AnalyticsService, athlete_id: str, today: _dt.date) -> InputCoverage:
    """Probe canonical HRV coverage for today (API-R15); fail-closed when no wellness row."""
    result = await svc.hrv(athlete_id, today)
    return InputCoverage(
        key="hrv", label="Recovery (HRV)", status=_scalar_status(result), reason=_reason_of(result)
    )


async def _signature_coverage(
    svc: AnalyticsService, athlete_id: str, today: _dt.date
) -> InputCoverage:
    """Probe whether an effective fitness signature (FTP/CP) resolves today (API-R15).

    The signature is the reference input every power metric depends on; without it the load and
    power deliverables degrade. The signature is SPORT-KEYED, so the probe resolves the athlete's
    CURRENT sport from the canonical profile (GBO-R13b) and grounds against THAT sport's signature —
    never a hardcoded ``"cycling"`` (CFG-R1a): a runner's running signature must read PRESENT and a
    runner's stale cycling signature must NOT. With no current sport set there is no sport to ground
    against, so coverage is reported MISSING (fail-closed, never a guessed sport).
    ``resolve_signature`` itself fails closed to an EMPTY params object when no effective signature
    exists for the sport, which is likewise reported MISSING (never a fabricated FTP).
    """
    sport = await svc.current_sport(athlete_id)
    if sport is None:
        return InputCoverage(
            key="fitness_signature",
            label="Fitness signature (FTP/CP)",
            status=InputStatus.MISSING,
            reason="no_current_sport",
        )
    params = await svc.resolve_signature(athlete_id, sport, today)
    present = params.ftp_w is not None or params.cp_w is not None
    return InputCoverage(
        key="fitness_signature",
        label="Fitness signature (FTP/CP)",
        status=InputStatus.PRESENT if present else InputStatus.MISSING,
        reason=None if present else "missing_required_input",
    )


async def diagnose_coverage(
    svc: AnalyticsService, athlete_id: str, *, today: _dt.date | None = None
) -> AgentDiagnosis:
    """Build the grounded, fail-closed data-quality / coverage diagnosis (API-R15).

    DETERMINISTIC end to end: probes each canonical input through the analytics service and
    projects the typed ``Computed``/``Unavailable`` envelope into per-input coverage lines — no
    model call, no retrieval planner, nothing to fabricate (GROUND-R7 / OUTCOME-R5). The run is
    ``completed`` when at least one input is present and ``degraded`` (with an OUTCOME-R4 caveat)
    when the athlete has NO usable canonical coverage at all. ``athlete_id`` is the server-derived
    owner (AGT-SEC-R1); the probe is read-only.
    """
    today = today or _dt.datetime.now(_dt.UTC).date()
    start = today - _dt.timedelta(days=DIAGNOSIS_WINDOW_DAYS)
    inputs = (
        await _training_load_coverage(svc, athlete_id, start, today),
        await _power_curve_coverage(svc, athlete_id, start, today),
        await _critical_power_coverage(svc, athlete_id, start, today),
        await _hrv_coverage(svc, athlete_id, today),
        await _signature_coverage(svc, athlete_id, today),
    )
    any_present = any(i.status is InputStatus.PRESENT for i in inputs)
    status = RunStatus.COMPLETED if any_present else RunStatus.DEGRADED
    caveat: dict[str, Any] | None = None
    if not any_present:
        missing = [i.key for i in inputs if i.status is not InputStatus.PRESENT]
        caveat = {"reason": "no_canonical_coverage", "inputs_unavailable": missing}
    return AgentDiagnosis(
        status=status,
        athlete_id=athlete_id,
        as_of=today.isoformat(),
        inputs=inputs,
        coverage_caveat=caveat,
    )


__all__ = [
    "DIAGNOSIS_WINDOW_DAYS",
    "AgentDiagnosis",
    "InputCoverage",
    "InputStatus",
    "diagnose_coverage",
]
