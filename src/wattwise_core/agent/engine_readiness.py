"""Readiness input-gathering + narration wiring for the production agent engine.

The focused sibling of :mod:`wattwise_core.agent.engine` that owns the readiness/form
plumbing the deployable :class:`~wattwise_core.agent.engine.GraphAgentEngine` drives
(QUAL-R9 size split): the DETERMINISTIC canonical-input gather and the structured
narration closure. It is kept here (not in ``engine``) so the engine module stays under
the size ceiling while the readiness JTBD's concrete wiring lives in one place.

The readiness JTBD is FIXED — its inputs are gathered deterministically (latest canonical
TSB/form + its date, latest HRV), NOT via the retrieval planner — and any unavailable
input fails closed to ``None`` so the deliverable abstains/degrades truthfully rather than
guessing. The narration closure wraps structured output at temperature 0 and raises
:class:`~wattwise_core.agent.readiness_deliverable.StructuredNarrationError` on a provider
failure so the deliverable falls back to its deterministic per-verdict state sentence.

Cited requirements: QA-EVAL-R2.4, GROUND-R7, STRUCT-R1, RUN-R4.1.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from wattwise_core.agent.contracts import ChatModel
from wattwise_core.agent.readiness_deliverable import (
    StructuredNarrationError,
    _ReadinessNarration,
)
from wattwise_core.agent.structured import StructuredOutputError, run_structured
from wattwise_core.analytics.constants import READINESS_MIN_FITNESS_CTL
from wattwise_core.analytics.result import is_computed
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.persistence.types import utcnow

# The trailing window the readiness gather scans for the latest canonical TSB (form) and
# HRV day. The readiness JTBD is FIXED — its inputs are gathered deterministically here,
# NOT via the retrieval planner (which is restricted to date-range capabilities). A 14-day
# look-back finds the most-recent computed day without dragging in stale state.
_READINESS_WINDOW_DAYS = 14


async def gather_readiness_inputs(
    svc: AnalyticsService, athlete_id: str
) -> tuple[float | None, str | None, float | None, float | None]:
    """Deterministically fetch the readiness inputs from canonical analytics (QA-EVAL-R2.4).

    Returns ``(form, as_of, hrv_rmssd, hrv_baseline)``. ``form`` is the latest computed
    canonical TSB over a trailing window and ``as_of`` its date (the most-recent
    :class:`Computed` PMC day); ``hrv_rmssd`` is the latest computed HRV (RMSSD, ms) for
    that day and ``hrv_baseline`` the athlete's HRV baseline for that SAME day (the midpoint
    of the source-reported ``hrv_baseline_low/high_ms`` band, read via
    :meth:`AnalyticsService.hrv_baseline`). When BOTH the RMSSD and the baseline are present
    the oracle's HRV-suppression nudge can fire (COACH-R1 #2); when the baseline is missing it
    fails closed to ``None`` and the verdict reads from form alone. Any unavailable input
    fails closed to ``None`` (mirroring the analytics Computed/Unavailable envelope) so the
    oracle abstains/degrades truthfully rather than guessing. This bypasses the retrieval
    planner by design — the readiness JTBD is fixed.
    """
    today = utcnow().date()
    start = today - _dt.timedelta(days=_READINESS_WINDOW_DAYS)
    series = await svc.pmc(athlete_id, start, today)
    form, as_of = _latest_form(series, start)
    if as_of is None:
        return form, as_of, None, None
    rmssd = await _latest_hrv_rmssd(svc, athlete_id, as_of)
    baseline = await svc.hrv_baseline(athlete_id, _dt.date.fromisoformat(as_of))
    return form, as_of, rmssd, baseline


def _latest_form(series: Sequence[Any], start: _dt.date) -> tuple[float | None, str | None]:
    """The most-recent computed TSB (form) + its ISO date, or ``(None, None)`` (fail-closed).

    The PMC series is returned oldest-first over ``[start, today]``; the latest computed
    day carries the canonical TSB the readiness verdict reads. An all-unavailable window
    (no rides → no computable form) yields ``(None, None)`` so the deliverable abstains.

    Cold-start guard (GROUND-R6 / PMC-R3/R5): a brand-new athlete with zero rides still gets
    an HONEST (0,0) origin seed, which materializes as a Computed PMC day with ctl≈atl≈tsb≈0
    — form 0.0 would otherwise read as a confident MAINTAIN ("keep training") on NO data. So
    when the latest computed day's ctl is below :data:`READINESS_MIN_FITNESS_CTL` (no real
    chronic training base accumulated — only a (0,0) cold-start, or a fully-detrained athlete
    with no recent fitness signal), the form is treated as UNAVAILABLE and ``(None, None)`` is
    returned so the deliverable abstains rather than emitting a verdict on an empty base.
    """
    for offset, day in enumerate(reversed(series)):
        if is_computed(day):
            ctl = day.value.ctl
            if ctl is None or not math.isfinite(ctl) or ctl < READINESS_MIN_FITNESS_CTL:
                return None, None
            as_of = start + _dt.timedelta(days=len(series) - 1 - offset)
            return float(day.value.tsb), as_of.isoformat()
    return None, None


async def _latest_hrv_rmssd(
    svc: AnalyticsService, athlete_id: str, as_of: str | None
) -> float | None:
    """The canonical HRV (RMSSD, ms) for the form day, or ``None`` when unavailable.

    Reads the same canonical day the form is as-of so the two inputs are coherent; an
    Unavailable HRV envelope (no wellness row / too-artifact-laden) fails closed to
    ``None`` (the oracle then records HRV unavailable and reads the verdict from form).
    """
    if as_of is None:
        return None
    day = _dt.date.fromisoformat(as_of)
    result = await svc.hrv(athlete_id, day)
    return float(result.value.rmssd_ms) if is_computed(result) else None


def readiness_narrator(
    model: ChatModel, *, system: str = ""
) -> Callable[[str], Awaitable[_ReadinessNarration]]:
    """Build the structured-narration closure the readiness deliverable drives (STRUCT-R1).

    Wraps ``run_structured`` over the closed :class:`_ReadinessNarration` schema at
    temperature 0 (provider-enforced); on a structured-output failure it raises
    :class:`StructuredNarrationError` so the deliverable falls back to the deterministic
    per-verdict state sentence rather than surfacing a model failure (fail-closed voice).

    ``system`` is the externalized readiness-narration system prompt (§16 / SKILL-R1, CFG-R3): the
    engine embeds NO prompt inline (ARCH-R29) — the engine threads the verbatim fragment loaded from
    the coach-config bundle. The empty default keeps the FakeModel suite green (it scripts the
    narration, so the prompt text is immaterial offline).
    """

    async def narrate(context: str) -> _ReadinessNarration:
        try:
            return await run_structured(
                model, system=system, data=context, schema=_ReadinessNarration
            )
        except (StructuredOutputError, NotImplementedError) as exc:
            raise StructuredNarrationError("readiness narration unavailable") from exc

    return narrate


__all__ = [
    "gather_readiness_inputs",
    "readiness_narrator",
]
