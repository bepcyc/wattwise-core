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
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.agent.contracts import ChatModel
from wattwise_core.agent.readiness_deliverable import (
    StructuredNarrationError,
    _ReadinessNarration,
)
from wattwise_core.agent.structured import StructuredOutputError, run_structured
from wattwise_core.analytics.constants import (
    READINESS_FRESH_STALENESS_DAYS,
    READINESS_MAX_STALENESS_DAYS,
    READINESS_MIN_FITNESS_CTL,
)
from wattwise_core.analytics.result import is_computed
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.analytics.sufficiency import RecordSufficiency, assess_record_sufficiency
from wattwise_core.domain.enums import ConnectionStatus, Fidelity
from wattwise_core.persistence.models import Connection
from wattwise_core.persistence.types import utcnow

# The trailing window the readiness gather scans for the latest canonical TSB (form) and
# HRV day. The readiness JTBD is FIXED — its inputs are gathered deterministically here,
# NOT via the retrieval planner (which is restricted to date-range capabilities). A 14-day
# look-back finds the most-recent computed day without dragging in stale state.
_READINESS_WINDOW_DAYS = 14


@dataclass(frozen=True, slots=True)
class ReadinessInputs:
    """The deterministically-gathered readiness inputs + the record's sufficiency (QA-EVAL-R2.4).

    ``form``/``as_of``/``hrv_rmssd``/``hrv_baseline`` are the canonical metric reads (any
    unavailable input is ``None``, fail-closed). ``sufficiency`` is the typed record-freshness +
    load-fidelity envelope (GROUND-R6 / DEGR-R2) the deliverable applies its asymmetric fail-closed
    policy against, so a current-state verdict is never asserted on a record whose observed data has
    gone stale behind a silently-withdrawn connector.
    """

    form: float | None
    as_of: str | None
    hrv_rmssd: float | None
    hrv_baseline: float | None
    sufficiency: RecordSufficiency


def connection_is_suspect(
    status: ConnectionStatus,
    last_synced_at: _dt.datetime | None,
    *,
    reference_date: _dt.date,
    sync_stale_after_days: int,
) -> bool:
    """True iff this connector should be delivering data but is broken or silently stalled.

    Pure predicate (the MNAR disambiguator behind the readiness freshness gate). A connector in
    ``REAUTH_REQUIRED``/``ERROR`` is overtly broken — it cannot deliver, so a data gap behind it is
    likely MISSING data, not rest. A ``CONNECTED`` source that has never synced, or whose last
    successful sync is itself older than ``sync_stale_after_days``, is silently stalled (the failure
    mode an expired credential leaves when the status flag lags). A ``DISCONNECTED`` source is an
    INTENTIONAL athlete action — no data is expected, so its gap is never suspect (fail-open: we do
    not manufacture a sync alarm from a deliberate disconnect).
    """
    if status in (ConnectionStatus.REAUTH_REQUIRED, ConnectionStatus.ERROR):
        return True
    if status is ConnectionStatus.CONNECTED:
        if last_synced_at is None:
            return True
        gap = (reference_date - last_synced_at.date()).days
        return gap > sync_stale_after_days
    return False


async def connection_sync_suspect(
    session: AsyncSession,
    athlete_id: str,
    *,
    reference_date: _dt.date,
    sync_stale_after_days: int,
) -> bool:
    """True iff ANY of the athlete's connectors is broken or silently stalled (issue #12 signal).

    Reads the canonical :class:`Connection` rows for the server-derived athlete (AGT-SEC-R1,
    athlete-scoped) and classifies each with :func:`connection_is_suspect`. An athlete with NO
    connectors (a pure manual-file-upload account) has no pipeline that could silently fail, so the
    gap is never "suspect" here — the freshness gate then trusts the record, as for a healthy
    sync. A malformed athlete id yields ``False`` (fail-soft for an input signal, never fail-open
    identity), so the deliverable still runs without the freshness corroboration.
    """
    try:
        aid = uuid.UUID(athlete_id)
    except (ValueError, AttributeError):  # pragma: no cover - athlete_id is server-derived/valid
        return False
    rows = (
        (await session.execute(select(Connection).where(Connection.athlete_id == aid)))
        .scalars()
        .all()
    )
    return any(
        connection_is_suspect(
            row.status,
            row.last_synced_at,
            reference_date=reference_date,
            sync_stale_after_days=sync_stale_after_days,
        )
        for row in rows
    )


async def gather_readiness_inputs(
    svc: AnalyticsService, athlete_id: str, *, sync_suspect: bool = False
) -> ReadinessInputs:
    """Deterministically fetch the readiness inputs from canonical analytics (QA-EVAL-R2.4).

    Returns a :class:`ReadinessInputs`. ``form`` is the latest computed canonical TSB over a
    trailing window and ``as_of`` its date (the most-recent :class:`Computed` PMC day);
    ``hrv_rmssd`` is the latest computed HRV (RMSSD, ms) for that day and ``hrv_baseline`` the
    athlete's HRV baseline for that SAME day (the midpoint of the source-reported
    ``hrv_baseline_low/high_ms`` band, read via :meth:`AnalyticsService.hrv_baseline`). When BOTH
    the RMSSD and the baseline are present the oracle's HRV-suppression nudge can fire (COACH-R1);
    when the baseline is missing it fails closed to ``None`` and the verdict reads from form alone.
    Any unavailable input fails closed to ``None`` (mirroring the analytics Computed/Unavailable
    envelope) so the oracle abstains/degrades truthfully rather than guessing.

    Crucially it ALSO measures record SUFFICIENCY (GROUND-R6): the latest PMC grid day is always
    filled to today (PMC-R6), so ``as_of`` alone cannot reveal that a connector silently stopped
    delivering — the EWMA simply decayed the unobserved tail as assumed-rest and form drifted up.
    So freshness is anchored on the most recent OBSERVED activity day, corroborated by
    ``sync_suspect`` (whether a connector that should be delivering is broken/stalled — the signal
    that keeps a legitimate taper's fresh-form ``go`` from being false-abstained), and any recent
    SUBSTITUTED (HR-modeled) load is flagged (DEGR-R2); the deliverable consumes that envelope. This
    bypasses the retrieval planner by design — the readiness JTBD is fixed.
    """
    today = utcnow().date()
    start = today - _dt.timedelta(days=_READINESS_WINDOW_DAYS)
    series = await svc.pmc(athlete_id, start, today)
    form, as_of = _latest_form(series, start)
    last_observed = await svc.latest_activity_date(athlete_id)
    sufficiency = assess_record_sufficiency(
        reference_date=today,
        last_observed_date=last_observed,
        fresh_within_days=READINESS_FRESH_STALENESS_DAYS,
        max_staleness_days=READINESS_MAX_STALENESS_DAYS,
        substituted=_recent_load_substituted(series),
        sync_suspect=sync_suspect,
    )
    if as_of is None:
        return ReadinessInputs(form, as_of, None, None, sufficiency)
    rmssd = await _latest_hrv_rmssd(svc, athlete_id, as_of)
    baseline = await svc.hrv_baseline(athlete_id, _dt.date.fromisoformat(as_of))
    return ReadinessInputs(form, as_of, rmssd, baseline, sufficiency)


def _recent_load_substituted(series: Sequence[Any]) -> bool:
    """True iff any computed day in the window used a SUBSTITUTED lower-fidelity load (DEGR-R2).

    A day whose load came from the HR-modeled member (the power source withdrawn) carries
    ``Fidelity.SUBSTITUTED`` load coverage. Surfacing it lets the deliverable disclose reduced
    fidelity rather than presenting a substituted-load form as full raw-stream truth.
    """
    for day in series:
        if not is_computed(day):
            continue
        coverage = getattr(day.value, "load_coverage", None)
        if coverage is not None and coverage.fidelity is Fidelity.SUBSTITUTED:
            return True
    return False


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


def localized_readiness_narrator(
    model: ChatModel, coach: Any, locale: str
) -> Callable[[str], Awaitable[_ReadinessNarration]]:
    """The readiness narrator whose system prompt carries the run locale's directive (issue #17).

    Composes the narrator's system prompt through the SAME any-language ``compose_system`` seam the
    free-form answer's compose node uses (graph_model_nodes.compose, LANG-R1/-R3): the run locale's
    config-templated directive — NOT an enumerated per-language pack — is layered onto the readiness
    persona, so the model narrates the readiness verdict IN the requested language (any language the
    model speaks; the LANG-R4 fallback is recorded). Grounding and the deterministic oracle verdict
    stay language-neutral (LANG-R3). ``coach`` is the engine's loaded coach bundle (its ``locales``
    policy + ``readiness_system`` persona).
    """
    system = coach.locales.compose_system(coach.readiness_system, locale)
    return readiness_narrator(model, system=system)


__all__ = [
    "ReadinessInputs",
    "connection_is_suspect",
    "connection_sync_suspect",
    "gather_readiness_inputs",
    "localized_readiness_narrator",
    "readiness_narrator",
]
