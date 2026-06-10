"""Stateless canonical-store loaders + coercions for the analytics service (doc 40).

Factored out of :mod:`wattwise_core.analytics.service` so the single canonical service
facade (:class:`~wattwise_core.analytics.service.AnalyticsService`, the ONE entry point per
ARCH-R5/R23) stays within the QUAL-R9 module-size ceiling WITHOUT splitting the facade into
sibling classes (which would create multiple entry points). These are the source-agnostic,
side-effect-light helpers the service composes: scalar/JSON coercions at the query boundary,
the `Stream` builder from a canonical channel (ANL-R7), the day-bounds window, and the small
async canonical-store loaders (athlete sex, wellness RR / HRV summary / HRV baseline). They
read ONLY named, typed canonical fields and channels (ANL-R1/R1a) and fail closed to ``None``
— never a fabricated value.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.analytics import mmp_cp as _mmp
from wattwise_core.analytics.result import Computed, MetricResult, is_computed
from wattwise_core.analytics.series import Stream
from wattwise_core.domain.enums import StreamChannelName
from wattwise_core.persistence.localdate import (
    MissingReferenceTimezone,
    project_local_date,
)
from wattwise_core.persistence.models import (
    Activity,
    Athlete,
    DailyWellness,
    StreamChannel,
    WellnessStreamSet,
)


def _uid(value: str | uuid.UUID) -> uuid.UUID:
    """Coerce a string id to a UUID at the query boundary (portable Uuid binds UUIDs)."""
    return value if isinstance(value, uuid.UUID) else uuid.UUID(value)


def _f(value: object) -> float | None:
    """Coerce a nullable numeric column to ``float | None`` for the pure functions."""
    return None if value is None else float(value)  # type: ignore[arg-type]


def _num_or_nan(value: object) -> float:
    """Coerce a JSON sample to ``float``; non-numeric (incl. ``None``) becomes a gap."""
    return float(value) if isinstance(value, int | float) else float("nan")


def _channel_to_stream(channel: StreamChannel, sample_rate_hz: float | None) -> Stream:
    """Build an analytic :class:`Stream` from a canonical channel (ANL-R7).

    ``None``/non-numeric samples become ``NaN`` gaps; the time axis is derived from
    the channel's nominal sample rate (default 1 Hz). Metric functions resample.
    """
    rate = sample_rate_hz if sample_rate_hz and sample_rate_hz > 0 else 1.0
    vals = np.array([_num_or_nan(v) for v in channel.values], dtype=np.float64)
    t = np.arange(vals.size, dtype=np.float64) / rate
    return Stream(t_seconds=t, values=vals)


def _day_bounds_for_tz(
    from_date: _dt.date, to_date: _dt.date, tz_name: str
) -> tuple[_dt.datetime, _dt.datetime]:
    """The half-open UTC instant range covering a LOCAL-date span in ``tz_name`` (GBO-R35).

    The bucketing rule (§3.8) attributes an activity to the calendar date of its
    ``start_time`` in the athlete's reference timezone, so a window for the local span
    ``[from_date, to_date]`` is the UTC instants between LOCAL midnight on ``from_date`` and
    LOCAL midnight on ``to_date + 1`` (DST-aware via ``zoneinfo``). A UTC-midnight window
    would wrongly drop an activity whose UTC date falls just outside the span while its LOCAL
    date is inside it. A blank/unresolvable ``tz_name`` fails closed with
    :class:`MissingReferenceTimezone` (CFG-R6) — never a silent UTC window.
    """
    if not tz_name.strip():
        raise MissingReferenceTimezone("athlete has no reference timezone for day-bucketing")
    try:
        zone = ZoneInfo(tz_name.strip())
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise MissingReferenceTimezone(f"unresolvable reference timezone: {tz_name!r}") from exc
    lo = _dt.datetime.combine(from_date, _dt.time.min, zone).astimezone(_dt.UTC)
    hi = _dt.datetime.combine(
        to_date + _dt.timedelta(days=1), _dt.time.min, zone
    ).astimezone(_dt.UTC)
    return lo, hi


async def _load_athlete(session: AsyncSession, athlete_id: str) -> Athlete | None:
    """Load the athlete master row (its reference tz drives every local-date bucket)."""
    return await session.get(Athlete, _uid(athlete_id))


async def _load_athlete_or_fail(session: AsyncSession, athlete_id: str) -> Athlete:
    """Load the athlete whose reference tz drives every local-date bucket, or fail closed.

    The reference timezone is the authoritative source of ``local_date`` (§3.8, GBO-R33); with
    no athlete row there is no tz, so the day-bucketing layer fails closed (CFG-R6) rather than
    guessing a UTC default — surfaced as MISSING_REQUIRED_INPUT by callers wrapping a metric,
    and as a refusal on the bare day-series path.
    """
    athlete = await _load_athlete(session, athlete_id)
    if athlete is None:
        raise MissingReferenceTimezone(f"no athlete row for {athlete_id}")
    return athlete


def _activity_local_date(activity: Activity, athlete: Athlete) -> _dt.date:
    """The activity's reproducible LOCAL day bucket (GBO-R35), recomputed when unstored.

    Prefers the persisted ``activity.local_date`` (assigned at ingest, the GBO-R34
    reproducible bucket). When absent — a row ingested before this projection existed — it is
    recomputed from the UTC ``start_time`` plus the athlete's effective-dated reference tz
    (GBO-R34: "recomputable purely from its UTC instant plus the as-of reference-timezone
    metadata"), passing the stored value as the as-of prior so a relocation never re-buckets.
    Fails closed (no tz) via :class:`localdate.MissingReferenceTimezone`, never a UTC default.
    """
    return project_local_date(
        activity.start_time, athlete, prior_local_date=activity.local_date
    )


async def _activities_in_local_range(
    session: AsyncSession, athlete: Athlete, from_date: _dt.date, to_date: _dt.date
) -> list[Activity]:
    """Resolved activities whose LOCAL day (§3.8) falls in ``[from_date, to_date]`` (GBO-R35).

    Prefilters on the local-day instant window padded ±1 day so a tz offset / DST / a prior-era
    persisted ``local_date`` (relocated athlete) can never push an in-range LOCAL day's instant
    outside the SQL bounds; the exact ``local_date`` filter then decides (DST-safe; the padded
    window is a superset). Fails closed on a blank/unresolvable reference tz (CFG-R6) inside
    :func:`_day_bounds_for_tz`.
    """
    lo, hi = _day_bounds_for_tz(from_date, to_date, athlete.reference_timezone)
    pad = _dt.timedelta(days=1)
    stmt = select(Activity).where(
        Activity.athlete_id == athlete.athlete_id,
        Activity.start_time >= lo - pad,
        Activity.start_time < hi + pad,
    )
    rows = list((await session.execute(stmt)).scalars().all())
    return [a for a in rows if from_date <= _activity_local_date(a, athlete) <= to_date]


async def _load_athlete_sex(session: AsyncSession, athlete_id: str) -> str | None:
    athlete = await session.get(Athlete, _uid(athlete_id))
    return None if athlete is None else str(athlete.sex)


async def _load_wellness_rr(
    session: AsyncSession, athlete_id: str, local_date: _dt.date
) -> list[float] | None:
    stmt = select(WellnessStreamSet).where(
        WellnessStreamSet.athlete_id == _uid(athlete_id),
        WellnessStreamSet.local_date == local_date,
    )
    for s in (await session.execute(stmt)).scalars().all():
        cstmt = select(StreamChannel).where(
            StreamChannel.stream_set_id == s.wellness_stream_set_id,
            StreamChannel.channel == StreamChannelName.RR_INTERVALS_MS,
        )
        ch = (await session.execute(cstmt)).scalar_one_or_none()
        if ch is not None:
            return [_num_or_nan(v) for v in ch.values if v is not None]
    return None


async def _load_wellness_hrv_summary(
    session: AsyncSession, athlete_id: str, local_date: _dt.date
) -> float | None:
    stmt = select(DailyWellness).where(
        DailyWellness.athlete_id == _uid(athlete_id),
        DailyWellness.local_date == local_date,
    )
    dw = (await session.execute(stmt)).scalar_one_or_none()
    return None if dw is None else _f(dw.hrv_rmssd_ms)


async def _load_wellness_hrv_baseline(
    session: AsyncSession, athlete_id: str, local_date: _dt.date
) -> float | None:
    """The athlete's HRV baseline (RMSSD ms) for a day, or ``None`` (fail-closed).

    Reads the source-reported ``hrv_baseline_low_ms`` / ``hrv_baseline_high_ms`` band on the
    day's :class:`DailyWellness` row and returns the MIDPOINT when both bounds are present
    (else whichever single bound is present, else ``None``). No row, both bounds absent, or a
    non-finite value all fail closed to ``None`` (GBO-R24c band → one comparable baseline).
    """
    stmt = select(DailyWellness).where(
        DailyWellness.athlete_id == _uid(athlete_id),
        DailyWellness.local_date == local_date,
    )
    dw = (await session.execute(stmt)).scalar_one_or_none()
    if dw is None:
        return None
    return _hrv_baseline_midpoint(_f(dw.hrv_baseline_low_ms), _f(dw.hrv_baseline_high_ms))


def _hrv_baseline_midpoint(low: float | None, high: float | None) -> float | None:
    """Midpoint of a low/high HRV-baseline band, a single present bound, or ``None``.

    Non-finite bounds are dropped before combining so no NaN/Inf escapes (fail-closed).
    """
    bounds = [b for b in (low, high) if b is not None and np.isfinite(b)]
    if not bounds:
        return None
    return sum(bounds) / len(bounds)


def _better_mmp(
    candidate: Computed[_mmp.MMPWindow], current: MetricResult[_mmp.MMPWindow]
) -> bool:
    """True if ``candidate`` is a higher mean power than the current best (MMP-R4)."""
    if not is_computed(current):
        return True
    return candidate.value.mean_power_w > current.value.mean_power_w


__all__ = [
    "_activities_in_local_range",
    "_activity_local_date",
    "_better_mmp",
    "_channel_to_stream",
    "_day_bounds_for_tz",
    "_f",
    "_hrv_baseline_midpoint",
    "_load_athlete",
    "_load_athlete_or_fail",
    "_load_athlete_sex",
    "_load_wellness_hrv_baseline",
    "_load_wellness_hrv_summary",
    "_load_wellness_rr",
    "_num_or_nan",
    "_uid",
]
