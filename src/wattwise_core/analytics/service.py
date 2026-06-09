"""Canonical analytics service — the single consumer surface (B-E3-T6, doc 40).

The agent (its `gather`/MCP tools) and the REST API are PEERS over this service
(doc 50 §1): both read computed analytics ONLY through here, and this is the only
place that bridges the canonical ORM store (tier 3) to the pure metric functions.
It reads exclusively named, typed canonical fields and stream channels (ANL-R1/R1a),
resolves athlete reference params from effective-dated ``fitness_signature`` (ANL-R9),
and returns the same typed :data:`MetricResult` envelopes the metric functions
produce — fail-closed, never a fabricated number.

The service is thin: all numeric truth lives in the pure functions; this layer only
loads canonical inputs (via the module-level loaders below), calls them, and threads
provenance.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.analytics import decoupling as _dec
from wattwise_core.analytics import hrv as _hrv
from wattwise_core.analytics import mmp_cp as _mmp
from wattwise_core.analytics import np_if_tss as _np
from wattwise_core.analytics import pmc as _pmc
from wattwise_core.analytics import trimp as _trimp
from wattwise_core.analytics import wbal as _wbal
from wattwise_core.analytics.result import (
    Computed,
    MetricResult,
    Unavailable,
    UnavailableReason,
    is_computed,
)
from wattwise_core.analytics.series import Stream, resample_to_1hz
from wattwise_core.domain.enums import StreamChannelName, StreamSetKind
from wattwise_core.persistence.models import (
    Activity,
    ActivityStreamSet,
    Athlete,
    DailyWellness,
    FitnessSignature,
    StreamChannel,
    WellnessStreamSet,
)

ENGINE_VERSION = "analytics-1"

_MISSING = UnavailableReason.MISSING_REQUIRED_INPUT


@dataclass(frozen=True, slots=True)
class SignatureParams:
    """Effective athlete reference params resolved as-of an activity date (ANL-R9)."""

    ftp_w: float | None = None
    cp_w: float | None = None
    w_prime_j: float | None = None
    max_hr_bpm: float | None = None
    resting_hr_bpm: float | None = None


# --- module-level helpers (stateless loaders + coercions) ---


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


def _day_bounds(from_date: _dt.date, to_date: _dt.date) -> tuple[_dt.datetime, _dt.datetime]:
    """The half-open UTC instant range [from 00:00, to+1 00:00) for a local-date span."""
    lo = _dt.datetime.combine(from_date, _dt.time.min, _dt.UTC)
    hi = _dt.datetime.combine(to_date + _dt.timedelta(days=1), _dt.time.min, _dt.UTC)
    return lo, hi


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


class AnalyticsService:  # noqa: size-limits
    """Computes canonical analytics for one athlete from the canonical store.

    Intentionally ONE class slightly over the derived class-size guard: ARCH-R5/R23
    mandate a SINGLE canonical entry point per analytics capability shared by the API
    and the agent, so splitting this facade into sibling classes would create multiple
    entry points — the opposite of the spec invariant. The module stays well under the
    400-line module ceiling and every method under the 60-line function ceiling; the
    stateless loaders are already factored to module level.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # --- canonical input loading ---

    async def _activity(self, activity_id: str) -> Activity | None:
        return await self._session.get(Activity, _uid(activity_id))

    async def _activity_channels(self, activity_id: str) -> dict[StreamChannelName, Stream]:
        """Load the activity's stream channels as analytic streams, keyed by channel."""
        stmt = select(ActivityStreamSet).where(
            ActivityStreamSet.activity_id == _uid(activity_id)
        )
        stream_set = (await self._session.execute(stmt)).scalar_one_or_none()
        if stream_set is None:
            return {}
        cstmt = select(StreamChannel).where(
            StreamChannel.stream_set_id == stream_set.stream_set_id,
            StreamChannel.set_kind == StreamSetKind.ACTIVITY,
        )
        channels = (await self._session.execute(cstmt)).scalars().all()
        rate = float(stream_set.sample_rate_hz) if stream_set.sample_rate_hz else 1.0
        return {c.channel: _channel_to_stream(c, rate) for c in channels}

    async def current_sport(self, athlete_id: str) -> str | None:
        """The athlete's current primary sport code from the canonical profile, else ``None``.

        Resolved from the ``Athlete`` row (GBO-R13b) — a HINT the signature/coverage probes key on,
        never hardcoded (CFG-R1a). A missing athlete or an unset ``current_sport`` returns ``None``
        so a caller fails closed (e.g. reports the sport-keyed signature MISSING) rather than
        guessing a sport.
        """
        athlete = await self._session.get(Athlete, _uid(athlete_id))
        return None if athlete is None else athlete.current_sport

    async def resolve_signature(
        self, athlete_id: str, signature_type: str, as_of: _dt.date
    ) -> SignatureParams:
        """Resolve the effective signature params as-of ``as_of`` (ANL-R9, GBO-R26/R27)."""
        stmt = (
            select(FitnessSignature)
            .where(
                FitnessSignature.athlete_id == _uid(athlete_id),
                FitnessSignature.signature_type == signature_type,
                FitnessSignature.effective_date <= as_of,
            )
            .order_by(FitnessSignature.effective_date.desc())
            .limit(1)
        )
        sig = (await self._session.execute(stmt)).scalar_one_or_none()
        if sig is None:
            return SignatureParams()
        return SignatureParams(
            ftp_w=_f(sig.ftp_w),
            cp_w=_f(sig.cp_w),
            w_prime_j=_f(sig.w_prime_j),
            max_hr_bpm=_f(sig.max_hr_bpm),
            resting_hr_bpm=_f(sig.resting_hr_bpm),
        )

    # --- per-activity metrics ---

    async def coggan(self, activity_id: str) -> MetricResult[_np.LoadMetricsBundle]:
        """Compute the NP/IF/TSS load-metrics bundle for an activity (LM-R1)."""
        act = await self._activity(activity_id)
        if act is None:
            return Unavailable(_MISSING, "unknown activity")
        channels = await self._activity_channels(activity_id)
        power = channels.get(StreamChannelName.POWER_W)
        if power is None:
            return Unavailable(_MISSING, "no power channel")
        hr = channels.get(StreamChannelName.HR_BPM)
        sig = await self.resolve_signature(str(act.athlete_id), act.sport, act.start_time.date())
        bundle = _np.load_metrics_bundle(
            power, hr, sig.ftp_w, _f(act.avg_power_w), _f(act.avg_hr_bpm)
        )
        return Computed(value=bundle)

    async def w_balance(self, activity_id: str) -> MetricResult[_wbal.WBalResult]:
        """Compute the W' balance series for an activity (WBAL-R1)."""
        act = await self._activity(activity_id)
        if act is None:
            return Unavailable(_MISSING, "unknown activity")
        channels = await self._activity_channels(activity_id)
        power = channels.get(StreamChannelName.POWER_W)
        if power is None:
            return Unavailable(_MISSING, "no power channel")
        sig = await self.resolve_signature(str(act.athlete_id), act.sport, act.start_time.date())
        return _wbal.wbal(resample_to_1hz(power), sig.cp_w, sig.w_prime_j)

    async def aerobic_decoupling(self, activity_id: str) -> MetricResult[float]:
        """Compute aerobic decoupling for an activity (DEC-R1)."""
        act = await self._activity(activity_id)
        if act is None:
            return Unavailable(_MISSING, "unknown activity")
        channels = await self._activity_channels(activity_id)
        hr = channels.get(StreamChannelName.HR_BPM)
        output = channels.get(StreamChannelName.POWER_W) or channels.get(
            StreamChannelName.SPEED_MPS
        )
        if hr is None or output is None:
            return Unavailable(_MISSING, "needs output + HR")
        return _dec.aerobic_decoupling(output, hr, act.sport)

    async def trimp(self, activity_id: str) -> MetricResult[float]:
        """Compute Banister-HRR HR load for an activity (TRIMP-R1)."""
        act = await self._activity(activity_id)
        if act is None:
            return Unavailable(_MISSING, "unknown activity")
        channels = await self._activity_channels(activity_id)
        hr = channels.get(StreamChannelName.HR_BPM)
        sig = await self.resolve_signature(str(act.athlete_id), act.sport, act.start_time.date())
        sex = await _load_athlete_sex(self._session, str(act.athlete_id))
        return _trimp.banister_hr_load(hr, sig.max_hr_bpm, sig.resting_hr_bpm, sex)

    # --- athlete-level metrics ---

    async def _activities_in_range(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date
    ) -> list[Activity]:
        lo, hi = _day_bounds(from_date, to_date)
        stmt = select(Activity).where(
            Activity.athlete_id == _uid(athlete_id),
            Activity.start_time >= lo,
            Activity.start_time < hi,
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def daily_load_series(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date
    ) -> dict[_dt.date, float | None]:
        """Per-day summed training load over resolved canonical activities (LOAD-R1).

        Reads RESOLVED canonical activities (DEDUP-R4 — never per-source rows); each
        contributes its computed power-TSS once. A day with no activity is a real ``0``
        rest day; a day whose only activity has no computable load contributes ``None``
        (surfaced), never a silent zero.
        """
        activities = await self._activities_in_range(athlete_id, from_date, to_date)
        by_day: dict[_dt.date, float] = defaultdict(float)
        seen: set[_dt.date] = set()
        for act in activities:
            day = act.start_time.date()
            seen.add(day)
            load = await self._activity_load(str(act.activity_id))
            if load is not None:
                by_day[day] += load
        out: dict[_dt.date, float | None] = {}
        cur = from_date
        while cur <= to_date:
            # A day with no activity is a real 0 rest day; a day that had an activity
            # but no computable load is a surfaced None, never a silent zero (LOAD-R1).
            out[cur] = by_day.get(cur, None if cur in seen else 0.0)
            cur += _dt.timedelta(days=1)
        return out

    async def _activity_load(self, activity_id: str) -> float | None:
        """Per-activity training load by the LOAD-R3 priority: power_tss -> hr_load -> None.

        Power TSS when a mechanical-power channel + effective FTP are present; otherwise
        the Banister HR load (TRIMP-R1) when HR + HR_max/HR_rest are present; otherwise
        the activity contributes nothing (a surfaced unknown-load day, never a silent 0).
        """
        bundle = await self.coggan(activity_id)
        if is_computed(bundle) and is_computed(bundle.value.tss):
            return float(bundle.value.tss.value)
        hr_load = await self.trimp(activity_id)
        return float(hr_load.value) if is_computed(hr_load) else None

    async def _earliest_activity_date(self, athlete_id: str) -> _dt.date | None:
        """The local-date of the athlete's first-ever activity, or None if none."""
        stmt = select(func.min(Activity.start_time)).where(
            Activity.athlete_id == _uid(athlete_id)
        )
        first = (await self._session.execute(stmt)).scalar_one_or_none()
        return None if first is None else first.date()

    async def pmc(
        self,
        athlete_id: str,
        from_date: _dt.date,
        to_date: _dt.date,
        *,
        seed: tuple[float, float] | None = None,
    ) -> list[MetricResult[_pmc.PmcDay]]:
        """Compute the PMC (CTL/ATL/TSB) series over the requested window (PMC-R1/R3/R5).

        The EWMA is seeded from the athlete's true training origin, not from
        ``from_date``: a mid-history window seeded at zero would report CTL/ATL near
        zero regardless of real history — the wrong-but-plausible number PMC-R5 forbids.
        So the daily-load grid is built from the athlete's first activity (or
        ``from_date`` when no earlier history exists, or an explicit ``seed``) through
        ``to_date``, the full series is integrated, and the requested ``[from_date,
        to_date]`` slice is returned with its correctly carried-forward state (PMC-R3).
        """
        origin = from_date if seed is not None else (await self._earliest_activity_date(athlete_id))
        if origin is None or origin > from_date:
            origin = from_date
        loads = await self.daily_load_series(athlete_id, origin, to_date)
        series = [loads[d] for d in sorted(loads)]
        full = _pmc.pmc(series, seed=seed)
        start_index = (from_date - origin).days
        return full[start_index:]

    async def power_curve(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date, *, sport: str = "cycling"
    ) -> dict[int, MetricResult[_mmp.MMPWindow]]:
        """Aggregate mean-maximal-power curve for ONE sport over the date range (MMP-R4).

        Sport-partitioned (ANL-R13): only activities of ``sport`` contribute, and the
        resolved ``sport`` is threaded into the metric so lineage never mislabels a
        non-cycling power effort as cycling or pools incommensurable sports into one curve.
        """
        activities = await self._activities_in_range(athlete_id, from_date, to_date)
        best: dict[int, MetricResult[_mmp.MMPWindow]] = {}
        for act in activities:
            if act.sport != sport:
                continue
            power = (await self._activity_channels(str(act.activity_id))).get(
                StreamChannelName.POWER_W
            )
            if power is None:
                continue
            for d, res in _mmp.mmp(resample_to_1hz(power), sport=sport).items():
                if is_computed(res) and (d not in best or _better_mmp(res, best[d])):
                    best[d] = res
        return best

    async def critical_power(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date, *, sport: str = "cycling"
    ) -> MetricResult[_mmp.CPFit]:
        """Fit CP/W' from the sport-partitioned aggregate power curve (CP-R1, ANL-R13)."""
        curve = await self.power_curve(athlete_id, from_date, to_date, sport=sport)
        points = {d: res.value.mean_power_w for d, res in curve.items() if is_computed(res)}
        return _mmp.cp_wprime(points, sport=sport)

    async def hrv(
        self, athlete_id: str, local_date: _dt.date
    ) -> MetricResult[_hrv.TimeDomainHrv]:
        """Compute time-domain HRV for a wellness day (HRV-R0/R3, fail-closed)."""
        rr = await _load_wellness_rr(self._session, athlete_id, local_date)
        summary = await _load_wellness_hrv_summary(self._session, athlete_id, local_date)
        return _hrv.time_domain_hrv(rr_intervals_ms=rr, summary_rmssd_ms=summary)

    async def hrv_baseline(self, athlete_id: str, local_date: _dt.date) -> float | None:
        """The athlete's source-reported HRV baseline (RMSSD ms) for a day, or ``None``.

        Reads the ``hrv_baseline_low_ms`` / ``hrv_baseline_high_ms`` band on the day's
        :class:`DailyWellness` row and returns the MIDPOINT when both bounds are present (else
        whichever single bound is present, else ``None``). No row / both bounds absent /
        non-finite all fail closed to ``None``. This feeds the readiness HRV-suppression nudge
        (COACH-R1 #2): the nudge can only fire when a real baseline is available, so without
        one the verdict is read from form alone — never against a fabricated baseline.
        """
        return await _load_wellness_hrv_baseline(self._session, athlete_id, local_date)


def _better_mmp(
    candidate: Computed[_mmp.MMPWindow], current: MetricResult[_mmp.MMPWindow]
) -> bool:
    """True if ``candidate`` is a higher mean power than the current best (MMP-R4)."""
    if not is_computed(current):
        return True
    return candidate.value.mean_power_w > current.value.mean_power_w


__all__ = ["ENGINE_VERSION", "AnalyticsService", "SignatureParams"]
