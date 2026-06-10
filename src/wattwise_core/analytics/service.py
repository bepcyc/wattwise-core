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
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.analytics import decoupling as _dec
from wattwise_core.analytics import hrv as _hrv
from wattwise_core.analytics import load_resolution as _load_res
from wattwise_core.analytics import load_substitution as _ls
from wattwise_core.analytics import mmp_cp as _mmp
from wattwise_core.analytics import np_if_tss as _np
from wattwise_core.analytics import pmc as _pmc
from wattwise_core.analytics import trimp as _trimp
from wattwise_core.analytics import wbal as _wbal
from wattwise_core.analytics._service_loaders import (
    _activities_in_local_range,
    _activity_local_date,
    _channel_to_stream,
    _f,
    _fold_curve_point,
    _gather_endurance_score,
    _load_athlete,
    _load_athlete_or_fail,
    _load_athlete_sex,
    _load_earliest_activity_date,
    _load_threshold_history,
    _load_wellness_hrv_baseline,
    _load_wellness_hrv_summary,
    _load_wellness_rr,
    _uid,
)
from wattwise_core.analytics.result import (
    Computed,
    MetricResult,
    Unavailable,
    UnavailableReason,
    is_computed,
)
from wattwise_core.analytics.series import Stream, resample_to_1hz
from wattwise_core.domain.coverage import Coverage
from wattwise_core.domain.enums import Fidelity, StreamChannelName, StreamSetKind
from wattwise_core.persistence.models import (
    Activity,
    ActivityStreamSet,
    Athlete,
    FitnessSignature,
    StreamChannel,
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


class AnalyticsService:  # noqa: size-limits
    """Computes canonical analytics for one athlete from the canonical store.

    Intentionally ONE class slightly over the derived class-size guard: ARCH-R5/R23
    mandate a SINGLE canonical entry point per analytics capability shared by the API
    and the agent, so splitting this facade into sibling classes would create multiple
    entry points — the opposite of the spec invariant. Every method stays under the
    60-line function ceiling; stateless loaders and the substitution carriers
    (:mod:`wattwise_core.analytics.load_substitution`) are factored to module level.
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

    async def _hr_load_for_activity(
        self, athlete_id: str, sport: str, as_of: _dt.date, hr: Stream | None
    ) -> MetricResult[float] | None:
        """Resolve the HR-path load honouring the stored athlete default (LOAD-R4).

        Loads the canonical inputs — HR_max/HR_rest (signature keyed by the activity's
        sport), athlete sex, and the stored ``default_training_load_model`` preference —
        then delegates the LOAD-R4 selection to the pure
        :func:`~wattwise_core.analytics.load_resolution.resolve_hr_load`. Reading the
        preference here is what CONSUMES it (it was previously orphaned).
        """
        if hr is None:
            return None
        sex = await _load_athlete_sex(self._session, athlete_id)
        sig = await self.resolve_signature(athlete_id, sport, as_of)
        athlete = await self._session.get(Athlete, _uid(athlete_id))
        preferred = None if athlete is None else athlete.default_training_load_model
        return _load_res.resolve_hr_load(
            hr,
            sig.max_hr_bpm,
            sig.resting_hr_bpm,
            sex,
            preferred_load_model=preferred,
        )

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
        """Compute the per-activity load-metrics bundle for an activity (LM-R1/R2, LOAD-R4).

        The bundle's power family (NP/IF/TSS/EF/VI/intensity_class) is gated on the
        activity's canonical ``sport`` (ANL-R12): a non-cycling-power sport yields a
        bundle whose power fields are ``NOT_APPLICABLE_FOR_SPORT``, never a fabricated
        cycling number. The load field is ``tss`` on the power path OR the labeled
        HR-load on the HR path (LM-R1); when neither power nor usable HR load is
        computable the load field is ``Unavailable`` while ``duration_valid_s`` is still
        reported when a power channel exists (LM-R2). The HR-load value (and its
        ``hr_load`` vs ``hr_load_zonal`` label) is resolved per the athlete default
        (LOAD-R4) and passed into the assembler.
        """
        act = await self._activity(activity_id)
        if act is None:
            return Unavailable(_MISSING, "unknown activity")
        channels = await self._activity_channels(activity_id)
        # A power-less activity still yields a bundle (LM-R2): the load field carries the
        # HR-load (when computable) and duration_valid_s where known — not a bare absence.
        power = channels.get(StreamChannelName.POWER_W) or Stream.from_values([])
        hr = channels.get(StreamChannelName.HR_BPM)
        sig = await self.resolve_signature(str(act.athlete_id), act.sport, act.start_time.date())
        hr_load = await self._hr_load_for_activity(
            str(act.athlete_id), act.sport, act.start_time.date(), hr
        )
        bundle = _np.load_metrics_bundle(
            power,
            hr,
            sig.ftp_w,
            _f(act.avg_power_w),
            _f(act.avg_hr_bpm),
            sport=act.sport,
            hr_load_result=hr_load,
        )
        return Computed(value=bundle)

    async def w_balance(self, activity_id: str) -> MetricResult[_wbal.WBalResult]:
        """Compute the W' balance series for an activity (WBAL-R1, ANL-R12)."""
        act = await self._activity(activity_id)
        if act is None:
            return Unavailable(_MISSING, "unknown activity")
        channels = await self._activity_channels(activity_id)
        power = channels.get(StreamChannelName.POWER_W)
        if power is None:
            # Absent channel on a power-applicable sport is MISSING_REQUIRED_INPUT; an
            # inapplicable sport is gated as NOT_APPLICABLE_FOR_SPORT inside wbal().
            return _wbal.wbal(None, None, None, sport=act.sport)
        sig = await self.resolve_signature(str(act.athlete_id), act.sport, act.start_time.date())
        return _wbal.wbal(
            resample_to_1hz(power), sig.cp_w, sig.w_prime_j, sport=act.sport
        )

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

    async def _athlete_or_fail(self, athlete_id: str) -> Athlete:
        """The athlete whose reference tz drives every local-date bucket; fail closed (CFG-R6)."""
        return await _load_athlete_or_fail(self._session, athlete_id)

    async def _activities_in_range(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date
    ) -> list[Activity]:
        """Resolved activities whose LOCAL day (§3.8) falls in ``[from_date, to_date]`` (GBO-R35).

        The query window is the UTC instants spanning LOCAL midnight on ``from_date`` through
        LOCAL midnight on ``to_date + 1`` in the athlete's reference timezone — NOT a UTC-date
        window, which would drop an activity whose UTC date sits just outside the local span.
        A final precise re-bucket on ``local_date`` (in the loaders) drops any boundary
        activity that the instant window over-includes. An athlete with no row owns no
        activities, so the empty series needs no tz (the fail-closed CFG-R6 path applies only
        when a real athlete has activities but no resolvable reference tz, raised by the bucket).
        """
        athlete = await _load_athlete(self._session, athlete_id)
        if athlete is None:
            return []  # an athlete with no row owns no activities; the empty series needs no tz
        return await _activities_in_local_range(self._session, athlete, from_date, to_date)

    async def daily_load_series(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date
    ) -> dict[_dt.date, float | None]:
        """Per-day summed training load over resolved canonical activities (LOAD-R1).

        Reads RESOLVED canonical activities (DEDUP-R4 — never per-source rows); a no-activity
        day is a real ``0`` rest day, a computable-load-less day is a surfaced ``None`` (never
        a silent zero). See :meth:`_daily_loads_with_coverage` for the per-day coverage.
        """
        loads, _ = await self._daily_loads_with_coverage(athlete_id, from_date, to_date)
        return loads

    async def _daily_loads_with_coverage(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date
    ) -> tuple[dict[_dt.date, float | None], dict[_dt.date, Coverage | None]]:
        """Resolve per-day load AND its equivalence-class coverage in one pass (LOAD-R1/DEGR-R2).

        Each resolved activity contributes its load once via the LOAD-R3 fallback; a day is
        flagged SUBSTITUTED iff ANY contributing activity's load came from a lower-fidelity
        member (DEGR-R2 — a partially substituted day is never presented at full fidelity).
        """
        activities = await self._activities_in_range(athlete_id, from_date, to_date)
        by_day: dict[_dt.date, float] = defaultdict(float)
        substituted_day: set[_dt.date] = set()
        loaded_day: set[_dt.date] = set()
        seen: set[_dt.date] = set()
        if activities:
            # Any activity in range implies a real athlete row; load it for bucketing only then,
            # so an athlete with no data yields an all-rest series without needing a reference tz.
            athlete = await self._athlete_or_fail(athlete_id)
            for act in activities:
                # GBO-R35: attribute the activity to its athlete-LOCAL calendar day (the
                # persisted local_date, recomputed from start_time + as-of tz when absent),
                # NOT the UTC date.
                day = _activity_local_date(act, athlete)
                seen.add(day)
                contribution = await self._activity_load(str(act.activity_id))
                if contribution is not None:
                    by_day[day] += contribution.value
                    loaded_day.add(day)
                    if contribution.coverage.fidelity is Fidelity.SUBSTITUTED:
                        substituted_day.add(day)
        loads: dict[_dt.date, float | None] = {}
        coverage: dict[_dt.date, Coverage | None] = {}
        cur = from_date
        while cur <= to_date:
            # A day with no activity is a real 0 rest day; a day that had an activity
            # but no computable load is a surfaced None, never a silent zero (LOAD-R1).
            loads[cur] = by_day.get(cur, None if cur in seen else 0.0)
            coverage[cur] = _ls.day_load_coverage(
                has_load=cur in loaded_day, substituted=cur in substituted_day
            )
            cur += _dt.timedelta(days=1)
        return loads, coverage

    async def _activity_load(self, activity_id: str) -> _ls.LoadContribution | None:
        """Per-activity training load by the LOAD-R3 priority: power_tss -> hr_load -> None.

        Resolves the ``training_load`` equivalence class (DM-SUB-R1) from the SINGLE
        per-activity bundle so the daily-load path and the activity-detail surface agree on
        the same load model (LOAD-R4): power-TSS (the ``raw_stream`` top member) else the
        labeled HR load (``hr_load`` / ``hr_load_zonal``, honouring the athlete default,
        resolved into the bundle per LOAD-R4) carried as a SUBSTITUTED contribution
        (DEGR-R2) — so a withdrawn power source never presents an HR load as power-TSS —
        else ``None`` (empty class: a surfaced unknown-load day, DEGR-R3). The bundle's
        ``load_model`` records which family produced it, so the two load families are never
        silently mixed (TSS-R3 / LOAD-R4).
        """
        bundle = await self.coggan(activity_id)
        if not is_computed(bundle):
            return None
        tss = bundle.value.tss
        if is_computed(tss):
            return _ls.LoadContribution(float(tss.value), _ls.LOAD_TOP_COVERAGE)
        hr_load = bundle.value.hr_load
        if is_computed(hr_load):
            return _ls.LoadContribution(float(hr_load.value), _ls.LOAD_SUBSTITUTED_COVERAGE)
        return None

    async def _earliest_activity_date(self, athlete_id: str) -> _dt.date | None:
        """The athlete-LOCAL date of the first-ever activity, or ``None`` if none (GBO-R35).

        The earliest activity by UTC ``start_time`` is also the earliest local instant, but its
        LOCAL calendar day (the PMC origin) is the projection of that instant — not its UTC
        date — so the EWMA grid starts on the correct local day (PMC-R3/R5).
        """
        return await _load_earliest_activity_date(self._session, athlete_id)

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
        loads, day_cov = await self._daily_loads_with_coverage(athlete_id, origin, to_date)
        ordered = sorted(loads)
        series = [loads[d] for d in ordered]
        # Thread per-day load coverage: a day fed by a substituted member carries
        # SUBSTITUTED + reduced confidence (DEGR-R2), never presented as raw power-TSS.
        full = _pmc.pmc(series, seed=seed, day_load_coverage=[day_cov[d] for d in ordered])
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
            aid, day = str(act.activity_id), act.start_time.date()
            _fold_curve_point(best, power, activity_id=aid, local_date=day, sport=sport)
        return best

    async def critical_power(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date, *, sport: str = "cycling"
    ) -> MetricResult[_mmp.CPFit]:
        """Fit CP/W' from the sport-partitioned aggregate power curve (CP-R1, ANL-R13)."""
        curve = await self.power_curve(athlete_id, from_date, to_date, sport=sport)
        points = {d: res.value.mean_power_w for d, res in curve.items() if is_computed(res)}
        return _mmp.cp_wprime(points, sport=sport)

    async def endurance_score(self, athlete_id: str, as_of: _dt.date) -> MetricResult[float]:
        # Composed of upstream capability results only (CTL / durability / decoupling):
        # gather lives in ._service_loaders, numeric truth in .endurance_score (ES-R2).
        """Composed ``[0,100]`` endurance score as-of a local date (ES-R1/R2/R3)."""
        return await _gather_endurance_score(self, athlete_id, as_of)

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

    async def threshold_history(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date
    ) -> list[FitnessSignature]:
        """The effective-dated ``fitness_signature`` history in range (doc 20 §3.6).

        A canonical READ of the versioned threshold rows (GBO-R26), NOT a doc-40
        computed metric — the API-R30 ``threshold-history`` exception (backed by
        doc 20). Chronological; ``[]`` when no signature exists in range (an empty
        page, never a fabricated zero). The query lives in :mod:`._service_loaders`.
        """
        return await _load_threshold_history(self._session, athlete_id, from_date, to_date)


__all__ = ["ENGINE_VERSION", "AnalyticsService", "SignatureParams"]
