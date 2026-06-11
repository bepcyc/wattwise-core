"""Daily training-load resolution path for the analytics service (LOAD-R1/R3, QUAL-R9 split).

The service-side resolution that turns resolved canonical activities into per-day
training loads and their equivalence-class coverage: the LOAD-R3 fallback over the
``training_load`` members (power-TSS -> labeled HR load -> session-RPE load) and the
per-day DEGR-R2 substitution surfacing. Split out of
:mod:`wattwise_core.analytics.service` to honor the module size ceiling (QUAL-R9),
mirroring :mod:`wattwise_core.analytics._service_es`.
"""

from __future__ import annotations

import datetime as _dt
from collections import defaultdict
from typing import TYPE_CHECKING

from wattwise_core.analytics import load_substitution as _ls
from wattwise_core.analytics._service_loaders import _activity_local_date
from wattwise_core.analytics.result import is_computed
from wattwise_core.domain.coverage import Coverage
from wattwise_core.domain.enums import Fidelity

if TYPE_CHECKING:
    from wattwise_core.analytics.service import AnalyticsService


async def daily_loads_with_coverage(
    svc: AnalyticsService, athlete_id: str, from_date: _dt.date, to_date: _dt.date
) -> tuple[dict[_dt.date, float | None], dict[_dt.date, Coverage | None]]:
    """Resolve per-day load AND its equivalence-class coverage in one pass (LOAD-R1/DEGR-R2).

    Each resolved activity contributes its load once via the LOAD-R3 fallback; a day is
    flagged SUBSTITUTED iff ANY contributing activity's load came from a lower-fidelity
    member (DEGR-R2 — a partially substituted day is never presented at full fidelity).
    A no-activity day is a real ``0`` rest day; a computable-load-less day is a surfaced
    ``None``, never a silent zero (LOAD-R1).
    """
    activities = await svc._activities_in_range(athlete_id, from_date, to_date)
    by_day: dict[_dt.date, float] = defaultdict(float)
    substituted_day: set[_dt.date] = set()
    loaded_day: set[_dt.date] = set()
    seen: set[_dt.date] = set()
    if activities:
        # Any activity in range implies a real athlete row; load it for bucketing only then,
        # so an athlete with no data yields an all-rest series without needing a reference tz.
        athlete = await svc._athlete_or_fail(athlete_id)
        for act in activities:
            # GBO-R35: attribute the activity to its athlete-LOCAL calendar day (the
            # persisted local_date, recomputed from start_time + as-of tz when absent),
            # NOT the UTC date.
            day = _activity_local_date(act, athlete)
            seen.add(day)
            contribution = await activity_load(svc, str(act.activity_id))
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


async def activity_load(svc: AnalyticsService, activity_id: str) -> _ls.LoadContribution | None:
    """Per-activity load by the LOAD-R3 priority: power_tss -> hr_load -> srpe_load -> None.

    Resolves the ``training_load`` equivalence class (DM-SUB-R1) from the SINGLE
    per-activity bundle so the daily-load path and the activity-detail surface agree on
    the same load model (LOAD-R4): power-TSS (the ``raw_stream`` top member) else the
    labeled HR load (``hr_load`` / ``hr_load_zonal``, honouring the athlete default,
    resolved into the bundle per LOAD-R4) else the session-RPE load (the
    ``summary_only`` last-resort member, SRPE-R1 — the only member a power-less,
    HR-less strength session or swim can supply).

    The fallback chain is FIDELITY-ordered and keyed on the availability of the COMPUTED
    member, NOT on raw sensor presence (LOAD-R3): each member is tried only when the one
    above it is not ``Computed``. So session-RPE correctly wins as the last resort INCLUDING
    when a power channel is present but TSS is Unavailable (no FTP) and ``hr_load`` is
    Unavailable (no HR) — an incomplete higher member never blocks the chain, it only fails
    its own step. Every below-top winner — the HR load AND the session-RPE load, including
    sRPE-when-power-present — is carried as a SUBSTITUTED contribution (DEGR-R2) at reduced
    confidence, so a withdrawn/uncomputable higher source never presents a lower member's
    load as power-TSS; an empty class yields ``None`` (a surfaced unknown-load day, DEGR-R3).
    The bundle's / metric's ``load_model`` records which family produced it, so the load
    families are never silently mixed (TSS-R3 / LOAD-R4).
    """
    bundle = await svc.coggan(activity_id)
    if is_computed(bundle):
        tss = bundle.value.tss
        if is_computed(tss):
            return _ls.LoadContribution(float(tss.value), _ls.LOAD_TOP_COVERAGE)
        hr_load = bundle.value.hr_load
        if is_computed(hr_load):
            return _ls.LoadContribution(float(hr_load.value), _ls.LOAD_SUBSTITUTED_COVERAGE)
    srpe_res = await svc.srpe(activity_id)
    if is_computed(srpe_res):
        return _ls.LoadContribution(float(srpe_res.value), _ls.LOAD_SUBSTITUTED_COVERAGE)
    return None
