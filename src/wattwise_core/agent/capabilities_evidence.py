"""Canonical grounding evidence: verbatim metric reads + first-party URL allow-list.

The focused sibling of :mod:`wattwise_core.agent.capabilities` (QUAL-R9 size split) that owns
the read-only grounding-evidence surface the grounder verifies a claim against. It depends only
on LEAF modules (the :class:`~wattwise_core.agent.capabilities_metrics.MetricName` vocabulary and
the :class:`~wattwise_core.agent.metric_equivalence.MetricEquivalence` resolver) plus the canonical
:class:`~wattwise_core.analytics.service.AnalyticsService`, so it carries no dependency on the
capability registry / gather path and never forms an import cycle with them.

Cited requirements (doc 50): GROUND-R7 (numbers are read VERBATIM from the canonical analytics
service — never re-derived or rounded — and URLs are checked against a first-party exact-host
allow-list), GROUND-R2 (a natural metric label resolves through the §16 equivalence layer),
GROUND-R3 (an unrecognized/uncomputable claim fails closed to ``None`` so the grounder scrubs),
and GROUND-R4 (the URL allow-list is config-loaded https + exact-host policy).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from wattwise_core.agent.capabilities_metrics import MetricName
from wattwise_core.agent.metric_equivalence import MetricEquivalence
from wattwise_core.analytics.result import MetricResult, is_computed
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.persistence.types import utcnow


@dataclass(frozen=True, slots=True)
class _AsOf:
    """The resolution of a claim's ``as_of`` token into three cases (GROUND-R7, H2 fail-closed).

    ABSENT (no date token, ``date=None invalid=False``) -> latest-day fallback is correct (a
    natural dateless claim "your fitness is 6.7"); PARSED (``date`` set) -> read AT that day;
    INVALID (a date token that failed to parse, ``invalid=True``) -> FAIL CLOSED (metric value
    resolves to ``None`` so the grounder scrubs). Silently reading an INVALID/past-dated claim
    against the LATEST day would ground "on May 1 your fitness was 100" against today's CTL — the
    H2 fabrication. A date we cannot pin is never "latest".
    """

    date: _dt.date | None
    invalid: bool

    @property
    def absent(self) -> bool:
        """True iff the claim carried no date token (latest-day fallback is allowed)."""
        return self.date is None and not self.invalid


def _latest_pmc_scalar(series: list[MetricResult[Any]], field: str) -> float | None:
    """The named scalar of the latest computed PMC day, or ``None`` (fail-closed)."""
    for day in reversed(series):
        if is_computed(day):
            return float(getattr(day.value, field))
    return None


def _scalar_of(metric: MetricName, value: object) -> float | None:
    """Read the requested scalar from a Computed value object VERBATIM (GROUND-R7)."""
    attr = {
        MetricName.CRITICAL_POWER_W: "cp_w",
        MetricName.W_PRIME_J: "w_prime_j",
        MetricName.HRV_RMSSD_MS: "rmssd_ms",
    }[metric]
    return float(getattr(value, attr))


# Days each maintenance AGGREGATE load-target metric multiplies the canonical CTL by (§16
# aggregates): steady-state maintenance load = CTL per day, so 7 days for the weekly target and
# 28 days for the 4-week ("monthly") target. A closed, code-owned derivation table — the metric
# vocabulary is the closed MetricName enum, never a free string.
_AGGREGATE_TARGET_DAYS: Mapping[MetricName, float] = {
    MetricName.WEEKLY_LOAD_TARGET: 7.0,
    MetricName.MONTHLY_LOAD_TARGET: 28.0,
}

# Conservative fallback window (days back from the reference date) the latest-available-date
# scan uses when a caller injects NO config-loaded lookback (the prior exact-key/no-config
# behaviour). Production wires the §16 ``[agent.coach].latest_lookback_days`` (CFG-R1a) in via
# the CoachBundle; this is the no-config default, never a hidden policy hardcode in the gate.
_DEFAULT_LOOKBACK_DAYS = 42


class CanonicalEvidence:
    """Read-only canonical evidence for the grounder (GROUND-R7, contracts.GroundingEvidence).

    Implements :class:`~wattwise_core.agent.contracts.GroundingEvidence`. Numbers come
    VERBATIM from the canonical :class:`AnalyticsService` for the engine-scoped athlete —
    this layer NEVER re-derives or rounds a value, it only reads what the service computed.
    A metric the service cannot compute returns ``None`` (the grounder scrubs the claim);
    ``url_allowed`` is a first-party exact-host allow-list.

    A real model states a number in NATURAL terms with no as-of date (e.g. ``"your fitness
    is 6.7"``). Two config-loaded bridges (§16 / SKILL-R1) make such a CORRECT claim ground
    instead of scrub: (1) ``equivalence`` maps the natural metric label to its canonical key
    (GROUND-R2); (2) when the claim carries NO date token the value is read at the metric's LATEST
    available day (anchored at ``reference_date``). Both bridges only widen WHICH claim can be
    checked — the value is read VERBATIM and an unmatched value is still scrubbed (GROUND-R3).
    """

    def __init__(
        self,
        svc: AnalyticsService,
        athlete_id: str,
        *,
        equivalence: MetricEquivalence | None = None,
        reference_date: _dt.date | None = None,
        allowed_hosts: frozenset[str] | None = None,
        lookback_days: int | None = None,
    ) -> None:
        self._svc = svc
        self._athlete_id = athlete_id
        # An absent alias map degenerates to canonical-only resolution (no equivalence),
        # preserving the prior exact-key behaviour for any caller that injects none.
        self._equivalence = equivalence if equivalence is not None else MetricEquivalence({})
        self._reference_date = reference_date or utcnow().date()
        # The first-party URL allow-list + dateless-claim lookback are config-loaded policy
        # (GROUND-R4 / §16, CFG-R1a) wired in by the CoachBundle; an absent injection falls back
        # to the conservative no-config defaults rather than a hidden policy hardcode in code.
        self._allowed_hosts = allowed_hosts if allowed_hosts is not None else frozenset()
        self._lookback_days = lookback_days if lookback_days is not None else _DEFAULT_LOOKBACK_DAYS

    async def metric_value(self, metric: str, as_of: str | None) -> float | None:
        """The canonical value of ``metric`` as-of a date, or ``None`` (GROUND-R7).

        ``metric`` resolves through the metric-equivalence layer (§16) to a canonical
        :class:`MetricName` (unrecognized -> ``None`` -> scrub). ``as_of`` is resolved by
        :meth:`_resolve_as_of` into ABSENT (read the LATEST day), PARSED (read AT that day), or
        INVALID (a date token that failed to parse -> FAIL CLOSED ``None``, never silently
        resolved to latest — that is the H2 fabrication). An uncomputable result is ``None`` too.
        """
        canonical = self._equivalence.canonical_key(metric)
        if canonical is None:
            return None
        name = MetricName(canonical)
        if name is MetricName.ACTIVITY_TSS:
            # Per-ride TSS (#47): the ``as_of`` argument is the claim's ACTIVITY ID (carried in
            # ref), NOT a date. Branch BEFORE _resolve_as_of — which would parse a non-ISO activity
            # id as INVALID and return None — and resolve the single ride's TSS by activity id.
            return await self._activity_tss(as_of)
        resolved = self._resolve_as_of(as_of)
        if resolved.invalid:
            # The claim carried a date token we could not parse: do NOT fall back to latest.
            return None
        return await self._dated_metric(name, resolved)

    async def _dated_metric(self, name: MetricName, resolved: _AsOf) -> float | None:
        """Resolve a DATE-keyed canonical metric (PMC scalars, aggregate targets, CP/W', HRV)."""
        if name in (MetricName.CTL, MetricName.ATL, MetricName.TSB, MetricName.FORM):
            return await self._pmc_scalar(name, resolved)
        if name in _AGGREGATE_TARGET_DAYS:
            return await self._aggregate_load_target(name, resolved)
        if name in (MetricName.CRITICAL_POWER_W, MetricName.W_PRIME_J):
            return await self._cp_scalar(name, resolved)
        return await self._hrv_scalar(name, resolved)

    async def _aggregate_load_target(self, name: MetricName, as_of: _AsOf) -> float | None:
        """A week/4-week maintenance load target derived from canonical PMC CTL (§16 aggregates).

        Deterministic CODE over the canonical analytic, never a model number: holding CTL steady
        needs an average daily load equal to CTL, so the weekly target is 7xCTL and the 4-week
        ("monthly") target 28xCTL. This is what lets a month-horizon maintenance PLAN ground its
        aggregate load targets instead of scrubbing them (the live month-plan DEGRADED defect).
        An unavailable CTL stays ``None`` — the claim is scrubbed, never a placeholder
        (GROUND-R7 fail-closed; the derivation itself is an accepted deviation from GROUND-R7's
        strict verbatim reading, per the §16 aggregate-verification mandate).
        """
        ctl = await self._pmc_scalar(MetricName.CTL, as_of)
        if ctl is None:
            return None
        return ctl * _AGGREGATE_TARGET_DAYS[name]

    async def _pmc_scalar(self, name: MetricName, as_of: _AsOf) -> float | None:
        # FORM is the athlete-facing alias of TSB: both read the canonical PmcDay.tsb field.
        field = MetricName.TSB.value if name is MetricName.FORM else name.value
        if as_of.absent:
            # No date token on the claim: read the metric at the latest available PMC day.
            window = await self._svc.pmc(
                self._athlete_id,
                self._reference_date - _dt.timedelta(days=self._lookback_days),
                self._reference_date,
            )
            return _latest_pmc_scalar(window, field)
        # Not absent and not invalid (the caller already returned ``None`` for invalid) -> a parsed
        # date; the reference-date fallback is a defensive no-op (``date`` is non-None here).
        day = as_of.date or self._reference_date
        series = await self._svc.pmc(self._athlete_id, day, day)
        return _latest_pmc_scalar(series, field)

    async def _cp_scalar(self, name: MetricName, as_of: _AsOf) -> float | None:
        day = as_of.date or self._reference_date
        fit = await self._svc.critical_power(self._athlete_id, day, day)
        if not is_computed(fit):
            return None
        return _scalar_of(name, fit.value)

    async def _hrv_scalar(self, name: MetricName, as_of: _AsOf) -> float | None:
        day = as_of.date or self._reference_date
        result = await self._svc.hrv(self._athlete_id, day)
        if not is_computed(result):
            return None
        return _scalar_of(name, result.value)

    async def _activity_tss(self, activity_id: str | None) -> float | None:
        """The Coggan TSS of a SINGLE ride, resolved by activity id, fail-closed (#47, GROUND-R7).

        DOUBLE-unwrapped: ``svc.coggan`` returns ``MetricResult[LoadMetricsBundle]`` whose ``tss``
        is itself a ``MetricResult[float]`` (Computed on the cycling-power path, Unavailable on the
        HR path / a non-power sport, LM-R2). Every fail-closed edge resolves to ``None`` so the
        grounder scrubs the claimed number — never a placeholder, never an HR-load value relabeled
        as power TSS, never a per-day fabrication:
        * no activity id (a per-ride claim with no ref — per-day ambiguous) -> None;
        * unknown / ungathered activity (``coggan`` -> Unavailable) -> None;
        * HR-based / non-power-sport ride (``bundle.tss`` Unavailable) -> None.
        """
        if not activity_id:
            return None
        bundle = await self._svc.coggan(activity_id)
        if not is_computed(bundle):
            return None
        tss = bundle.value.tss
        if not is_computed(tss):
            return None
        return float(tss.value)

    @staticmethod
    def _resolve_as_of(as_of: str | None) -> _AsOf:
        """Resolve a claim's ``as_of`` token into ABSENT / PARSED / INVALID (GROUND-R7).

        ``None`` / all-whitespace -> ABSENT (no date token; latest-day fallback allowed). A token
        that parses as an ISO date -> PARSED. A token that does NOT parse -> INVALID (fail closed
        to ``None``), so a past-dated-but-unparseable claim is scrubbed, never grounded against the
        latest day (H2: "on May 1 your fitness was 100").
        """
        if as_of is None or not as_of.strip():
            return _AsOf(date=None, invalid=False)
        try:
            return _AsOf(date=_dt.date.fromisoformat(as_of.strip()), invalid=False)
        except ValueError:
            return _AsOf(date=None, invalid=True)

    def url_allowed(self, url: str) -> bool:
        """True iff ``url`` is https + an exact-host first-party allow-list link (GROUND-R4)."""
        parsed = urlparse(url)
        return parsed.scheme == "https" and parsed.hostname in self._allowed_hosts


__all__ = ["CanonicalEvidence"]
