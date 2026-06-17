"""Unit tests for the capability registry, deterministic gather, and canonical evidence.

Cited requirements (doc 50): PLAN-R2 (typed params, never source/table names), PLAN-R3
(one capability == one canonical-service method), PLAN-R5 (athlete scope is the gather
ARGUMENT, never a value inside the request), TOOL-R1 (thin 1:1 wrapper), TOOL-R5 (an
unavailable canonical computation records a typed GAP, never fabricated success), and
GROUND-R7 (the grounder reads metric values VERBATIM from the canonical service and checks
URLs against a first-party allow-list).

These are offline: a seeded in-memory analytics service (``FakeAnalyticsService``) returns
the SAME typed :class:`MetricResult` envelopes the real
:class:`~wattwise_core.analytics.service.AnalyticsService` produces, so the registry and
gather are exercised against canonical-shaped results without a database.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pytest
from pydantic import BaseModel

from wattwise_core.agent.capabilities import (
    CAPABILITIES,
    CAPABILITY_BY_KEY,
    ActivityParams,
    CanonicalEvidence,
    DateRangeParams,
    MetricEquivalence,
    MetricName,
    WellnessDayParams,
    gather,
)
from wattwise_core.agent.contracts import Capability, GroundingEvidence, RetrievalRequest
from wattwise_core.analytics.cp import CPFit
from wattwise_core.analytics.hrv import TimeDomainHrv
from wattwise_core.analytics.pmc import PmcDay
from wattwise_core.analytics.result import (
    Computed,
    MetricResult,
    Unavailable,
    UnavailableReason,
)

pytestmark = pytest.mark.unit

_ATHLETE = "11111111-1111-1111-1111-111111111111"
_OTHER_ATHLETE = "22222222-2222-2222-2222-222222222222"
_DAY = _dt.date(2026, 6, 1)


# --------------------------------------------------------------------------- #
# Seeded in-memory analytics service (canonical-shaped, network-free)          #
# --------------------------------------------------------------------------- #


class FakeAnalyticsService:
    """A seeded stand-in for :class:`AnalyticsService` returning canonical envelopes.

    Records the ``athlete_id`` each method is called with (to assert PLAN-R5 scoping) and
    returns scripted :class:`MetricResult` values shaped exactly like the real service's.
    """

    def __init__(self, *, pmc_ctl: float = 50.0, missing: bool = False) -> None:
        self._pmc_ctl = pmc_ctl
        self._missing = missing
        self.athlete_calls: list[str] = []
        self.activity_calls: list[str] = []

    async def pmc(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date, *, seed: Any = None
    ) -> list[MetricResult[PmcDay]]:
        self.athlete_calls.append(athlete_id)
        if self._missing:
            return [Unavailable(UnavailableReason.INSUFFICIENT_DATA, "no load")]
        return [Computed(value=PmcDay(ctl=self._pmc_ctl, atl=40.0, tsb=10.0))]

    async def critical_power(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date
    ) -> MetricResult[CPFit]:
        self.athlete_calls.append(athlete_id)
        if self._missing:
            return Unavailable(UnavailableReason.POOR_FIT, "bad fit")
        return Computed(
            value=CPFit(
                cp_w=280.0,
                w_prime_j=20000.0,
                r2=0.99,
                se_cp=1.0,
                se_wprime=100.0,
                residuals=(),
            )
        )

    async def power_curve(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date
    ) -> dict[int, MetricResult[Any]]:
        self.athlete_calls.append(athlete_id)
        return {}

    async def coggan(self, activity_id: str) -> MetricResult[Any]:
        self.activity_calls.append(activity_id)
        if self._missing:
            return Unavailable(UnavailableReason.MISSING_REQUIRED_INPUT, "no power")
        return Computed(value="load-bundle")

    async def aerobic_decoupling(self, activity_id: str) -> MetricResult[float]:
        self.activity_calls.append(activity_id)
        return Computed(value=4.2)

    async def trimp(self, activity_id: str) -> MetricResult[float]:
        self.activity_calls.append(activity_id)
        return Computed(value=88.0)

    async def durability(self, activity_id: str) -> MetricResult[Any]:
        self.activity_calls.append(activity_id)
        if self._missing:
            return Unavailable(UnavailableReason.INSUFFICIENT_DATA, "no fatigued state")
        return Computed(value="durability-decrement")

    async def hrv(self, athlete_id: str, local_date: _dt.date) -> MetricResult[TimeDomainHrv]:
        self.athlete_calls.append(athlete_id)
        if self._missing:
            return Unavailable(UnavailableReason.INSUFFICIENT_DATA, "no rr")
        return Computed(
            value=TimeDomainHrv(rmssd_ms=42.0, sdnn_ms=55.0, pnn50_pct=12.0, mean_nn_ms=900.0)
        )


def _svc(**kw: Any) -> Any:
    """A seeded fake typed as the analytics service the gather expects (duck-typed)."""
    return FakeAnalyticsService(**kw)


# --------------------------------------------------------------------------- #
# The registry (PLAN-R3, TOOL-R1: one entry == one canonical method, typed)    #
# --------------------------------------------------------------------------- #

_EXPECTED_METHOD_BY_KEY = {
    "weekly_load": "pmc",
    "critical_power": "critical_power",
    "power_curve": "power_curve",
    "load_metrics": "coggan",
    "hrv": "hrv",
    "decoupling": "aerobic_decoupling",
    "trimp": "trimp",
    "durability": "durability",
}


def test_registry_has_the_phase1_set_with_unique_keys() -> None:
    keys = [c.key for c in CAPABILITIES]
    assert set(keys) == set(_EXPECTED_METHOD_BY_KEY)
    assert len(keys) == len(set(keys)), "capability keys are unique"
    assert all(isinstance(c, Capability) for c in CAPABILITIES)


def test_each_capability_maps_1to1_to_a_real_service_method() -> None:
    for cap in CAPABILITIES:
        assert cap.service_method == _EXPECTED_METHOD_BY_KEY[cap.key]
        # PLAN-R3: the named method exists on the canonical service surface.
        assert hasattr(FakeAnalyticsService, cap.service_method)


def test_capability_by_key_indexes_every_capability() -> None:
    assert set(CAPABILITY_BY_KEY) == {c.key for c in CAPABILITIES}
    for key, cap in CAPABILITY_BY_KEY.items():
        assert cap.key == key


def test_param_schemas_are_typed_and_carry_no_source_or_table_names() -> None:
    """PLAN-R2: params are typed dates / activity ref / metric enum — never source/table."""
    banned = {"source", "table", "column", "query", "sql", "athlete_id", "tenant"}
    for cap in CAPABILITIES:
        schema = cap.param_schema
        assert issubclass(schema, BaseModel)
        fields = set(schema.model_fields)
        assert fields, f"{cap.key} has typed params"
        assert not (fields & banned), f"{cap.key} leaks a forbidden param: {fields & banned}"


def test_param_schemas_reject_unknown_keys() -> None:
    """PLAN-R2: closed schemas — a smuggled key (e.g. a source/table) is rejected."""
    with pytest.raises(ValueError):
        DateRangeParams.model_validate({"from_date": _DAY, "to_date": _DAY, "source": "garmin"})
    with pytest.raises(ValueError):
        ActivityParams.model_validate({"activity_id": "a", "table": "activities"})


def test_metric_name_is_a_closed_enum() -> None:
    assert MetricName("ctl") is MetricName.CTL
    with pytest.raises(ValueError):
        MetricName("power_w")  # not a member -> a model cannot request it


def test_typed_param_schemas_parse_their_own_inputs() -> None:
    assert DateRangeParams.model_validate({"from_date": _DAY, "to_date": _DAY}).from_date == _DAY
    assert ActivityParams.model_validate({"activity_id": "a"}).activity_id == "a"
    assert WellnessDayParams.model_validate({"local_date": _DAY}).local_date == _DAY
    with pytest.raises(ValueError):
        ActivityParams.model_validate({"activity_id": ""})  # min_length=1


# --------------------------------------------------------------------------- #
# gather (PLAN-R3/R5, TOOL-R5: deterministic, athlete from arg, gaps not lies) #
# --------------------------------------------------------------------------- #


async def test_gather_keys_results_by_canonical_capability_id() -> None:
    svc = _svc()
    reqs = [
        RetrievalRequest("weekly_load", {"from_date": _DAY, "to_date": _DAY}),
        RetrievalRequest("trimp", {"activity_id": "act-1"}),
    ]
    out = (await gather(svc, _ATHLETE, reqs)).records
    assert set(out) == {"weekly_load", "trimp"}
    assert out["trimp"].value == 88.0  # canonical Computed envelope, verbatim


async def test_gather_ignores_scope_override_and_emits_anomaly() -> None:
    """PLAN-R5/AGT-OBS-R5a: an athlete-shaped key in params is IGNORED + flagged.

    The smuggled ``athlete_id`` is stripped before validation (so the request still
    resolves under the authenticated scope), and a typed anomaly event records the
    attempt, the ignored override, and the authenticated scope used — never adopting it.
    """
    svc = _svc()
    reqs = [
        RetrievalRequest(
            "weekly_load",
            {"from_date": _DAY, "to_date": _DAY, "athlete_id": _OTHER_ATHLETE},
        )
    ]
    result = await gather(svc, _ATHLETE, reqs)
    # The request resolved under the AUTHENTICATED athlete; the other was never called.
    assert _OTHER_ATHLETE not in svc.athlete_calls
    assert svc.athlete_calls == [_ATHLETE]
    # The override was detected, ignored, and recorded (AGT-OBS-R5a).
    assert len(result.anomalies) == 1
    anomaly = result.anomalies[0]
    assert anomaly.kind == "scope_override_ignored"
    assert anomaly.attempted_keys == ("athlete_id",)
    assert anomaly.ignored_override == {"athlete_id": _OTHER_ATHLETE}
    assert anomaly.authenticated_scope == _ATHLETE


async def test_gather_uses_only_the_argument_athlete_for_valid_requests() -> None:
    svc = _svc()
    reqs = [RetrievalRequest("hrv", {"local_date": _DAY})]
    result = await gather(svc, _ATHLETE, reqs)
    assert svc.athlete_calls == [_ATHLETE]
    assert result.anomalies == ()


async def test_gather_records_gap_for_unknown_capability_not_fabrication() -> None:
    """TOOL-R5: an out-of-registry capability is a typed gap, never a crash or success."""
    svc = _svc()
    out = (await gather(svc, _ATHLETE, [RetrievalRequest("does_not_exist", {})])).records
    gap = out["does_not_exist"]
    assert gap["available"] is False
    assert gap["reason"] == "unknown_capability"


async def test_gather_records_gap_for_invalid_params() -> None:
    svc = _svc()
    out = (await gather(svc, _ATHLETE, [RetrievalRequest("load_metrics", {})])).records
    assert out["load_metrics"]["available"] is False
    assert out["load_metrics"]["reason"] == "invalid_params"


async def test_gather_passes_through_unavailable_canonical_result() -> None:
    """TOOL-R5: an Unavailable canonical computation is surfaced verbatim, never faked."""
    svc = _svc(missing=True)
    out = (
        await gather(svc, _ATHLETE, [RetrievalRequest("load_metrics", {"activity_id": "act-1"})])
    ).records
    result = out["load_metrics"]
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.MISSING_REQUIRED_INPUT


async def test_gather_routes_activity_capabilities_by_activity_id() -> None:
    svc = _svc()
    out = (
        await gather(svc, _ATHLETE, [RetrievalRequest("decoupling", {"activity_id": "act-9"})])
    ).records
    assert svc.activity_calls == ["act-9"]
    assert out["decoupling"].value == 4.2


async def test_gather_routes_durability_by_activity_id() -> None:
    """The durability capability resolves to svc.durability for the named activity (PLAN-R3)."""
    svc = _svc()
    out = (
        await gather(svc, _ATHLETE, [RetrievalRequest("durability", {"activity_id": "act-d"})])
    ).records
    assert svc.activity_calls == ["act-d"]
    assert out["durability"].value == "durability-decrement"


# --------------------------------------------------------------------------- #
# CanonicalEvidence (GROUND-R7: verbatim values + first-party URL allow-list)  #
# --------------------------------------------------------------------------- #


def test_canonical_evidence_satisfies_grounding_protocol() -> None:
    assert isinstance(CanonicalEvidence(_svc(), _ATHLETE), GroundingEvidence)


async def test_metric_value_reads_pmc_scalar_verbatim() -> None:
    svc = _svc(pmc_ctl=63.5)
    ev = CanonicalEvidence(svc, _ATHLETE)
    assert await ev.metric_value("ctl", _DAY.isoformat()) == 63.5
    # GROUND-R7 / PLAN-R5: evidence is scoped to the constructed athlete only.
    assert svc.athlete_calls == [_ATHLETE]


async def test_metric_value_form_is_a_verbatim_alias_of_canonical_tsb() -> None:
    """GROUND-R7: ``form`` is the athlete-facing alias of TSB -> same canonical PmcDay.tsb.

    The seeded PMC day has ``tsb=10.0``; a ``form`` NUMBER claim must ground to the IDENTICAL
    value as a ``tsb`` claim (a pure alias, never a second/different number), and stays
    scoped to the constructed athlete only.
    """
    svc = _svc()
    ev = CanonicalEvidence(svc, _ATHLETE)
    form = await ev.metric_value("form", _DAY.isoformat())
    assert form == 10.0
    assert form == await ev.metric_value("tsb", _DAY.isoformat())
    # GROUND-R7 / PLAN-R5: every canonical read stays scoped to the constructed athlete.
    assert set(svc.athlete_calls) == {_ATHLETE}


async def test_metric_value_reads_critical_power_fields_verbatim() -> None:
    ev = CanonicalEvidence(_svc(), _ATHLETE)
    assert await ev.metric_value("critical_power_w", _DAY.isoformat()) == 280.0
    assert await ev.metric_value("w_prime_j", _DAY.isoformat()) == 20000.0


async def test_metric_value_reads_hrv_verbatim() -> None:
    ev = CanonicalEvidence(_svc(), _ATHLETE)
    assert await ev.metric_value("hrv_rmssd_ms", _DAY.isoformat()) == 42.0


async def test_metric_value_unknown_metric_returns_none() -> None:
    ev = CanonicalEvidence(_svc(), _ATHLETE)
    assert await ev.metric_value("vo2max", _DAY.isoformat()) is None


async def test_metric_value_no_date_token_falls_back_to_latest_day() -> None:
    """A claim with NO date token reads the metric's LATEST available day (§16 fallback).

    A real model states a number with no as-of date ("your fitness is 6.7"); rather than
    failing closed (which would scrub a CORRECT answer), the evidence reads the metric at its
    latest available PMC day so the natural dateless claim still grounds (GROUND-R7). The seeded
    fake returns a computed day for any window, so a ``None`` (and an empty/whitespace token,
    which is also "no date") resolves to that latest value.
    """
    ev = CanonicalEvidence(_svc(pmc_ctl=63.5), _ATHLETE)
    assert await ev.metric_value("ctl", None) == 63.5
    assert await ev.metric_value("ctl", "   ") == 63.5  # whitespace-only == no date token


async def test_metric_value_unparseable_date_fails_closed_not_latest() -> None:
    """A claim with a date token that fails to parse FAILS CLOSED, never latest (H2 / GROUND-R7).

    The fabrication path: "On May 1 your fitness was 100" extracts as_of="May 1", which is NOT an
    ISO date and fails to parse. Silently resolving it to the LATEST day would ground a PAST-dated
    claim against today's value (grounded "100" while May-1 differed). An invalid/unparseable
    as_of must therefore resolve to ``None`` (scrub), distinct from a truly absent date token.
    The fake would return 63.5 for any window, so a leak would surface as 63.5; fail-closed is None.
    """
    ev = CanonicalEvidence(_svc(pmc_ctl=63.5), _ATHLETE)
    assert await ev.metric_value("ctl", "not-a-date") is None
    assert await ev.metric_value("ctl", "May 1") is None
    # The fail-closure spans every metric family, not just PMC (CP / HRV too).
    assert await ev.metric_value("critical_power_w", "May 1") is None
    assert await ev.metric_value("hrv_rmssd_ms", "May 1") is None


async def test_metric_value_resolves_natural_label_via_equivalence() -> None:
    """A natural metric label grounds through the config-loaded equivalence layer (§16/GROUND-R2).

    The model emits ``"Chronic Training Load (CTL)"`` / ``"fitness"`` / ``"form"`` — not the
    canonical key — so without the metric-equivalence bridge every NUMBER claim scrubs and the
    grounder abstains on a CORRECT answer (the headline bug). With the alias map injected, those
    labels resolve to their canonical metric and read the same verbatim value.
    """
    equiv = MetricEquivalence(
        {"Chronic Training Load (CTL)": "ctl", "fitness": "ctl", "form": "tsb"}
    )
    ev = CanonicalEvidence(_svc(pmc_ctl=63.5), _ATHLETE, equivalence=equiv)
    assert await ev.metric_value("Chronic Training Load (CTL)", _DAY.isoformat()) == 63.5
    assert await ev.metric_value("fitness", _DAY.isoformat()) == 63.5
    # An unmapped, non-canonical label still fails closed (GROUND-R3).
    assert await ev.metric_value("vibes", _DAY.isoformat()) is None


async def test_metric_value_without_equivalence_is_canonical_key_only() -> None:
    """With no equivalence injected, only exact canonical keys resolve (back-compat)."""
    ev = CanonicalEvidence(_svc(pmc_ctl=63.5), _ATHLETE)
    assert await ev.metric_value("ctl", _DAY.isoformat()) == 63.5
    # A natural label is NOT resolved without the alias layer -> None (fail-closed).
    assert await ev.metric_value("fitness", _DAY.isoformat()) is None


async def test_metric_value_unavailable_canonical_result_returns_none() -> None:
    """When the canonical service cannot compute it, evidence yields None -> grounder scrubs."""
    ev = CanonicalEvidence(_svc(missing=True), _ATHLETE)
    assert await ev.metric_value("ctl", _DAY.isoformat()) is None
    assert await ev.metric_value("critical_power_w", _DAY.isoformat()) is None
    assert await ev.metric_value("hrv_rmssd_ms", _DAY.isoformat()) is None


def test_url_allowed_is_a_first_party_https_allow_list() -> None:
    """The allow-list is the config-loaded host set (CFG-R1a), exact-host + https (GROUND-R4)."""
    hosts = frozenset({"wattwise.app", "www.wattwise.app", "docs.wattwise.app"})
    ev = CanonicalEvidence(_svc(), _ATHLETE, allowed_hosts=hosts)
    assert ev.url_allowed("https://wattwise.app/training/load") is True
    assert ev.url_allowed("https://docs.wattwise.app/glossary") is True
    # Off-list hosts, plaintext, and look-alikes are all rejected (GROUND-R7).
    assert ev.url_allowed("http://wattwise.app/x") is False
    assert ev.url_allowed("https://evil.com/wattwise.app") is False
    assert ev.url_allowed("https://wattwise.app.evil.com") is False
    assert ev.url_allowed("not a url") is False


def test_url_allowed_with_no_configured_hosts_rejects_all() -> None:
    """With NO config-loaded host list injected, the allow-list is empty -> every URL fails closed.

    The host set is loaded policy (CFG-R1a) wired in by the CoachBundle; a bare evidence object
    with no hosts must reject every link (fail-closed, GROUND-R4), never silently re-introduce a
    code-baked default host.
    """
    ev = CanonicalEvidence(_svc(), _ATHLETE)
    assert ev.url_allowed("https://wattwise.app/training/load") is False


# --------------------------------------------------------------------------- #
# Per-ride TSS resolver (#47): activity-keyed, double-unwrapped, fail-closed   #
# --------------------------------------------------------------------------- #


class _Bundle:
    """A minimal LoadMetricsBundle-shaped object: only the ``tss`` MetricResult is read."""

    def __init__(self, tss: MetricResult[float]) -> None:
        self.tss = tss


class _CogganService:
    """Analytics stub whose ``coggan`` returns a double-wrapped per-activity load bundle.

    ``act-power`` -> Computed bundle with Computed TSS (the power path); ``act-hr`` -> Computed
    bundle whose TSS is Unavailable (HR path / non-power sport); any other id -> Unavailable
    (unknown/ungathered activity). Records the activity ids it was asked for.
    """

    def __init__(self) -> None:
        self.activity_calls: list[str] = []

    async def coggan(self, activity_id: str) -> MetricResult[Any]:
        self.activity_calls.append(activity_id)
        if activity_id == "act-power":
            return Computed(value=_Bundle(Computed(value=100.0)))
        if activity_id == "act-hr":
            return Computed(
                value=_Bundle(Unavailable(UnavailableReason.OUT_OF_DOMAIN, "hr_load not power tss"))
            )
        return Unavailable(UnavailableReason.MISSING_REQUIRED_INPUT, "unknown activity")


def _tss_evidence() -> CanonicalEvidence:
    eq = MetricEquivalence({"tss": "activity_tss", "training stress score": "activity_tss"})
    return CanonicalEvidence(_CogganService(), _ATHLETE, equivalence=eq)  # type: ignore[arg-type]


async def test_metric_value_resolves_per_ride_tss_via_coggan() -> None:
    """#47: a per-ride TSS claim resolves verbatim via svc.coggan(activity_id).value.tss.

    The ``as_of`` argument is the ACTIVITY id (the claim's ref), not a date — proving the
    ACTIVITY_TSS branch fires BEFORE _resolve_as_of (a non-ISO id would otherwise scrub as INVALID).
    """
    ev = _tss_evidence()
    assert await ev.metric_value("tss", "act-power") == 100.0
    # The alias surface resolves too, and the same non-date ref keys the lookup.
    assert await ev.metric_value("training stress score", "act-power") == 100.0


async def test_metric_value_per_ride_tss_fails_closed() -> None:
    """#47 fail-closed (GROUND-R7): HR-based, unknown, ref-less per-ride TSS all resolve None."""
    ev = _tss_evidence()
    assert await ev.metric_value("tss", "act-hr") is None  # HR path: tss Unavailable -> scrub
    assert await ev.metric_value("tss", "act-unknown") is None  # ungathered activity -> scrub
    assert await ev.metric_value("tss", None) is None  # no activity ref -> per-day ambiguous scrub
