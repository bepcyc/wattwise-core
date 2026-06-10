"""Per-activity load-metrics bundle assembly (doc 40 §4 / §7B, LM-R1/R2/R3, LOAD-R4).

Composes the cycling-power load family computed in
:mod:`wattwise_core.analytics.np_if_tss` (NP/IF/TSS) plus the ratio metrics
(efficiency factor, variability index), the per-hour TSS, and the single honest
``load_model`` selection into the typed :class:`LoadMetricsBundle`. Split out of
``np_if_tss`` to keep each module within the QUAL-R9 size ceiling while preserving the
public ``np_if_tss.load_metrics_bundle`` / ``np_if_tss.LoadMetricsBundle`` entry points
(re-exported from that module). One-directional dependency: this module imports the core
NP/IF/TSS metrics from ``np_if_tss``; ``np_if_tss`` never imports back at module-body
time (only the bottom-of-file re-export pulls these names in, after all its own
definitions exist), so there is no import cycle.

Pure, deterministic, fail-closed (ANL-R2/R30/R3/R4): every bundle field is an independent
:data:`~wattwise_core.analytics.result.MetricResult` that propagates ``Unavailable`` with
the exact typed reason; never a fabricated 0/default, and never an HR-load value relabeled
as power TSS (LM-R2/LM-T2). The power family is cycling-power-specific (ANL-R11): a
non-cycling ``sport`` yields ``NOT_APPLICABLE_FOR_SPORT`` via the NP gate (ANL-R12).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wattwise_core.analytics.np_if_tss import (
    APPLICABLE_SPORTS,
    LOAD_MODEL_POWER_TSS,
    POWER_CHANNEL,
    NormalizedPowerValue,
    _not_applicable_for_sport,
    _resampled_power,
    _valid_moving_seconds,
    intensity_class,
    intensity_factor,
    normalized_power,
    power_tss,
)
from wattwise_core.analytics.result import (
    Computed,
    InputLineage,
    MetricResult,
    QualityReport,
    Unavailable,
    UnavailableReason,
    is_computed,
)
from wattwise_core.analytics.series import Stream
from wattwise_core.analytics.trimp import LOAD_MODEL_HR_LOAD


@dataclass(frozen=True, slots=True)
class LoadMetricsBundle:
    """Per-activity load-metrics bundle (LM-R1), each field an independent result.

    Distinct from the daily load-metrics time-series (doc 40 §7 note 9). The bundle's
    load field is ``tss`` on the power path OR ``hr_load`` on the HR path (LM-R1); the
    two are never both populated as "the" load (LM-R2/LM-T2). ``load_model`` is the
    mandatory honest label of which member of the canonical LOAD-R2 set
    (``power_tss | hr_load | hr_load_zonal``) produced the load field — carrying
    ``hr_load_zonal`` whenever the zonal HR path was resolved (LOAD-R4), never a token
    outside that set.
    """

    duration_valid_s: MetricResult[int]
    np: MetricResult[NormalizedPowerValue]
    if_: MetricResult[float]
    tss: MetricResult[float]
    hr_load: MetricResult[float]
    tss_per_hour: MetricResult[float]
    efficiency_factor: MetricResult[float]
    variability_index: MetricResult[float]
    intensity_class: MetricResult[str]
    load_model: str


def _ratio_metric(
    base: MetricResult[NormalizedPowerValue],
    denominator: float | None,
    *,
    denom_name: str,
) -> MetricResult[float]:
    """NP / denominator with fail-closed handling (EF and VI, LM-R1).

    Computed only when NP is Computed AND ``denominator`` is present and ``> 0``;
    otherwise the typed reason: absent denominator ⇒ ``MISSING_REQUIRED_INPUT``,
    non-positive denominator ⇒ ``OUT_OF_DOMAIN`` (§6). Never substitutes a 0/default.
    """
    if isinstance(base, Unavailable):
        return base
    if denominator is None:
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            f"no {denom_name}",
        )
    if not math.isfinite(denominator) or denominator <= 0.0:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            f"{denom_name} must be a positive finite value",
        )
    value = base.value.np_w / denominator
    if not math.isfinite(value):  # ANL-R32
        return Unavailable(UnavailableReason.OUT_OF_DOMAIN, f"non-finite NP/{denom_name}")
    return Computed(
        value,
        quality=QualityReport(extra={"np_w": base.value.np_w, denom_name: denominator}),
        provenance=base.provenance,
    )


def _tss_per_hour(tss_result: MetricResult[float], duration_valid_s: int) -> MetricResult[float]:
    """``tss_per_hour = tss / (duration_valid_s / 3600)`` (LM-R1).

    Computed only when TSS is Computed and ``duration_valid_s > 0``.
    """
    if isinstance(tss_result, Unavailable):
        return tss_result
    if duration_valid_s <= 0:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "duration_valid_s must be positive for tss_per_hour",
        )
    value = tss_result.value / (duration_valid_s / 3600.0)
    if not math.isfinite(value):  # ANL-R32
        return Unavailable(UnavailableReason.OUT_OF_DOMAIN, "non-finite tss_per_hour")
    return Computed(
        value,
        quality=QualityReport(
            extra={"tss": tss_result.value, "duration_valid_s": duration_valid_s}
        ),
        provenance=tss_result.provenance,
    )


def _hr_load_model_label(hr_load_result: MetricResult[float] | None) -> str | None:
    """The HR-path ``load_model`` carried by a Computed HR-load result, else ``None``.

    Reads the label the HR-load metric itself stamped in its ``QualityReport`` (TRIMP-R2/
    R4): ``hr_load`` for Banister-HRR or ``hr_load_zonal`` for the zone-weighted variant,
    so the bundle never re-derives or downgrades the label (LOAD-R4 / LM-T2).
    """
    if hr_load_result is None or isinstance(hr_load_result, Unavailable):
        return None
    label = hr_load_result.quality.extra.get("load_model")
    return label if isinstance(label, str) else LOAD_MODEL_HR_LOAD


def _resolve_load_model(
    *,
    power_applicable: bool,
    tss_result: MetricResult[float],
    hr_load_result: MetricResult[float] | None,
) -> tuple[str, MetricResult[float], MetricResult[float]]:
    """Pick the single honest ``load_model`` and the (tss, hr_load) load fields (LM-R2).

    The two load families are never both populated as "the" load (LM-T2): the power TSS
    wins when it is ``Computed`` (and the HR field is then propagated Unavailable); else
    the HR-load value carries the load when it is ``Computed`` (and TSS stays Unavailable
    with its own typed reason). When neither is computable the field is the power TSS's
    typed Unavailable (or a missing-HR Unavailable) and ``load_model`` is the
    preferred/attempted member — ``power_tss`` when the sport admits power, else the
    HR-path label (LM-R2: a populated ``load_model`` is mandatory even on the all-
    Unavailable bundle). ``load_model`` is always a member of the canonical LOAD-R2 set.
    """
    hr_model = _hr_load_model_label(hr_load_result)
    if is_computed(tss_result):
        # Power path wins (TSS is Computed ⇒ NP Computed ⇒ sport applies the power
        # family); the HR load is NOT also "the" load on a power activity (LM-T2 / LM-R2:
        # never report an HR load WHILE also reporting power TSS). The winning model is
        # already named by ``load_model = power_tss``; the deselected HR field carries a
        # reason that does NOT fabricate channel-absence (HR may well be present — it is
        # simply not the canonical load here), so OUT_OF_DOMAIN (the load-as-HR domain
        # precondition is "no power channel") — never MISSING_REQUIRED_INPUT (LM-R1).
        hr_field: MetricResult[float] = Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "hr_load is not the load on a power-TSS activity (load_model=power_tss won)",
        )
        return LOAD_MODEL_POWER_TSS, tss_result, hr_field
    if hr_load_result is not None and is_computed(hr_load_result):
        # HR path carries the load; the power TSS keeps its own typed Unavailable.
        return hr_model or LOAD_MODEL_HR_LOAD, tss_result, hr_load_result
    # Neither computable: a populated load_model is still mandatory (LM-R2).
    hr_field = (
        hr_load_result
        if hr_load_result is not None
        else Unavailable(UnavailableReason.MISSING_REQUIRED_INPUT, "no HR-load input")
    )
    label = LOAD_MODEL_POWER_TSS if power_applicable else (hr_model or LOAD_MODEL_HR_LOAD)
    return label, tss_result, hr_field


def load_metrics_bundle(
    power_stream: Stream,
    hr_stream: Stream | None,
    ftp_w: float | None,
    avg_power_w: float | None,
    avg_hr_bpm: float | None,
    *,
    sport: str = "cycling",
    hr_load_result: MetricResult[float] | None = None,
) -> LoadMetricsBundle:
    """Assemble the per-activity load-metrics bundle (LM-R1/R2/R3, LOAD-R4).

    Every field is an independent :data:`MetricResult`. The power family (NP/IF/TSS/EF/
    VI/intensity_class) is cycling-power-specific (ANL-R11): for a ``sport`` outside
    :data:`APPLICABLE_SPORTS` those fields all propagate ``NOT_APPLICABLE_FOR_SPORT``
    via the NP gate (ANL-R12) — never a fabricated cross-sport power number. When NP is
    Unavailable, ``if``, ``tss``, ``tss_per_hour``, ``efficiency_factor``,
    ``variability_index`` and ``intensity_class`` all propagate that Unavailable (LM-R2)
    — never a 0 / default and never an HR-load value relabeled as power TSS.

    The bundle's load field is ``tss`` on the power path OR ``hr_load`` on the HR path
    (LM-R1). ``hr_load_result`` is the caller-resolved HR-load metric (Banister-HRR or
    the zonal variant per the athlete default, LOAD-R4) carrying its own ``load_model``
    label; it surfaces as the bundle's load when the power TSS is Unavailable. The two
    are never both populated as "the" load (LM-T2). ``load_model`` is the honest member
    of the canonical LOAD-R2 set that produced the load, populated even on an all-
    Unavailable bundle (LM-R2). ``duration_valid_s`` is still reported whenever a valid
    power channel exists, even if NP itself is Unavailable.

    ``efficiency_factor = NP / avg_hr_bpm`` and ``variability_index = NP / avg_power``
    use the FULL valid-moving-window means (doc 40 §7B): the caller passes those scalars;
    they are honoured as denominators with fail-closed handling.

    Internal consistency (LM-R3) holds by construction: ``if == np/FTP`` and
    ``tss == duration_valid_s*np^2/(FTP^2*3600)*100`` because every field is derived from
    the same NP result and FTP.

    ``hr_stream`` is accepted for signature compatibility; the HR-load value itself is
    resolved by the caller (the service, which owns the LOAD-R4 default selection and the
    canonical HR_max/HR_rest/sex inputs) and passed in via ``hr_load_result``.
    """
    del hr_stream  # the HR-load value is resolved by the caller (LOAD-R4) and passed in.

    power_applicable = sport in APPLICABLE_SPORTS
    power_1hz = _resampled_power(power_stream)
    duration_valid_value = _valid_moving_seconds(power_1hz)

    # duration_valid_s is reported whenever a valid power channel exists (LM-R2), even if
    # NP itself is Unavailable. For an inapplicable sport the power channel does not mean
    # mechanical power, so it is NOT_APPLICABLE_FOR_SPORT, not a valid-moving count.
    duration_result: MetricResult[int]
    if not power_applicable:
        duration_result = _not_applicable_for_sport("duration_valid_s (power)", sport)
    elif power_1hz.size == 0 or not np.any(~np.isnan(power_1hz)):
        duration_result = Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT, "no valid power channel"
        )
    elif duration_valid_value <= 0:  # pragma: no cover - implied by the branch above
        duration_result = Unavailable(
            UnavailableReason.INSUFFICIENT_DATA, "no valid-moving seconds"
        )
    else:
        duration_result = Computed(
            duration_valid_value,
            quality=QualityReport(
                coverage_fraction=duration_valid_value / power_1hz.size,
                sample_rate_hz=1.0,
                gap_count=int(np.count_nonzero(np.isnan(power_1hz))),
            ),
            provenance=InputLineage(channels=(POWER_CHANNEL,)),
        )

    np_result = normalized_power(power_stream, sport=sport)
    if_result = intensity_factor(np_result, ftp_w)
    tss_result = power_tss(np_result, ftp_w, duration_valid_value)
    tss_per_hour_result = _tss_per_hour(tss_result, duration_valid_value)
    ef_result = _ratio_metric(np_result, avg_hr_bpm, denom_name="avg_hr_bpm")
    vi_result = _ratio_metric(np_result, avg_power_w, denom_name="avg_power_w")
    ic_result = intensity_class(if_result)

    load_model, tss_field, hr_field = _resolve_load_model(
        power_applicable=power_applicable,
        tss_result=tss_result,
        hr_load_result=hr_load_result,
    )

    return LoadMetricsBundle(
        duration_valid_s=duration_result,
        np=np_result,
        if_=if_result,
        tss=tss_field,
        hr_load=hr_field,
        tss_per_hour=tss_per_hour_result,
        efficiency_factor=ef_result,
        variability_index=vi_result,
        intensity_class=ic_result,
        load_model=load_model,
    )


__all__ = [
    "LoadMetricsBundle",
    "load_metrics_bundle",
]
