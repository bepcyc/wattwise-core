"""W' balance — Skiba (2012) differential model (doc 40 §6; WBAL-R1..R6).

W' balance tracks the moment-to-moment depletion and reconstitution of the
anaerobic work capacity ``W'`` over a cycling power stream. The default model is
the Skiba *differential* recurrence (WBAL-R1), evaluated per second on a uniform
1 Hz power series with ``Δt = 1 s`` and seeded ``W'bal(0) = W'``::

    # Expenditure  (P(t) >= CP):
    W'bal(t) = W'bal(t-1) - (P(t) - CP) * Δt

    # Recovery     (P(t) <  CP):
    W'bal(t) = W'bal(t-1) + (W' - W'bal(t-1)) * (1 - exp(-Δt / τ_W(t)))
    τ_W(t)   = 546 * e^(-0.01 * D_CP) + 316
    D_CP     = CP - P(t)        # INSTANTANEOUS per-second deficit (W below CP)

The Skiba constants ``546 / -0.01 / 316`` come from :mod:`wattwise_core.analytics.constants`
(``SKIBA_TAU_A/B/C``), externalized open-core config (WBAL-R1), never inlined here.
``τ_W(t)`` is recomputed **every** recovery second from the instantaneous deficit —
not a single per-athlete constant. The boundary ``P == CP`` belongs to the
**expenditure** branch (work-above-CP is zero, so the value is unchanged), NOT to
the recovery branch (WBAL-R5).

Invariants enforced here:

* ``W'bal(t) <= W'`` for every ``t`` (recovery never overfills; WBAL-R2/R5).
* The series may go **negative** on over-exhaustion when ``floor`` is disabled
  (the default) — this is permitted, not an error, and is reported raw, never
  silently floored (WBAL-R2). ``floor=True`` applies the separate, named 0-floor
  clamping policy yielding ``max(0, raw)`` and records it in provenance.
* Every value is finite (WBAL-R6); any non-finite path fails closed to
  ``Unavailable(OUT_OF_DOMAIN)`` (ANL-R32).

Fail-closed reason mapping (doc 40 §6):

* Missing power stream / missing ``CP`` / missing ``W'`` ⇒
  ``Unavailable(MISSING_REQUIRED_INPUT)`` (WBAL-R4).
* Non-finite ``CP``/``W'``, non-positive ``CP``/``W'``, or a non-finite value
  produced anywhere on the recurrence ⇒ ``Unavailable(OUT_OF_DOMAIN)`` (WBAL-R6/
  ANL-R32).

Every metric here is a PURE function (ANL-R2/R30): no I/O, no wall-clock, no RNG,
no global mutable state; it returns the typed :data:`MetricResult` envelope
(ANL-R3/R4), never a bare number.

Citation: Skiba, Chidnok, Vanhatalo & Jones (2012), "Modeling the Expenditure and
Reconstitution of Work Capacity above Critical Power," *Med. Sci. Sports Exerc.*
44(8):1526-1532.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wattwise_core.analytics.constants import SKIBA_TAU_A, SKIBA_TAU_B, SKIBA_TAU_C
from wattwise_core.analytics.result import (
    Computed,
    InputLineage,
    MetricResult,
    QualityReport,
    Unavailable,
    UnavailableReason,
)
from wattwise_core.analytics.series import FloatArray

__all__ = ["WBalResult", "wbal"]

# Cycling-power-specific metric (doc 40 §5): true mechanical power required.
APPLICABLE_SPORTS: tuple[str, ...] = ("cycling",)


@dataclass(frozen=True, slots=True)
class WBalResult:
    """The value carried by a :class:`Computed` W'-balance result.

    ``series`` is the per-second W' balance in joules (same length as the input
    power series), and ``w_prime_balance_min`` is its minimum (the deepest
    depletion; most negative when ``floor`` is disabled and the athlete went into
    over-exhaustion).
    """

    series: FloatArray
    w_prime_balance_min: float


def _tau_w(d_cp: float) -> float:
    """Instantaneous recovery time constant ``τ_W`` (WBAL-R1).

    ``τ_W(t) = SKIBA_TAU_A * exp(SKIBA_TAU_B * D_CP) + SKIBA_TAU_C`` with
    ``D_CP = CP - P(t)`` the instantaneous deficit (watts below CP), recomputed
    every recovery second from the externalized Skiba constants.
    """
    return SKIBA_TAU_A * math.exp(SKIBA_TAU_B * d_cp) + SKIBA_TAU_C


def _validate_wbal_inputs(  # noqa: PLR0911 -- each fail-closed gate (WBAL-R4/R6) is a distinct typed return
    power_1hz: FloatArray | None,
    cp_w: float | None,
    w_prime_j: float | None,
) -> tuple[FloatArray, float, float] | Unavailable:
    """Presence + domain gates for the W'-balance inputs (WBAL-R4/R6).

    Returns ``(power, cp, w_prime)`` on success, else a typed :class:`Unavailable`:
    a missing/empty/all-gap power stream or a missing CP/W' ⇒
    ``MISSING_REQUIRED_INPUT``; a non-1-D, non-finite, or non-positive CP/W' ⇒
    ``OUT_OF_DOMAIN``.
    """
    # --- WBAL-R4: required-input presence ----------------------------------
    if power_1hz is None:  # defensive; typed param but tolerate explicit None
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            "power stream is required for W' balance",
        )
    power = np.asarray(power_1hz, dtype=np.float64)
    if power.ndim != 1:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "power_1hz must be a 1-D series",
        )
    if power.size == 0:
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            "power stream is empty",
        )
    if not np.any(~np.isnan(power)):
        # All-gap stream: no usable power at all.
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            "power stream has no valid samples",
        )

    # --- WBAL-R4/R6: CP and W' presence + domain ---------------------------
    if cp_w is None or w_prime_j is None:
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            "both CP and W' are required for W' balance",
        )
    cp = float(cp_w)
    w_prime = float(w_prime_j)
    if not math.isfinite(cp) or not math.isfinite(w_prime):
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "CP and W' must be finite",
        )
    if cp <= 0.0 or w_prime <= 0.0:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "CP and W' must be strictly positive",
        )
    return power, cp, w_prime


def _compute_wbal_series(
    power: FloatArray, cp: float, w_prime: float
) -> tuple[FloatArray, int, int] | Unavailable:
    """Run the Skiba differential recurrence per second (WBAL-R1/R2/R5/R6).

    Seeded ``W'bal(0) = W'``. Expenditure (``P >= CP``, boundary included) subtracts
    work-above-CP; recovery recomputes ``τ_W`` from the instantaneous deficit; a gap
    carries the balance forward. Recovery never overfills above ``W'`` (WBAL-R2). A
    non-finite value fails closed to ``OUT_OF_DOMAIN`` (WBAL-R6/ANL-R32). Returns
    ``(series, gap_count, valid_seconds)``.
    """
    n = power.size
    series = np.empty(n, dtype=np.float64)
    prev = w_prime  # W'bal at t-1; seed is W' itself (WBAL-R1).
    gap_count = 0
    valid_seconds = 0
    for t in range(n):
        p = power[t]
        if math.isnan(p):
            # Gap: carry the balance forward unchanged (no work, no recovery).
            gap_count += 1
            cur = prev
        else:
            valid_seconds += 1
            if p >= cp:
                # Expenditure (boundary P == CP lives here, WBAL-R5): zero work
                # above CP when P == CP leaves the value unchanged.
                cur = prev - (p - cp)
            else:
                # Recovery: τ_W recomputed from the instantaneous deficit.
                d_cp = cp - p
                tau = _tau_w(d_cp)
                cur = prev + (w_prime - prev) * (1.0 - math.exp(-1.0 / tau))
            # WBAL-R2: recovery must never overfill above W'.
            cur = min(cur, w_prime)
        if not math.isfinite(cur):
            # WBAL-R6 / ANL-R32: never let a non-finite value escape a Computed.
            return Unavailable(
                UnavailableReason.OUT_OF_DOMAIN,
                f"non-finite W' balance at t={t}",
            )
        series[t] = cur
        prev = cur
    return series, gap_count, valid_seconds


def _build_wbal_result(
    series: FloatArray,
    *,
    cp: float,
    w_prime: float,
    gap_count: int,
    valid_seconds: int,
    floor: bool,
    sport: str,
) -> Computed[WBalResult]:
    """Apply the optional 0-floor policy and assemble the ``Computed`` envelope.

    ``floor=True`` applies the separate named ``max(0, raw)`` clamping policy and
    records it in provenance (WBAL-R2/R5); otherwise the raw (possibly negative)
    series is reported. Coverage is valid seconds / series length (ANL-R5).
    """
    if floor:
        series = np.maximum(series, 0.0)
    raw_min = float(np.min(series))

    n = series.size
    coverage = valid_seconds / n if n else 0.0
    quality = QualityReport(
        coverage_fraction=coverage,
        sample_rate_hz=1.0,
        gap_count=gap_count,
        confidence=1.0,
        extra={
            "model": "skiba_2012_differential",
            "floor_policy": "max_0" if floor else "raw",
            "valid_seconds": valid_seconds,
        },
    )
    provenance = InputLineage(
        sport=sport,
        channels=("power",),
        reference_params={
            "cp_w": cp,
            "w_prime_j": w_prime,
            "skiba_tau_a": SKIBA_TAU_A,
            "skiba_tau_b": SKIBA_TAU_B,
            "skiba_tau_c": SKIBA_TAU_C,
            "floor": floor,
        },
    )
    return Computed(
        value=WBalResult(series=series, w_prime_balance_min=raw_min),
        quality=quality,
        provenance=provenance,
    )


def wbal(
    power_1hz: FloatArray | None,
    cp_w: float | None,
    w_prime_j: float | None,
    floor: bool = False,
    *,
    sport: str = "cycling",
) -> MetricResult[WBalResult]:
    """Compute the Skiba (2012) differential W' balance series (WBAL-R1..R6).

    Parameters
    ----------
    power_1hz:
        Uniform 1 Hz power series in watts (``Δt = 1 s``). ``NaN`` marks a gap.
        Must be already resampled to 1 Hz by the caller (ANL-R8).
    cp_w:
        Critical power in watts (canonical, time-effective; ANL-R9). Never the
        FTP surrogate without recorded provenance (WBAL-R4).
    w_prime_j:
        Anaerobic work capacity ``W'`` in joules (canonical, time-effective).
    floor:
        When ``False`` (default) the raw series is reported and may go negative
        (WBAL-R2). When ``True`` the separate named 0-floor clamping policy is
        applied (``max(0, raw)``) and recorded in provenance (WBAL-R5).
    sport:
        The canonical sport of the activity (ANL-R11). W'balance is cycling-power-
        specific: a ``sport`` outside :data:`APPLICABLE_SPORTS` fails closed with
        ``NOT_APPLICABLE_FOR_SPORT`` BEFORE any computation — a power channel carried
        by a non-power sport is never turned into a cycling W'bal trace (ANL-R12,
        SPORT-T2/T3). This is distinct from the ``MISSING_REQUIRED_INPUT`` returned
        when the sport CAN have power but the channel/CP/W' is absent (WBAL-R4).

    Returns
    -------
    MetricResult[WBalResult]
        ``Computed`` carrying the per-second series and its minimum, plus a
        :class:`QualityReport` and :class:`InputLineage`; or a typed
        :class:`Unavailable` on any fail-closed path.

    Notes
    -----
    Boundary ``P == CP`` is the expenditure branch (WBAL-R5): ``(P - CP) * Δt``
    is zero so the value is unchanged. A ``NaN`` (gap) second carries the balance
    forward unchanged (the gap contributes no work and no recovery), preserving
    ``W'bal <= W'`` and finiteness.
    """
    if sport not in APPLICABLE_SPORTS:
        return Unavailable(
            UnavailableReason.NOT_APPLICABLE_FOR_SPORT,
            f"W'balance is a cycling-power-family metric, not defined for sport {sport!r}",
        )

    validated = _validate_wbal_inputs(power_1hz, cp_w, w_prime_j)
    if isinstance(validated, Unavailable):
        return validated
    power, cp, w_prime = validated

    # --- recurrence (WBAL-R1), seeded W'bal(0) = W' ------------------------
    computed = _compute_wbal_series(power, cp, w_prime)
    if isinstance(computed, Unavailable):
        return computed
    series, gap_count, valid_seconds = computed

    return _build_wbal_result(
        series,
        cp=cp,
        w_prime=w_prime,
        gap_count=gap_count,
        valid_seconds=valid_seconds,
        floor=floor,
        sport=sport,
    )
