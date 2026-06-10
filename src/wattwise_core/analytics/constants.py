"""Externalized analytics constants (doc 40 §4 — the quick-reference table).

Sports-science constants live here as overridable config, not as magic numbers
buried in formula code (ANL — externalized constants are open-core config so a
commercial re-fit can override them without a code change). Each constant cites the
requirement that owns its default.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Final

_DEFAULTS_PATH = Path(__file__).parents[1] / "config" / "defaults.toml"


def _analytics_default(key: str) -> float:
    """Read one ``[analytics]`` scalar from the packaged dead config file (CFG-R1a).

    The VALUE lives in ``defaults.toml`` — never as a code-baked literal. A key absent
    from the packaged defaults fails closed (CFG-R6), never silently falling back to a
    literal. This is a one-time module-load read, so the pure metric functions that import
    the resolved constant stay free of per-call I/O (ANL-R2/R30).
    """
    with _DEFAULTS_PATH.open("rb") as fh:
        analytics = tomllib.load(fh).get("analytics", {})
    if key not in analytics:
        raise RuntimeError(
            f"fail-closed: analytics.{key} is absent from {_DEFAULTS_PATH.name} "
            "(CFG-R1a: the value belongs in defaults.toml, not a code literal)"
        )
    return float(analytics[key])

# --- resampling / time base ---
MAX_INTERP_GAP_S: Final = 3.0  # ANL-R8
RDP_MAX_BISECTION_STEPS: Final = 32  # ANL-R8b

# --- PMC ---
CTL_TIME_CONSTANT_DAYS: Final = 42.0  # PMC-R1
ATL_TIME_CONSTANT_DAYS: Final = 7.0  # PMC-R1
WINDOWED_EQUIV_ABS_TOL: Final = 1e-9  # PMC-R4 (× max(1,|value|))  # noqa: RUF003 - math notation

# --- training-load equivalence class (DM-SUB-R1 worked example; LOAD-R3 priority) ---
# The canonical ``training_load`` channel's declared equivalence-class label. Power-based
# TSS is the top (``raw_stream``) member; the HR-derived Banister load (TRIMP) is the
# lowest ``modeled`` member (DM-SUB-R1 worked example). When a higher member is withdrawn
# and the load is recomputed from a lower one, coverage carries ``Fidelity.SUBSTITUTED`` +
# ``Substitution(class, from_fidelity)`` (DEGR-R2). The label is config (CFG-R1a), not a
# formula constant; the member fidelity tiers live in :mod:`wattwise_core.analytics.service`
# beside the LOAD-R3 fallback that produces them.
TRAINING_LOAD_CLASS: Final = "training_load"  # DM-SUB-R1 / DEGR-R2 equivalence-class label
# DEGR-R2 reduced-confidence multiplier on substitution. The VALUE is NOT a code literal
# (CFG-R1a): it is loaded from ``[analytics]`` in the packaged defaults.toml (the dead config
# file), typed + range-validated by ``Settings.analytics__training_load_confidence_penalty``,
# and overridable by the operator file / environment via that schema field.
TRAINING_LOAD_CONFIDENCE_PENALTY: Final[float] = _analytics_default(
    "training_load_confidence_penalty"
)
# GBO-R28: the stated fit-quality floor below which a stored MODELED signature is
# REFUSED by resolution (fail-closed, typed gap — never thresholds from a bad fit).
SIGNATURE_MIN_FIT_R2: Final[float] = _analytics_default("signature_min_fit_r2")

# --- Normalized Power / TSS ---
NP_ROLLING_WINDOW_S: Final = 30  # NP-R1
NP_CONSTANT_POWER_TOL: Final = 1e-6  # NP-R4
TSS_GOLDEN_TOL: Final = 1e-6  # TSS-R2

# --- W′ balance (Skiba 2012) ---  # noqa: RUF003 - W-prime (W′) math notation
SKIBA_TAU_A: Final = 546.0  # WBAL-R1
SKIBA_TAU_B: Final = -0.01  # WBAL-R1
SKIBA_TAU_C: Final = 316.0  # WBAL-R1
WBAL_INTEGRAL_PARITY_FRAC: Final = 0.02  # WBAL-R3 (× W′)  # noqa: RUF003 - math notation

# --- Critical Power / W′ ---  # noqa: RUF003 - W-prime (W′) math notation
CP_MIN_POINTS: Final = 3  # CP-R3 (never < 2)
CP_DURATION_RATIO_MIN: Final = 3.0  # CP-R3 (max/min)
CP_R2_MIN: Final = 0.95  # CP-R3
CP_DOMAIN_MIN_S: Final = 120  # CP-R2
CP_DOMAIN_MAX_S: Final = 1200  # CP-R2
CP_LONG_DURATION_BIAS_S: Final = 1200  # CP-R6 (fires strictly above)

# --- MMP grid ---
MMP_DURATION_GRID_S: Final[tuple[int, ...]] = (
    1, 5, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600,
)  # MMP-R2

# --- TRIMP (Banister-HRR), sex constants (a multiplicative, b exponential) ---
TRIMP_A_MALE: Final = 0.64  # TRIMP-R1
TRIMP_B_MALE: Final = 1.92  # TRIMP-R1
TRIMP_A_FEMALE: Final = 0.86  # TRIMP-R1
TRIMP_B_FEMALE: Final = 1.67  # TRIMP-R1

# --- HRV ---
HRV_ARTIFACT_CEILING_FRAC: Final = 0.05  # HRV-R2
HRV_MIN_DURATION_S: Final = 120  # HRV-R4 (≥ 2 min usable)
HRV_TACHOGRAM_RESAMPLE_HZ: Final = 4.0  # HRV-R6
HRV_LF_BAND_HZ: Final[tuple[float, float]] = (0.04, 0.15)  # HRV-R6
HRV_HF_BAND_HZ: Final[tuple[float, float]] = (0.15, 0.40)  # HRV-R6

# --- aerobic decoupling ---
DECOUPLING_MIN_DURATION_S: Final = 1200  # DEC-R4 (≥ 20 min)
DECOUPLING_SMOOTHING_WINDOW_S: Final = 30  # DEC-R3

# --- intensity_class IF cut-points (LM-R1) ---
INTENSITY_CLASS_CUTS: Final[tuple[float, ...]] = (0.55, 0.75, 0.90, 1.05)
INTENSITY_CLASS_LABELS: Final[tuple[str, ...]] = (
    "recovery", "endurance", "tempo", "threshold", "vo2",
)

# --- readiness/form verdict bands (QA-EVAL-R2.4) ---
# The spec pins only the DIRECTION (QA-EVAL-R2.4): deep-negative form must NOT yield
# a hard "go" day. These TSB-zone cutoffs are a defensible default banding of canonical
# form (TSB), overridable like the τ constants above; the consistency invariant lives in
# the direction, not in any one cutoff. HRV only ever nudges toward MORE caution (never
# toward GO) — fail-safe (COACH-R1 #2 / GROUND-R7).
READINESS_FRESH_FORM: Final = 5.0  # form (TSB) above this => fresh => GO
READINESS_NEUTRAL_FLOOR: Final = -10.0  # -10 <= form <= 5 => MAINTAIN
READINESS_FATIGUE_FLOOR: Final = -20.0  # -20 <= form < -10 => EASE ; form < -20 => REST
READINESS_HRV_SUPPRESSION_FRAC: Final = 0.10  # hrv >=10% below baseline => one step more cautious
# Minimum chronic fitness (CTL) for the form signal to be trustworthy enough to read a verdict
# (GROUND-R6 / PMC-R3/R5). An experienced athlete always carries ctl > 0; only a brand-new
# (0,0)-seed cold-start (PMC-R3/R5 honest origin) sits at ~0 ctl, where form is ~0 and would
# otherwise yield a confident MAINTAIN on NO training base — and a fully-detrained athlete
# legitimately reads "not enough recent fitness signal". Below this epsilon the gather treats
# form as UNAVAILABLE so the deliverable abstains rather than guessing.
READINESS_MIN_FITNESS_CTL: Final = 1.0  # GROUND-R6 / PMC-R3/R5

# --- generic numeric tolerance ---
DEFAULT_CLOSED_FORM_ABS_TOL: Final = 1e-9  # ANL-R31 (× max(1,|x|))  # noqa: RUF003 - math notation
