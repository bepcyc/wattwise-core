"""Externalized analytics constants (doc 40 §4 — the quick-reference table).

Sports-science constants live here as overridable config, not as magic numbers
buried in formula code (ANL — externalized constants are open-core config so a
commercial re-fit can override them without a code change). Each constant cites the
requirement that owns its default.
"""

from __future__ import annotations

from typing import Final

# --- resampling / time base ---
MAX_INTERP_GAP_S: Final = 3.0  # ANL-R8
RDP_MAX_BISECTION_STEPS: Final = 32  # ANL-R8b

# --- PMC ---
CTL_TIME_CONSTANT_DAYS: Final = 42.0  # PMC-R1
ATL_TIME_CONSTANT_DAYS: Final = 7.0  # PMC-R1
WINDOWED_EQUIV_ABS_TOL: Final = 1e-9  # PMC-R4 (× max(1,|value|))  # noqa: RUF003 - math notation

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

# --- generic numeric tolerance ---
DEFAULT_CLOSED_FORM_ABS_TOL: Final = 1e-9  # ANL-R31 (× max(1,|x|))  # noqa: RUF003 - math notation
