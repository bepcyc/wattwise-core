"""Unit-equivalence + source-timezone invariance properties (ANL-T-R1.10).

Two analytics-output invariants:

- **Unit equivalence:** a metric computed from an input quantity expressed in an
  equivalent unit and then normalized to the canonical SI unit equals the metric
  computed from the SI input directly (km/h vs m/s on the pace-sport decoupling
  output channel) — normalization changes representation, never the metric.
- **Source-timezone invariance:** metric outputs depend only on the canonical UTC
  instant, never the source's wall-clock representation: the same instant authored
  under different source UTC offsets projects to the same athlete ``local_date``
  and produces an identical PMC series from the resulting day buckets.

Generators stay bounded and offline (TIER-R1); shrinking is hypothesis default.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from wattwise_core.analytics.pmc import pmc
from wattwise_core.analytics.result import is_computed
from wattwise_core.persistence.localdate import project_local_date

pytestmark = pytest.mark.property

CI_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@dataclass
class _Athlete:
    """Minimal as-of tz surface for local-date projection (GBO-R33/R34)."""

    reference_timezone: str
    reference_timezone_effective_from: _dt.datetime | None = None


# NOTE (#93): a former `test_metric_invariant_under_equivalent_speed_units` was REMOVED here — it
# built `normalized = [v * 3.6 / 3.6 for v in speeds]` (≡ v) and compared `aerobic_decoupling` to
# itself on the identical series. `aerobic_decoupling` takes no unit argument and does no km/h↔m/s
# normalization, so the "unit equivalence" property exercised NOTHING (a tautology giving false
# confidence). Genuine unit-equivalence belongs at the ingest/mapping seam (raw source units →
# canonical m/s), not at this analytics function; assert it there if/when needed.


# --- source-timezone invariance: only the canonical instant matters ----------------
_SOURCE_ZONES = ("UTC", "Europe/Berlin", "America/New_York", "Asia/Tokyo", "Pacific/Auckland")


@CI_SETTINGS
@given(
    epoch_s=st.integers(
        min_value=int(_dt.datetime(2025, 1, 2, tzinfo=_dt.UTC).timestamp()),
        max_value=int(_dt.datetime(2026, 6, 1, tzinfo=_dt.UTC).timestamp()),
    ),
    source_zone=st.sampled_from(_SOURCE_ZONES),
    load=st.floats(min_value=1.0, max_value=300.0, allow_nan=False),
)
def test_local_date_and_pmc_invariant_to_source_wall_clock(
    epoch_s: int, source_zone: str, load: float
) -> None:
    """The same instant authored under any source UTC offset buckets to the same
    athlete local_date and yields an identical PMC series (ANL-T-R1.10 tz invariance)."""
    athlete = _Athlete(reference_timezone="Europe/Berlin")
    canonical = _dt.datetime.fromtimestamp(epoch_s, tz=_dt.UTC)
    as_source_wall_clock = canonical.astimezone(ZoneInfo(source_zone))

    day_utc = project_local_date(canonical, athlete)
    day_src = project_local_date(as_source_wall_clock, athlete)
    assert day_src == day_utc  # bucket depends on the instant, not its representation

    series_a = pmc({day_utc: load, day_utc + _dt.timedelta(days=1): 0.0})
    series_b = pmc({day_src: load, day_src + _dt.timedelta(days=1): 0.0})
    assert len(series_a) == len(series_b)
    for a, b in zip(series_a, series_b, strict=True):
        assert is_computed(a) == is_computed(b)
        if is_computed(a) and is_computed(b):
            assert b.value.ctl == pytest.approx(a.value.ctl, rel=1e-12)
            assert b.value.atl == pytest.approx(a.value.atl, rel=1e-12)
