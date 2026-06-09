"""The typed-gap reason taxonomy + lifecycle vocab (ING-GAP-R3, ING-GAP-R4).

ING-GAP-R3 mandates the gap ``reason`` taxonomy MUST AT MINIMUM include exactly these
ten members and MUST NOT include analytics-dependency reasons. ING-GAP-R4 mandates a
distinguishable open vs. closed state. These are closed canonical enums (GBO-R12); the
test fails if any mandated member is dropped/renamed (non-vacuous).
"""

from __future__ import annotations

import pytest

from wattwise_core.domain.enums import GapReason, GapState

pytestmark = pytest.mark.unit

# The verbatim ten members ING-GAP-R3 mandates AT MINIMUM.
_REQUIRED_REASONS = frozenset(
    {
        "auth_revoked",
        "needs_reauth",
        "rate_limited",
        "source_unavailable",
        "discovery_incomplete",
        "fetch_failed",
        "schema_mismatch",
        "mapping_field_missing",
        "source_removed",
        "coverage_stale",
    }
)


def test_gap_reason_taxonomy_covers_the_ten_mandated_members() -> None:
    """ING-GAP-R3: the reason taxonomy includes exactly the ten mandated members."""
    assert {r.value for r in GapReason} == _REQUIRED_REASONS


def test_gap_reason_excludes_analytics_dependency_reasons() -> None:
    """ING-GAP-R3: no analytics-dependency reason may appear in the ingestion taxonomy."""
    values = {r.value for r in GapReason}
    for forbidden in ("dependency_missing", "metric_uncomputable", "pipeline_failed"):
        assert forbidden not in values


def test_gap_state_distinguishes_open_from_closed() -> None:
    """ING-GAP-R4: a consumer can distinguish an open gap from a closed one."""
    assert {s.value for s in GapState} == {"open", "closed"}
    assert GapState.OPEN != GapState.CLOSED
