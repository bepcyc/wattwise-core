"""Regression tests pinning the file-upload mapping fixes (MAP-R4/R2).

The file-upload adapter must translate a source ``sub_sport`` token through a vocab to
a canonical registry code (or ``None``), never echo the raw lowercased source token
into the canonical FK field (MAP-R4/MAP-R2) — mirroring the Intervals adapter.
"""

from __future__ import annotations

from wattwise_core.ingestion.adapters._map_activity import _sub_sport


def test_sub_sport_maps_known_token_to_registry_code() -> None:
    """A known source sub_sport token maps to a seeded registry code (MAP-R4)."""
    assert _sub_sport("gravel") == "cycling_other"
    assert _sub_sport("Trail") == "running_other"
    assert _sub_sport("open_water") == "swimming_other"


def test_sub_sport_unknown_token_is_typed_gap_not_passthrough() -> None:
    """An unmapped/absent token yields None, never the raw lowercased token (MAP-R2)."""
    assert _sub_sport("road_bike_with_unknown_suffix") is None
    assert _sub_sport("generic") is None
    assert _sub_sport("all") is None
    assert _sub_sport("") is None
    assert _sub_sport(None) is None
    assert _sub_sport(42) is None  # type: ignore[arg-type]
