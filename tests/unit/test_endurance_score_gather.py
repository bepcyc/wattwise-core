"""Unit tests for the endurance-score service gather's fail-closed seams (ES-R2).

The gather (:func:`wattwise_core.analytics._service_loaders._gather_endurance_score`)
is exercised against a stubbed service so the defensive branches a healthy canonical
store never reaches — training history present but NO computable PMC day — are still
proven to fail closed (typed ``Unavailable``, never a fabricated CTL).
"""

from __future__ import annotations

import datetime as _dt
from typing import cast

import pytest

from wattwise_core.analytics._service_es import _gather_endurance_score
from wattwise_core.analytics.result import Unavailable, UnavailableReason
from wattwise_core.analytics.service import AnalyticsService


class _StubSvc:
    """Duck-typed service: history exists but PMC yields no computable day."""

    async def _earliest_activity_date(self, athlete_id: str) -> _dt.date:
        return _dt.date(2026, 1, 1)

    async def pmc(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date
    ) -> list[object]:
        return []  # no computable PMC day

    async def current_sport(self, athlete_id: str) -> None:
        return None


@pytest.mark.unit
async def test_gather_no_computable_pmc_day_fails_closed() -> None:
    """ES-R2(a): history without a computable PMC day ⇒ Unavailable(MISSING_REQUIRED_INPUT)."""
    svc = cast(AnalyticsService, _StubSvc())
    result = await _gather_endurance_score(svc, "athlete-x", _dt.date(2026, 6, 1))
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.MISSING_REQUIRED_INPUT
    assert "CTL" in result.detail or "PMC" in result.detail
