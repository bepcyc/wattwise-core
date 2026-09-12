"""Offline proof that the live-success fixture has real sport and power coverage."""

import pytest

from tests.integration.test_agent_live import _REFERENCE_NOW
from tests.integration.test_agent_live import (
    live_db as live_db,  # noqa: PLC0414 - shared pytest fixture
)
from wattwise_core.agent.capabilities import gather
from wattwise_core.agent.contracts import RetrievalRequest
from wattwise_core.analytics.result import is_computed
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.identity import OWNER_ATHLETE_ID
from wattwise_core.persistence import Database

pytestmark = pytest.mark.integration


async def test_live_success_fixture_has_gatherable_load_and_power(live_db: Database) -> None:
    """API-R46b/COACH-R6: live completion starts with selected sport and usable power data."""
    async with live_db.session() as session:
        service = AnalyticsService(session)
        athlete_id = str(OWNER_ATHLETE_ID)
        assert await service.current_sport(athlete_id) == "cycling"
        params = {"from_date": "2026-05-01", "to_date": _REFERENCE_NOW.date().isoformat()}
        result = await gather(
            service,
            athlete_id,
            [
                RetrievalRequest(name, params)
                for name in ("weekly_load", "power_curve", "critical_power")
            ],
        )
    load = result.records["weekly_load"]
    assert load and all(is_computed(day) for day in load), load
    assert any(day.value.ctl > 0 for day in load)
    curve = result.records["power_curve"]
    assert curve and all(is_computed(point) for point in curve.values())
    fit = result.records["critical_power"]
    assert is_computed(fit), fit
    assert fit.value.cp_w > 0 and fit.value.w_prime_j > 0
