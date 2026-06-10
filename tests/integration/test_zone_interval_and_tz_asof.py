"""Effective-dated closure on the write path — GBO-R13d zones + GBO-R34 tz as-of.

Two effective-dated-closure gaps on the canonical write path:

* GBO-R13d — training-zone intervals for a given ``(athlete_id, zone_kind, sport)`` MUST
  NOT overlap and at most ONE row may stay open (``effective_to IS NULL``). Opening a new
  effective interval MUST close the prior open one (set its ``effective_to``), so a future
  re-zoning is a data action and a past date reproduces the zones then in effect.
* GBO-R34 — changing the athlete reference timezone MUST stamp
  ``reference_timezone_effective_from`` (the as-of metadata) so prior days are not
  retroactively re-bucketed under the new tz. This is the real consumer of the field.

Runs on in-memory SQLite (the portable substrate, GBO-R8b); no concurrency under test.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.api.auth import Principal, Scope, authenticate
from wattwise_core.api.errors import install_error_handlers
from wattwise_core.api.ratelimit import RateLimiter
from wattwise_core.api.routers import athlete as athlete_router
from wattwise_core.api.routers import user_settings as settings_router
from wattwise_core.domain.enums import ZoneBasis, ZoneKind
from wattwise_core.persistence.models import Athlete, Base, Sport, TrainingZoneSet

pytestmark = pytest.mark.integration

UTC = _dt.UTC


@dataclass
class Env:
    """The wired app + its client/session/owner for one seeded scenario."""

    client: AsyncClient
    session: AsyncSession
    athlete_id: str


@pytest_asyncio.fixture
async def seeded() -> AsyncIterator[Env]:
    """An app mounting the zones + athlete routers over one seeded owner (UTC, no eff)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        athlete = Athlete(sex="male", reference_timezone="UTC")
        session.add(athlete)
        await session.flush()
        athlete_id = str(athlete.athlete_id)
        await session.commit()
        app = _build_app(session, athlete_id)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            yield Env(client, session, athlete_id)
    await engine.dispose()


def _build_app(session: AsyncSession, athlete_id: str) -> FastAPI:
    app = FastAPI()
    app.state.rate_limiter = RateLimiter()
    install_error_handlers(app)
    app.include_router(settings_router.router)
    app.include_router(athlete_router.router)
    overrides = {
        authenticate: lambda: Principal(subject=athlete_id, scopes=frozenset(Scope)),
        settings_router.require_read_scope: lambda: None,
        settings_router.require_write_scope: lambda: None,
        settings_router.current_athlete_id: lambda: athlete_id,
        settings_router.current_session: lambda: session,
        athlete_router.require_read_scope: lambda: None,
        athlete_router.require_write_scope: lambda: None,
        athlete_router.current_athlete_id: lambda: athlete_id,
        athlete_router.current_session: lambda: session,
    }
    app.dependency_overrides.update(overrides)
    return app


# --- GBO-R13d: opening a new zone interval closes the prior open one ----------------


async def test_new_zone_interval_closes_prior_open_interval(seeded: Env) -> None:
    """A new zone set closes the prior open interval — never two open rows (GBO-R13d).

    Seed a power zone set effective YESTERDAY (open, ``effective_to IS NULL``). A PUT writes
    a today-effective set; the prior interval MUST be closed (``effective_to`` set), leaving
    at most ONE open row for ``(athlete_id, power, all-sports)``.
    """
    yesterday = _dt.datetime.now(tz=UTC).date() - _dt.timedelta(days=1)
    seeded.session.add(
        TrainingZoneSet(
            athlete_id=uuid.UUID(seeded.athlete_id),
            zone_kind=ZoneKind.POWER,
            effective_date=yesterday,
            basis=ZoneBasis.ABSOLUTE,
            boundaries=[{"zone_index": 1, "label": "Z1", "lower": 0, "upper": 150}],
        )
    )
    await seeded.session.flush()

    put = await seeded.client.put(
        "/v1/user-settings/zones",
        json={
            "kind": "power",
            "basis": "absolute",
            "boundaries": [{"zone_index": 1, "label": "Z1", "lower": 0, "upper": 160}],
        },
    )
    assert put.status_code == 200

    rows = (
        (
            await seeded.session.execute(
                select(TrainingZoneSet).where(TrainingZoneSet.zone_kind == ZoneKind.POWER)
            )
        )
        .scalars()
        .all()
    )
    open_rows = [r for r in rows if r.effective_to is None]
    assert len(open_rows) == 1  # at most one open interval (GBO-R13d)
    closed = [r for r in rows if r.effective_to is not None]
    assert len(closed) == 1  # the prior interval was closed, not left overlapping
    assert closed[0].effective_date == yesterday


# --- GBO-R34: changing the reference timezone stamps the as-of effective_from --------


async def test_changing_reference_timezone_stamps_effective_from(seeded: Env) -> None:
    """A reference-timezone change records ``reference_timezone_effective_from`` (GBO-R34).

    The athlete starts with ``effective_from = NULL``. Changing the tz to Europe/Berlin via
    PUT must set ``reference_timezone_effective_from`` (the as-of metadata) so prior days are
    not retroactively re-bucketed under the new zone. The field was previously orphaned.
    """
    before = _dt.datetime.now(tz=UTC)
    resp = await seeded.client.put("/v1/athlete", json={"reference_timezone": "Europe/Berlin"})
    assert resp.status_code == 200
    after = _dt.datetime.now(tz=UTC)

    owner = (
        await seeded.session.execute(
            select(Athlete).where(Athlete.athlete_id == uuid.UUID(seeded.athlete_id))
        )
    ).scalar_one()
    assert owner.reference_timezone == "Europe/Berlin"
    eff = owner.reference_timezone_effective_from
    assert eff is not None  # the as-of metadata was stamped (real consumer, GBO-R34)
    assert before <= eff.astimezone(UTC) <= after


async def test_unchanged_reference_timezone_does_not_restamp(seeded: Env) -> None:
    """Re-PUTting the SAME timezone does not move effective_from (stable as-of, GBO-R34)."""
    await seeded.client.put("/v1/athlete", json={"reference_timezone": "Europe/Berlin"})
    owner = (
        await seeded.session.execute(
            select(Athlete).where(Athlete.athlete_id == uuid.UUID(seeded.athlete_id))
        )
    ).scalar_one()
    first_eff = owner.reference_timezone_effective_from
    assert first_eff is not None
    # Re-PUT the same tz: effective_from must NOT advance (only a real change re-stamps).
    await seeded.client.put("/v1/athlete", json={"reference_timezone": "Europe/Berlin"})
    await seeded.session.refresh(owner)
    assert owner.reference_timezone_effective_from == first_eff
