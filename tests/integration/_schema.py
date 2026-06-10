"""Shared test helper: provision the canonical schema + stamp the migration head.

The production token-issuance route persists the revocable refresh token (SEC-R2.3) and
the readiness probe gates on the stamped migration revision (RUN-R6), so an app-level
test that boots the REAL ``create_app`` needs its throwaway database to (a) carry the
ORM schema and (b) be stamped at the expected migration head — exactly what the
documented ``just migrate`` bootstrap produces on a real deployment.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from wattwise_core.persistence.base import Base
from wattwise_core.persistence.migrations_state import _ALEMBIC_VERSION, expected_head


async def provision_schema(engine: AsyncEngine) -> None:
    """Create the ORM schema and stamp ``alembic_version`` at the expected head."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(lambda sync: _ALEMBIC_VERSION.create(sync, checkfirst=True))
        head = expected_head()
        if head is not None:
            await conn.execute(insert(_ALEMBIC_VERSION).values(version_num=head))


def provision_app_schema(app: FastAPI) -> None:
    """Synchronous wrapper for app-factory tests (no running event loop)."""
    asyncio.run(provision_schema(app.state.database.engine))
