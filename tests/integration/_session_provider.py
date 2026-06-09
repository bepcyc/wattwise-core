"""A test ``SessionProvider`` over a bare session factory (SEAM-R11 / ARCH-R31).

The ingestion integration tests build a transactional session factory directly over an
isolated SQLite pool (a zero-arg ``async with factory() as session`` context). The
:class:`~wattwise_core.ingestion.sync.SyncOrchestrator` now takes the ONE engine-owned
:class:`~wattwise_core.seams.SessionProvider` seam, never a raw factory, so these tests
drive it THROUGH that seam via this thin adapter rather than around it.

It mirrors the OSS default :class:`~wattwise_core.seams.EngineSessionProvider`: it carries
the server-derived ``subject`` but applies NO tenant scoping (single-athlete OSS,
ARCH-R31), delegating to the wrapped factory's transactional session.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession


class FactorySessionProvider:
    """A :class:`SessionProvider` over the test session factory (the un-scoped OSS shape)."""

    def __init__(self, factory: Any) -> None:
        self._factory = factory

    @asynccontextmanager
    async def session(self, *, subject: str) -> AsyncIterator[AsyncSession]:
        """Yield the factory's transactional session; ``subject`` is carried, not scoped."""
        _ = subject  # un-scoped OSS shape (ARCH-R31); subject is carried, not scoped
        async with self._factory() as session:
            yield session
