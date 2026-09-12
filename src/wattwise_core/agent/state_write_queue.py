"""Factory-owned admission for saver writes (CKPT-R2b, ARCH-R13).

A graph schedules checkpoint and pending-write persistence concurrently. Savers for separate
requests share a state-store session factory; their short transactions must take turns rather
than starve one another inside SQLite's busy wait. The same policy bounds checkpoint writers
on server backends while graph nodes and reads remain concurrent.

One factory belongs to one event loop. This queue coordinates that factory's saver operations,
not independent engines or processes. Weak keys keep factory lifetime authoritative; the lock
value does not reference its factory. Database constraints remain the cross-process authority.
"""

from __future__ import annotations

import asyncio
from weakref import WeakKeyDictionary

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_WRITERS: WeakKeyDictionary[async_sessionmaker[AsyncSession], asyncio.Lock] = WeakKeyDictionary()


def writer_lock_for(factory: async_sessionmaker[AsyncSession]) -> asyncio.Lock:
    """Return the shared FIFO admission lock without retaining the factory (CKPT-R2b)."""
    lock = _WRITERS.get(factory)
    if lock is None:
        lock = asyncio.Lock()
        _WRITERS[factory] = lock
    return lock
