"""Factory ownership and lifetime of the writer queue (CKPT-R2b)."""

from __future__ import annotations

import gc
import weakref

from sqlalchemy.ext.asyncio import async_sessionmaker

from wattwise_core.agent.state_write_queue import writer_lock_for


async def test_unrelated_factories_make_independent_progress() -> None:
    """An active writer in one store does not block another store."""
    first, second = async_sessionmaker(), async_sessionmaker()
    first_lock, second_lock = writer_lock_for(first), writer_lock_for(second)
    assert writer_lock_for(first) is first_lock
    async with first_lock, second_lock:
        assert first_lock.locked() and second_lock.locked()


def test_lock_does_not_retain_factory() -> None:
    """Keeping a lock alive cannot retain an otherwise unused session factory."""
    factory = async_sessionmaker()
    reference = weakref.ref(factory)
    lock = writer_lock_for(factory)
    del factory
    gc.collect()
    assert reference() is None
    assert not lock.locked()
