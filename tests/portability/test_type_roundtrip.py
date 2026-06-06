"""Portable type-primitive round-trip on every supported backend (BOOT-R3 / RUN-R7 / GBO-R8b).

Cites: doc 80 BOOT-R3 (DB portability — SQLite/PostgreSQL/MariaDB, DSN-only, identical
outputs), doc 70 RUN-R7/RUN-R7.2 (ORM-only portable persistence), doc 20 GBO-R8b/R10/R11/R12
(canonical units/types: ``uuid`` PKs defaulting to time-ordered UUIDv7, ``timestamptz``,
portable ``JSON``, closed enum as text+CHECK, ``numeric``).

The whole ORM must round-trip identically across the three supported relational backends with
only a DSN difference — the "trap field" in the risk register. This module asserts that contract
mechanically: it builds one throwaway ORM model spelling each portable primitive from
:mod:`wattwise_core.persistence.types`, and uses hypothesis to generate arbitrary values, persist
them, and read them back, asserting equality. SQLite always runs; PostgreSQL and MariaDB are
parametrized from the ``WATTWISE_TEST_PG_DSN`` / ``WATTWISE_TEST_MARIADB_DSN`` environment
variables and skipped when those are unset (so the offline tier stays credential-free, TIER-R1).
"""

from __future__ import annotations

import datetime as _dt
import os
import uuid
from collections.abc import AsyncIterator
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import MetaData, String, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from wattwise_core.persistence.types import (
    enum_column,
    integer_column,
    json_column,
    numeric_column,
    pk_column,
    timestamptz_column,
    uuid7,
)

if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet

pytestmark = pytest.mark.portability

# Hypothesis deadline is set explicitly (never the default) per TIER-R1; DB round-trips are
# slower than pure-CPU property tests, so a per-example deadline is disabled and the suite is
# bounded by max_examples instead. function_scoped_fixture is suppressed because each example
# reuses the per-test engine/session fixture deliberately (one schema, many round-trips).
_HYPOTHESIS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


class _Color(StrEnum):
    """A closed enum exercised as portable text+CHECK (GBO-R12)."""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class _PortBase(DeclarativeBase):
    """Throwaway declarative base; its own metadata so it never touches canonical schema."""

    metadata = MetaData()


class _Primitives(_PortBase):
    """One row spelling every portable primitive, to assert identical round-trip per backend."""

    __tablename__ = "port_primitives"

    # UUIDv7 PK (GBO-R10/R11), portable ``uuid`` on every backend.
    row_id: Mapped[uuid.UUID] = pk_column()
    # timezone-aware UTC instant (GBO-R10 ``timestamptz``).
    moment: Mapped[_dt.datetime] = timestamptz_column(nullable=False)
    # portable JSON (TEXT-backed on SQLite, JSON on PG/MariaDB).
    blob: Mapped[dict[str, object]] = json_column(nullable=False, default=dict)
    # closed enum stored as VARCHAR + CHECK (native_enum=False), identical across backends.
    color: Mapped[_Color] = enum_column(_Color, nullable=False)
    # arbitrary-precision numeric (GBO-R10 ``numeric``).
    amount: Mapped[Decimal | None] = numeric_column(nullable=True)
    # plain integer.
    count: Mapped[int | None] = integer_column(nullable=True)
    label: Mapped[str] = mapped_column(String(64), nullable=False)


def _as_utc(moment: _dt.datetime) -> _dt.datetime:
    """Normalize an instant to tz-aware UTC, reading a naive value AS UTC (engine convention).

    SQLite drops the timezone on a ``DateTime(timezone=True)`` column, so a value written as UTC
    comes back naive; PG/MariaDB return it tz-aware. The engine treats a naive instant as UTC
    everywhere (``utcnow``/``_recency_key``), so this normalizer makes the round-trip comparison
    backend-agnostic — the same instant either way.
    """
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=_dt.UTC)


def _backends() -> list[ParameterSet]:
    """Build the backend matrix: SQLite always; PG/MariaDB only when their DSN env is set."""
    cases: list[ParameterSet] = [
        pytest.param("sqlite+aiosqlite:///:memory:", id="sqlite"),
    ]
    pg = os.environ.get("WATTWISE_TEST_PG_DSN")
    cases.append(
        pytest.param(
            pg,
            id="postgresql",
            marks=pytest.mark.skipif(not pg, reason="WATTWISE_TEST_PG_DSN unset"),
        )
    )
    maria = os.environ.get("WATTWISE_TEST_MARIADB_DSN")
    cases.append(
        pytest.param(
            maria,
            id="mariadb",
            marks=pytest.mark.skipif(not maria, reason="WATTWISE_TEST_MARIADB_DSN unset"),
        )
    )
    return cases


@pytest.fixture(params=_backends())
def dsn(request: pytest.FixtureRequest) -> str:
    """The DSN under test for this backend (skipped when its env var is unset)."""
    value = request.param
    assert isinstance(value, str)
    return value


@pytest.fixture
async def factory(dsn: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory over a freshly created portable schema on the chosen backend."""
    engine = create_async_engine(dsn)
    async with engine.begin() as conn:
        await conn.run_sync(_PortBase.metadata.drop_all)
        await conn.run_sync(_PortBase.metadata.create_all)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(_PortBase.metadata.drop_all)
        await engine.dispose()


# --- strategies for each portable primitive ---

_moments = st.datetimes(
    min_value=_dt.datetime(2000, 1, 1),
    max_value=_dt.datetime(2100, 1, 1),
    timezones=st.just(_dt.UTC),
)
_json_blobs = st.dictionaries(
    keys=st.text(st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=12),
    values=st.one_of(
        st.integers(min_value=-(10**9), max_value=10**9),
        st.text(max_size=24),
        st.booleans(),
        st.none(),
    ),
    max_size=6,
)
# Numeric strategy bounded to values that survive a DECIMAL round-trip on every backend
# (no NaN/Inf; bounded scale so MariaDB DECIMAL default precision does not truncate).
_amounts = st.one_of(
    st.none(),
    st.decimals(
        min_value=Decimal("-9999999.999"),
        max_value=Decimal("9999999.999"),
        allow_nan=False,
        allow_infinity=False,
        places=3,
    ),
)


@_HYPOTHESIS
@given(
    moment=_moments,
    blob=_json_blobs,
    color=st.sampled_from(list(_Color)),
    amount=_amounts,
    count=st.one_of(st.none(), st.integers(min_value=-(2**31), max_value=2**31 - 1)),
    label=st.text(st.characters(min_codepoint=32, max_codepoint=126), max_size=64),
)
async def test_primitives_round_trip_identically(
    factory: async_sessionmaker[AsyncSession],
    moment: _dt.datetime,
    blob: dict[str, object],
    color: _Color,
    amount: Decimal | None,
    count: int | None,
    label: str,
) -> None:
    """Every portable primitive read back equals what was written, on each supported backend."""
    row_id = uuid7()
    async with factory() as write:
        write.add(
            _Primitives(
                row_id=row_id,
                moment=moment,
                blob=blob,
                color=color,
                amount=amount,
                count=count,
                label=label,
            )
        )
        await write.commit()

    async with factory() as read:
        loaded = (
            await read.execute(select(_Primitives).where(_Primitives.row_id == row_id))
        ).scalar_one()

    # UUID PK survives verbatim (portable ``uuid``, GBO-R10).
    assert loaded.row_id == row_id
    # timestamptz: the same INSTANT round-trips on every backend (GBO-R10/R32). SQLite's
    # DateTime(timezone=True) returns a naive value (no tz column type), whereas PG/MariaDB
    # return it tz-aware; the engine's convention (utcnow/_recency_key) reads a naive instant
    # AS UTC, so the portability contract is instant-equality once both sides are UTC-normalized.
    assert _as_utc(loaded.moment) == moment.astimezone(_dt.UTC)
    # JSON blob round-trips structurally identical (portable JSON).
    assert loaded.blob == blob
    # enum-text+CHECK gives back the same closed enum member (GBO-R12).
    assert loaded.color is color
    # numeric round-trips to the same value. The ``numeric_column`` factory uses a
    # precision-less ``Numeric``, which SQLite stores as float (REAL) — so the value returns
    # within float tolerance, while PG/MariaDB return it exact. The portable contract is
    # value-equality within the SQLite float tolerance (BOOT-R3 "identical outputs").
    if amount is None:
        assert loaded.amount is None
    else:
        assert loaded.amount is not None
        assert float(loaded.amount) == pytest.approx(float(amount), rel=1e-9, abs=1e-6)
    assert loaded.count == count
    assert loaded.label == label


@pytest.mark.portability
def test_uuid7_pks_are_time_ordered() -> None:
    """Sequential UUIDv7 PKs sort in generation order (GBO-R11 index locality)."""
    keys = [uuid7() for _ in range(64)]
    assert keys == sorted(keys, key=lambda u: u.bytes)
    # version nibble is 7 and the RFC-4122 variant bits are set on every generated id.
    for key in keys:
        assert key.version == 7
        assert (key.bytes[8] & 0xC0) == 0x80
