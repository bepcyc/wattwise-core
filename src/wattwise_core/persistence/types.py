"""Portable type primitives and column factories (GBO-R8b, GBO-R10, GBO-R11).

The whole ORM runs unchanged on **SQLite, PostgreSQL, and MariaDB** via a DSN-only
switch. SQLAlchemy's portable :class:`~sqlalchemy.Uuid`, ``DateTime(timezone=True)``,
:class:`~sqlalchemy.JSON`, and ``Enum(native_enum=False)`` give backend-agnostic
storage; this module wraps them in factories so every model spells a primitive the
same way and the 3-backend round-trip stays identical (the trap field per the risk
register). No vendor SQL lives here — the only sanctioned dialect branch is the
upsert seam in :mod:`wattwise_core.persistence.upsert`.
"""

from __future__ import annotations

import datetime as _dt
import os
import time
import uuid
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    TypeDecorator,
    Uuid,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

# Canonical fractional numeric: explicit precision/scale so MariaDB does NOT collapse
# an unparameterized NUMERIC to integer DECIMAL(10,0) (GBO-R8b/R10 — identical fractional
# storage across SQLite/PostgreSQL/MariaDB); ``asdecimal=False`` returns a Python ``float``
# on every backend (SQLite has no decimal), matching the ``Mapped[float]`` field hints and
# keeping value equality byte-stable across backends (GBO-AC-1).
_CANONICAL_NUMERIC = Numeric(precision=18, scale=6, asdecimal=False)

# Last-generated UUIDv7 millisecond + counter for intra-millisecond monotonicity
# (UUIDv7 PKs give index locality on append-heavy ingest, GBO-R11). Held in a
# mutable list to keep monotonic state without a `global` statement.
_uuid7_state: list[int] = [0, 0]  # [last_ms, seq]


def uuid7() -> uuid.UUID:
    """Generate a UUIDv7 (RFC 9562): 48-bit Unix-ms timestamp + random tail.

    Time-ordered, so PKs cluster on insert. A small per-millisecond sequence keeps
    values monotonic within the same millisecond. Used only for surrogate PKs —
    never for a canonical natural key, and never inside a pure mapper/resolver.
    """
    ms = int(time.time() * 1000)
    if ms == _uuid7_state[0]:
        _uuid7_state[1] = (_uuid7_state[1] + 1) & 0x0FFF
    else:
        _uuid7_state[0] = ms
        _uuid7_state[1] = 0
    rand_a = _uuid7_state[1].to_bytes(2, "big")  # 12 bits in time-low/rand_a region
    rand_b = os.urandom(8)
    raw = bytearray(ms.to_bytes(6, "big") + rand_a + rand_b)
    raw[6] = 0x70 | (raw[6] & 0x0F)  # version 7
    raw[8] = 0x80 | (raw[8] & 0x3F)  # RFC 4122 variant
    return uuid.UUID(bytes=bytes(raw))


def utcnow() -> _dt.datetime:
    """Timezone-aware UTC now (GBO-R32: instants are always tz-aware UTC)."""
    return _dt.datetime.now(_dt.UTC)


class UtcDateTime(TypeDecorator[_dt.datetime]):
    """A ``timestamptz`` that always reads back tz-aware UTC on every backend (GBO-R32).

    SQLite (and MariaDB) drop the timezone on a ``DateTime(timezone=True)`` column, so a
    naive datetime would come back from those backends while PostgreSQL returns tz-aware.
    This decorator normalizes both directions — it coerces a stored instant to UTC on the
    way in and re-attaches UTC tzinfo on the way out — so the substrate delivers an
    identical tz-aware UTC round-trip across SQLite/PostgreSQL/MariaDB without per-call-site
    fixups, never storing a local-time instant.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(
        self, value: _dt.datetime | None, dialect: Dialect
    ) -> _dt.datetime | None:
        if value is None:
            return None
        return value.astimezone(_dt.UTC) if value.tzinfo else value.replace(tzinfo=_dt.UTC)

    def process_result_value(
        self, value: _dt.datetime | None, dialect: Dialect
    ) -> _dt.datetime | None:
        if value is None:
            return None
        return value if value.tzinfo else value.replace(tzinfo=_dt.UTC)


# --- column factories (one spelling per primitive) ---


def pk_column() -> Mapped[uuid.UUID]:
    """Primary key: portable UUID, default UUIDv7 (GBO-R10/R11)."""
    return mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid7)


def fk_uuid_column(target: str, *, nullable: bool = False, index: bool = True) -> Mapped[Any]:
    """A UUID foreign-key column; FK columns are always indexed (IDX-R1)."""
    return mapped_column(Uuid(as_uuid=True), ForeignKey(target), nullable=nullable, index=index)


def timestamptz_column(*, nullable: bool = False) -> Mapped[Any]:
    """Timezone-aware UTC timestamp (``timestamptz``)."""
    return mapped_column(UtcDateTime(), nullable=nullable)


def created_at_column() -> Mapped[_dt.datetime]:
    return mapped_column(UtcDateTime(), default=utcnow, nullable=False)


def updated_at_column() -> Mapped[_dt.datetime]:
    return mapped_column(UtcDateTime(), default=utcnow, onupdate=utcnow, nullable=False)


def enum_column[E: StrEnum](enum_cls: type[E], *, nullable: bool = False, **kw: Any) -> Mapped[Any]:
    """Closed enum stored as text + CHECK constraint (GBO-R12, portable).

    ``native_enum=False`` yields ``VARCHAR + CHECK`` on every backend rather than a
    PG-native ``ENUM`` type, so the schema is identical across SQLite/PG/MariaDB.
    """
    return mapped_column(
        Enum(
            enum_cls,
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=lambda e: [m.value for m in e],
            length=64,
        ),
        nullable=nullable,
        **kw,
    )


def numeric_column(*, nullable: bool = True, **kw: Any) -> Mapped[Any]:
    return mapped_column(_CANONICAL_NUMERIC, nullable=nullable, **kw)


def integer_column(*, nullable: bool = True, **kw: Any) -> Mapped[Any]:
    return mapped_column(Integer, nullable=nullable, **kw)


def smallint_column(*, nullable: bool = True, **kw: Any) -> Mapped[Any]:
    return mapped_column(SmallInteger, nullable=nullable, **kw)


def json_column(*, nullable: bool = False, **kw: Any) -> Mapped[Any]:
    """Portable JSON column (JSON on PG/MariaDB, TEXT-backed on SQLite)."""
    return mapped_column(JSON, nullable=nullable, **kw)


__all__ = [
    "CheckConstraint",
    "created_at_column",
    "enum_column",
    "fk_uuid_column",
    "integer_column",
    "json_column",
    "numeric_column",
    "pk_column",
    "smallint_column",
    "timestamptz_column",
    "updated_at_column",
    "utcnow",
    "uuid7",
]
