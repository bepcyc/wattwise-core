"""Declarative base, naming conventions, and shared mixins (GBO-R8b, GBO-R9, GBO-R11).

All identifiers are lowercase ``snake_case`` (GBO-R9). A deterministic naming
convention for indexes/constraints keeps Alembic autogenerate stable and identical
across SQLite/PostgreSQL/MariaDB.
"""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase, Mapped

from wattwise_core.persistence.types import (
    created_at_column,
    pk_column,
    updated_at_column,
)

# Deterministic constraint/index naming (portable, stable autogenerate).
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for every canonical ORM model."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    """Server-set UTC ``created_at`` / ``updated_at`` on every record (GBO-R11)."""

    created_at: Mapped[_dt.datetime] = created_at_column()
    updated_at: Mapped[_dt.datetime] = updated_at_column()


class UuidPkMixin:
    """Surrogate UUIDv7 primary key named after the table's domain noun.

    Concrete models declare their own named PK (e.g. ``activity_id``) for clarity;
    this mixin is used where a generic ``id`` is acceptable.
    """

    id: Mapped[uuid.UUID] = pk_column()


__all__ = ["NAMING_CONVENTION", "Base", "TimestampMixin", "UuidPkMixin"]
