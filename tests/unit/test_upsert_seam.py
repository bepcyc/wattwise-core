"""Unit tests for the dialect-aware upsert seam (UPS-R2/R3)."""

from __future__ import annotations

import pytest
from sqlalchemy.dialects import sqlite

from wattwise_core.persistence.models import DailyWellness
from wattwise_core.persistence.upsert import UnsupportedDialectError, build_upsert


def test_auto_update_excludes_pk_and_created_at() -> None:
    """The auto-derived ON CONFLICT update never clobbers the PK or created_at (UPS-R3)."""
    table = DailyWellness.__table__
    values = {
        "daily_wellness_id": "00000000-0000-7000-8000-000000000001",
        "athlete_id": "a",
        "local_date": "2026-06-01",
        "resting_hr_bpm": 48,
        "created_at": "2026-06-01T00:00:00Z",
        "updated_at": "2026-06-01T00:00:00Z",
    }
    stmt = build_upsert(
        "sqlite", table, values, conflict_keys=["athlete_id", "local_date"], update_columns=None
    )
    sql = str(stmt.compile(dialect=sqlite.dialect()))
    set_clause = sql.split("ON CONFLICT", 1)[1].split("DO UPDATE SET", 1)[1]
    assert "daily_wellness_id =" not in set_clause  # surrogate PK preserved
    assert "created_at =" not in set_clause  # insert timestamp preserved
    assert "resting_hr_bpm =" in set_clause  # a real field is refreshed
    assert "updated_at =" in set_clause


def test_unsupported_dialect_raises() -> None:
    """An unsupported backend fails closed, never silently no-ops (BOOT-R3)."""
    with pytest.raises(UnsupportedDialectError):
        build_upsert("oracle", DailyWellness.__table__, {"athlete_id": "a"}, ["athlete_id"], None)
