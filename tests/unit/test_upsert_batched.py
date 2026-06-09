"""Unit tests for the batched multi-row VALUES upsert path (PERF-R1).

PERF-R1 forbids per-row insert loops for bulk data and mandates batched upserts with a
SINGLE round-trip per batch. The seam therefore exposes a batched form that compiles a
list of row mappings into ONE multi-row ``VALUES`` insert-or-update statement.
"""

from __future__ import annotations

import pytest
from sqlalchemy.dialects import sqlite

from wattwise_core.persistence.models import DailyWellness
from wattwise_core.persistence.upsert import build_upsert


def test_batched_values_is_one_multi_row_statement() -> None:
    """A list of rows compiles to ONE multi-row VALUES insert (single round-trip; PERF-R1)."""
    table = DailyWellness.__table__
    rows = [
        {
            "daily_wellness_id": f"00000000-0000-7000-8000-00000000000{i}",
            "athlete_id": "a",
            "local_date": f"2026-06-0{i}",
            "resting_hr_bpm": 40 + i,
            "created_at": "2026-06-01T00:00:00Z",
            "updated_at": "2026-06-01T00:00:00Z",
        }
        for i in range(1, 4)
    ]
    stmt = build_upsert(
        "sqlite", table, rows, conflict_keys=["athlete_id", "local_date"], update_columns=None
    )
    compiled = stmt.compile(dialect=sqlite.dialect())
    sql = str(compiled)
    # One INSERT, three bind-parameter row groups -> a single multi-row VALUES round-trip,
    # NOT three separate statements (the prohibited per-row loop).
    assert sql.count("INSERT INTO") == 1
    params = compiled.params
    assert params["resting_hr_bpm_m0"] == 41
    assert params["resting_hr_bpm_m1"] == 42
    assert params["resting_hr_bpm_m2"] == 43
    # On conflict the real field is still refreshed; the surrogate PK / created_at are not.
    set_clause = sql.split("ON CONFLICT", 1)[1].split("DO UPDATE SET", 1)[1]
    assert "resting_hr_bpm =" in set_clause
    assert "daily_wellness_id =" not in set_clause
    assert "created_at =" not in set_clause


def test_empty_batch_is_rejected() -> None:
    """An empty batch has no column set and cannot form a VALUES clause (fail-closed)."""
    with pytest.raises(ValueError, match="empty"):
        build_upsert(
            "sqlite", DailyWellness.__table__, [], conflict_keys=["athlete_id"], update_columns=None
        )
