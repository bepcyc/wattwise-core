"""Per-write-domain DB role isolation: the canonical write-paths are structural.

This is the write-path / role test mandated by **ARCH-R24** ("only the Ingestion/Sync
service can write the source-derived canonical store and L5-L7 credentials are read-only")
and **DEPLOY-R4** (FOUR distinct roles + the full reciprocal write-denial matrix it
requires), enforcing **ARCH-R3** at the data-access-credential level rather than merely by
import convention: source-derived canonical data is written only by the canonical-write
role, athlete-authored master-data only by the master-data-write role, the read-only role
writes nothing, and the agent-state-write role writes only the agent-state store.

It runs ONLY against a real PostgreSQL (``WATTWISE_PG_DSN``), because the guarantee is a
GRANT/REVOKE privilege check that no per-connection role exists for on SQLite/MariaDB the
same way — the clause's "distinct read-only role" is a backend role concept. (On SQLite the
same matrix is asserted structurally on the pure grant plan by the unit leg.) The admin DSN
provisions the four roles via :func:`provision_roles` against an ISOLATED, throwaway
schema (``CREATE SCHEMA ... CASCADE`` torn down at the end), so no pre-existing/host table
is read or modified (TASK data-safety): the test owns every object it touches.

The asserted matrix (DEPLOY-R4 verbatim):

* the **read-only** role rejects ALL writes (both canonical partitions + agent state);
* the **canonical-write** role writes ONLY the source-derived canonical tables and is
  rejected on the master-data tables and the agent-state store;
* the **master-data-write** role writes ONLY the master-data tables and is rejected on the
  source-derived canonical tables and the agent-state store;
* the **agent-state-write** role writes ONLY the agent-state store and is rejected on the
  canonical store (both partitions).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from wattwise_core.persistence.engine import normalize_dsn
from wattwise_core.persistence.roles import (
    DbRole,
    provision_roles,
    teardown_roles,
)

pytestmark = pytest.mark.integration

# A throwaway schema unique to this test run, so nothing host/pre-existing is touched.
_SCHEMA = f"ww_role_test_{uuid.uuid4().hex[:12]}"
_CANON_TABLE = f"{_SCHEMA}.canon_probe"
_MASTER_TABLE = f"{_SCHEMA}.master_probe"
_STATE_TABLE = f"{_SCHEMA}.state_probe"
# Distinct passwords for the provisioned, throwaway roles (test-only, torn down).
_PASSWORDS = {
    DbRole.CANONICAL_WRITE: "cw_" + uuid.uuid4().hex,
    DbRole.MASTER_DATA_WRITE: "md_" + uuid.uuid4().hex,
    DbRole.READ_ONLY: "ro_" + uuid.uuid4().hex,
    DbRole.AGENT_STATE_WRITE: "as_" + uuid.uuid4().hex,
}


def _admin_dsn() -> str | None:
    return os.environ.get("WATTWISE_PG_DSN")


def _role_dsn(admin_dsn: str, role: DbRole, role_name: str) -> str:
    """The admin DSN re-pointed at one provisioned role's credentials (same host/db)."""
    url = make_url(normalize_dsn(admin_dsn))
    url = url.set(username=role_name, password=_PASSWORDS[role])
    return url.render_as_string(hide_password=False)


@pytest_asyncio.fixture
async def provisioned() -> AsyncIterator[dict[DbRole, str]]:
    """Provision the four roles + an isolated schema with one probe table per partition.

    Yields ``{role: role_name}``. On teardown drops the schema and the roles so the test
    leaves the throwaway database exactly as it found it.
    """
    admin_dsn = _admin_dsn()
    assert admin_dsn is not None  # guarded by skipif on the test
    admin = create_async_engine(normalize_dsn(admin_dsn), isolation_level="AUTOCOMMIT")
    role_names = {role: f"{_SCHEMA}_{role.value}" for role in DbRole}
    try:
        async with admin.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{_SCHEMA}"'))
            for table in (_CANON_TABLE, _MASTER_TABLE, _STATE_TABLE):
                await conn.execute(
                    text(f"CREATE TABLE {table} (id integer PRIMARY KEY, val text)")
                )
        # Provision the four roles + grant the per-domain privileges (the unit under test).
        await provision_roles(
            admin,
            schema=_SCHEMA,
            canonical_tables=[_CANON_TABLE],
            master_data_tables=[_MASTER_TABLE],
            agent_state_tables=[_STATE_TABLE],
            role_names=role_names,
            passwords=_PASSWORDS,
        )
        yield role_names
    finally:
        await teardown_roles(admin, schema=_SCHEMA, role_names=role_names)
        await admin.dispose()


async def _connect(admin_dsn: str, role: DbRole, role_name: str) -> AsyncEngine:
    return create_async_engine(_role_dsn(admin_dsn, role, role_name))


async def _insert_ok(engine: AsyncEngine, table: str, row_id: int) -> None:
    async with engine.begin() as conn:
        await conn.execute(text(f"INSERT INTO {table} (id, val) VALUES ({row_id}, 'x')"))


async def _insert_denied(engine: AsyncEngine, table: str, row_id: int) -> None:
    with pytest.raises((DBAPIError, ProgrammingError)):
        async with engine.begin() as conn:
            await conn.execute(text(f"INSERT INTO {table} (id, val) VALUES ({row_id}, 'x')"))


@pytest.mark.skipif(_admin_dsn() is None, reason="no PG DSN (WATTWISE_PG_DSN unset)")
async def test_full_four_role_reciprocal_denial_matrix(
    provisioned: dict[DbRole, str],
) -> None:
    """ARCH-R3/ARCH-R24/DEPLOY-R4: the FULL 4-role x 3-partition write matrix, live on PG.

    Each write-owning role INSERTs successfully into ITS OWN partition; every other
    role-partition write is REFUSED at the privilege level (a DB error, not a silent no-op).
    Reads outside the write domain stay possible on the canonical partitions (ARCH-R3
    "read-only outside the write domain"), and the read-only role can read both.
    """
    admin_dsn = _admin_dsn()
    assert admin_dsn is not None
    engines = {role: await _connect(admin_dsn, role, provisioned[role]) for role in DbRole}
    cw = engines[DbRole.CANONICAL_WRITE]
    mdw = engines[DbRole.MASTER_DATA_WRITE]
    ro = engines[DbRole.READ_ONLY]
    asw = engines[DbRole.AGENT_STATE_WRITE]
    try:
        # Positive diagonal: each role writes its OWN partition.
        await _insert_ok(cw, _CANON_TABLE, 1)
        await _insert_ok(mdw, _MASTER_TABLE, 1)
        await _insert_ok(asw, _STATE_TABLE, 1)

        # Read-only role: SELECT works on BOTH canonical partitions...
        async with ro.connect() as conn:
            canon = (await conn.execute(text(f"SELECT val FROM {_CANON_TABLE}"))).scalars()
            assert canon.all() == ["x"]
            master = (await conn.execute(text(f"SELECT val FROM {_MASTER_TABLE}"))).scalars()
            assert master.all() == ["x"]
        # ...but ALL writes are refused (canonical, master-data, agent-state).
        await _insert_denied(ro, _CANON_TABLE, 90)
        await _insert_denied(ro, _MASTER_TABLE, 90)
        await _insert_denied(ro, _STATE_TABLE, 90)

        # Canonical-write role: refused on master-data and the agent-state store; can still
        # READ the master-data partition (read-only outside its write domain).
        await _insert_denied(cw, _MASTER_TABLE, 91)
        await _insert_denied(cw, _STATE_TABLE, 91)
        async with cw.connect() as conn:
            rows = (await conn.execute(text(f"SELECT val FROM {_MASTER_TABLE}"))).scalars()
            assert rows.all() == ["x"]

        # Master-data-write role: refused on source-derived canonical and the agent-state
        # store; can still READ the source-derived partition.
        await _insert_denied(mdw, _CANON_TABLE, 92)
        await _insert_denied(mdw, _STATE_TABLE, 92)
        async with mdw.connect() as conn:
            rows = (await conn.execute(text(f"SELECT val FROM {_CANON_TABLE}"))).scalars()
            assert rows.all() == ["x"]

        # Agent-state-write role: refused on the canonical store, BOTH partitions.
        await _insert_denied(asw, _CANON_TABLE, 93)
        await _insert_denied(asw, _MASTER_TABLE, 93)
    finally:
        for engine in engines.values():
            await engine.dispose()
