"""The Alembic-migrated schema is exercised at the persistence boundary (DOD-R2, parity).

The contract/fuzz suites build the schema from ``Base.metadata.create_all`` (the LIVE ORM),
so a model-vs-migration drift — an enum value added to the model but not to a migration's
CHECK — is invisible to them. This suite runs the REAL Alembic chain to ``head`` and then
ingests a PWX activity END TO END (decode → pure map → :class:`IngestService` →
``activity_file`` row), so the ``activity_file.format`` CHECK and the tier-1 original-file
capture are exercised on a MIGRATED database — what production actually runs.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet

from wattwise_core.domain.enums import ActivityFileFormat, SourceKind
from wattwise_core.identity import OWNER_ATHLETE_ID
from wattwise_core.ingestion.adapters.file_upload import FileUploadAdapter, decode, native_id
from wattwise_core.ingestion.base import FetchContext, SourceDescriptorRef
from wattwise_core.ingestion.ingest import IngestService, OriginalFile
from wattwise_core.persistence.models import ActivityFile, SourceDescriptor
from wattwise_core.security.crypto import EnvelopeCipher
from wattwise_core.storage import LocalObjectStore

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[2]
_PWX = _REPO / "tests" / "contract" / "fixtures" / "file_upload" / "ride.pwx"


def _alembic_cfg(dsn: str, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Build the Alembic config bound to ``dsn`` (the env-driven ``env.py`` reads the DSN)."""
    monkeypatch.setenv("WATTWISE_APP__ENVIRONMENT", "development")
    monkeypatch.setenv("WATTWISE_DATABASE_DSN", dsn)
    monkeypatch.setenv("WATTWISE_ENCRYPTION_ROOT_KEY", EnvelopeCipher.generate_root_key())
    monkeypatch.setenv("WATTWISE_TOKEN_SIGNING_KEY", "migration-test-signing-key-0123456789")
    cfg = Config(str(_REPO / "alembic.ini"))
    cfg.set_main_option("script_location", str(_REPO / "migrations"))
    return cfg


def _upgrade_to_head(dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real Alembic migration chain to ``head`` against ``dsn``."""
    command.upgrade(_alembic_cfg(dsn, monkeypatch), "head")


async def _ingest_pwx(dsn: str, object_root: Path) -> ActivityFile | None:
    """Decode + map + ingest ``ride.pwx`` against the migrated DB; return its ActivityFile."""
    engine = create_async_engine(dsn)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        async with factory() as session:
            descriptor = (
                await session.execute(
                    select(SourceDescriptor).where(SourceDescriptor.source_key == "file_import")
                )
            ).scalar_one()  # seeded by migration 0001
            data = _PWX.read_bytes()
            asbo = decode(data, filename="ride.pwx")
            ref = SourceDescriptorRef(
                source_descriptor_id=str(descriptor.source_descriptor_id),
                source_key="file_import",
                kind=SourceKind.FILE_UPLOAD,
            )
            ctx = FetchContext(
                ingest_run_id=str(uuid.uuid4()),
                fetched_at=_dt.datetime(2026, 6, 4, 9, 0, tzinfo=_dt.UTC),
                connection_id=None,
            )
            candidates = FileUploadAdapter().map_upload(data, asbo, ref, ctx)
            original = OriginalFile(
                data=data,
                file_format=ActivityFileFormat.PWX,
                source_native_id=native_id(asbo, data),
            )
            ingest = IngestService(session, object_store=LocalObjectStore(object_root))
            await ingest.ingest(
                str(OWNER_ATHLETE_ID),
                descriptor.source_descriptor_id,
                candidates,
                original_files=[original],
            )
            await session.commit()
            return (await session.execute(select(ActivityFile))).scalars().first()
    finally:
        await engine.dispose()


def test_pwx_persists_under_the_migrated_format_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PWX import stores an ``activity_file`` with format='pwx' on a MIGRATED DB (DOD-R2).

    Catches the model/migration drift class directly: migration 0004 must widen the
    ``activity_file.format`` CHECK to accept 'pwx', or this INSERT fails closed under the
    constraint even though decode + map succeed.
    """
    dsn = f"sqlite+aiosqlite:///{tmp_path / 'migrated.db'}"
    _upgrade_to_head(dsn, monkeypatch)
    activity_file = asyncio.run(_ingest_pwx(dsn, tmp_path / "objects"))
    assert activity_file is not None, "the PWX activity_file row must persist on the migrated DB"
    assert activity_file.format is ActivityFileFormat.PWX


# --- migration <-> model parity gate (BOOT-R3) ---------------------------------------------
#
# The suite historically built every schema from ``Base.metadata.create_all`` and never ran
# the Alembic chain, so a migration that diverged from the model (an enum CHECK value, a
# column type) stayed green. This gate runs the chain to head and then ``alembic check`` —
# which fails if the migrated schema does not match the live ORM. SQLite catches STRUCTURAL
# drift; PostgreSQL/MariaDB additionally catch type drift (e.g. a tz-naive vs tz-aware
# timestamp) that SQLite's type system cannot see. PG/MariaDB run only when their throwaway
# test DSN is set, mirroring the portability suite.


def _migration_backends() -> list[ParameterSet]:
    """SQLite always; PostgreSQL/MariaDB only when their throwaway test DSN env var is set.

    Uses ``WATTWISE_PG_DSN`` / ``WATTWISE_MARIADB_DSN`` — the names the CI db-portability job
    and ``just test-db-portable`` actually export — so the PG/MariaDB legs really run in CI
    (PostgreSQL is the only backend whose ``alembic check`` can see tz-naive-vs-aware drift).
    """
    cases: list[ParameterSet] = [pytest.param(None, id="sqlite")]
    pg = os.environ.get("WATTWISE_PG_DSN")
    cases.append(
        pytest.param(pg, id="postgresql", marks=pytest.mark.skipif(not pg, reason="no PG DSN"))
    )
    maria = os.environ.get("WATTWISE_MARIADB_DSN")
    cases.append(
        pytest.param(
            maria, id="mariadb", marks=pytest.mark.skipif(not maria, reason="no MariaDB DSN")
        )
    )
    return cases


async def _reset_schema(dsn: str) -> None:
    """Drop everything in the (throwaway) test database so the chain runs from empty."""
    engine = create_async_engine(dsn)
    try:
        async with engine.begin() as conn:
            if engine.dialect.name == "postgresql":
                await conn.exec_driver_sql("DROP SCHEMA public CASCADE")
                await conn.exec_driver_sql("CREATE SCHEMA public")
            else:  # mysql / mariadb
                await conn.exec_driver_sql("SET FOREIGN_KEY_CHECKS=0")
                rows = await conn.exec_driver_sql("SHOW TABLES")
                for (table,) in rows.fetchall():
                    await conn.exec_driver_sql(f"DROP TABLE IF EXISTS `{table}`")
                await conn.exec_driver_sql("SET FOREIGN_KEY_CHECKS=1")
    finally:
        await engine.dispose()


@pytest.mark.parametrize("backend_dsn", _migration_backends())
def test_migration_chain_matches_the_model(
    backend_dsn: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Alembic chain to head matches the live ORM — ``alembic check`` finds no drift (BOOT-R3).

    Guards the whole class of model/migration drift the contract suites cannot see (they build
    from ``create_all``): a forgotten CHECK value, a column type, a tz-naive vs tz-aware column.
    """
    if backend_dsn is None:
        dsn = f"sqlite+aiosqlite:///{tmp_path / 'drift.db'}"
    else:
        dsn = backend_dsn
        asyncio.run(_reset_schema(dsn))
    cfg = _alembic_cfg(dsn, monkeypatch)
    command.upgrade(cfg, "head")
    command.check(cfg)  # raises AutogenerateDiffsDetected if the schema diverges from the model
