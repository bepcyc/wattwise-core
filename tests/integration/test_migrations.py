"""The Alembic-migrated schema is exercised at the persistence boundary (DOD-R2, parity).

The contract/fuzz suites build the schema from ``Base.metadata.create_all`` (the LIVE ORM),
so a model-vs-migration drift — an enum value added to the model but not to a migration's
CHECK — is invisible to them. This suite runs the REAL Alembic chain to ``head`` and then
ingests a TCX activity END TO END (decode → pure map → :class:`IngestService` →
``activity_file`` row), so the ``activity_file.format`` CHECK and the tier-1 original-file
capture are exercised on a MIGRATED database — what production actually runs. The CHECK is
the SPEC-CLOSED 5-member ``fit|gpx|tcx|json|other`` set (SCHEMA-R3 / API-R33) — migration
0011 narrows it back after the since-reverted ``pwx`` widening of 0004.
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

from wattwise_core.config import get_settings
from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import ActivityFileFormat, Fidelity, SourceKind
from wattwise_core.identity import OWNER_ATHLETE_ID
from wattwise_core.ingestion.adapters.file_upload import FileUploadAdapter, decode, native_id
from wattwise_core.ingestion.base import FetchContext, SourceDescriptorRef
from wattwise_core.ingestion.ingest import IngestService, OriginalFile
from wattwise_core.persistence.models import (
    Activity,
    ActivityFile,
    AthleteSourcePreference,
    SourceDescriptor,
)
from wattwise_core.security.crypto import EnvelopeCipher
from wattwise_core.storage import LocalObjectStore, content_hash

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[2]
_TCX = _REPO / "tests" / "contract" / "fixtures" / "file_upload" / "ride.tcx"


def _alembic_cfg(dsn: str, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Build the Alembic config bound to ``dsn`` (the env-driven ``env.py`` reads the DSN)."""
    monkeypatch.setenv("WATTWISE_APP__ENVIRONMENT", "development")
    monkeypatch.setenv("WATTWISE_DATABASE_DSN", dsn)
    monkeypatch.setenv("WATTWISE_ENCRYPTION_ROOT_KEY", EnvelopeCipher.generate_root_key())
    monkeypatch.setenv("WATTWISE_TOKEN_SIGNING_KEY", "migration-test-signing-key-0123456789")
    # ``env.py`` resolves the DSN through the lru_cached ``get_settings``; clear it so each
    # upgrade in the same process rebinds to THIS test's DSN (a second in-process upgrade test
    # would otherwise inherit the first test's cached settings and migrate the wrong database).
    get_settings.cache_clear()
    cfg = Config(str(_REPO / "alembic.ini"))
    cfg.set_main_option("script_location", str(_REPO / "migrations"))
    return cfg


def _upgrade_to_head(dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real Alembic migration chain to ``head`` against ``dsn``."""
    command.upgrade(_alembic_cfg(dsn, monkeypatch), "head")


async def _ingest_tcx(dsn: str, object_root: Path) -> ActivityFile | None:
    """Decode + map + ingest ``ride.tcx`` against the migrated DB; return its ActivityFile."""
    engine = create_async_engine(dsn)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        async with factory() as session:
            descriptor = (
                await session.execute(
                    select(SourceDescriptor).where(SourceDescriptor.source_key == "file_import")
                )
            ).scalar_one()  # seeded by migration 0001
            data = _TCX.read_bytes()
            asbo = decode(data, filename="ride.tcx")
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
                file_format=ActivityFileFormat.TCX,
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


def test_tcx_persists_under_the_migrated_format_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A TCX import stores an ``activity_file`` with format='tcx' on a MIGRATED DB (DOD-R2).

    Catches the model/migration drift class directly: the live ORM enum and the migrated
    ``activity_file.format`` CHECK must agree on the spec-closed 5-member set
    (``fit|gpx|tcx|json|other``, SCHEMA-R3) or this INSERT fails closed under the
    constraint even though decode + map succeed.
    """
    dsn = f"sqlite+aiosqlite:///{tmp_path / 'migrated.db'}"
    _upgrade_to_head(dsn, monkeypatch)
    activity_file = asyncio.run(_ingest_tcx(dsn, tmp_path / "objects"))
    assert activity_file is not None, "the TCX activity_file row must persist on the migrated DB"
    assert activity_file.format is ActivityFileFormat.TCX


# --- per-athlete source override on the MIGRATED schema (PRV-R7, ARCH-P2-03) ----------------
#
# test_ingestion_findings proves the resolver honours an ``athlete_source_preference`` override
# on a ``create_all`` schema; that never inserts the row through the REAL migration 0005 table
# — its UNIQUE/FK/CHECK constraints, its named Fidelity CHECK, the enum-text round-trip. This
# exercises the WHOLE override path end to end on the Alembic-migrated DB: INSERT the row, then
# re-resolve through ``IngestService`` and prove the winning source for a contested field flips.

_RIDE_START = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=_dt.UTC)


def _ride(*, native_id: str, watts: float, tier: Fidelity, seconds: int = 3600) -> GboCandidate:
    """A constant-power cycling candidate (same start/sport/duration ⇒ one identity)."""
    payload = {
        "start_time": _RIDE_START,
        "sport": "cycling",
        "elapsed_time_s": seconds,
        "moving_time_s": seconds,
        "avg_power_w": watts,
        "streams": {
            "power_w": {"values": [watts] * seconds, "sample_basis": "time", "sample_rate_hz": 1.0}
        },
    }
    return GboCandidate(
        gbo_type="activity",
        source_descriptor_id="placeholder",
        source_native_id=native_id,
        content_hash=content_hash(f"{native_id}:{watts}".encode()),
        payload=payload,
        trust_tier=tier,
        fetched_at=_dt.datetime(2026, 6, 1, 9, 0, tzinfo=_dt.UTC),
    )


async def _override_flips_winner(dsn: str) -> tuple[float, float]:
    """On the migrated DB: contest avg_power_w across two sources; return (before, after).

    ``file_import`` (RAW_STREAM, 200 W; seeded by migration 0001) out-ranks a second
    PLATFORM_COMPUTED source (320 W) by adapter tier, so the no-config winner is 200 W.
    Then INSERT an ``athlete_source_preference`` row demoting file_import's ``avg_power_w``
    to SUMMARY_ONLY and RE-INGEST: ``load_trust_policy`` reads that row and the winner must
    flip to the 320 W source — proving the migrated table's row drives real resolution.
    """
    engine = create_async_engine(dsn)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        async with factory() as session:
            athlete_id = OWNER_ATHLETE_ID  # seeded by migration 0001
            file_src = (
                await session.execute(
                    select(SourceDescriptor).where(SourceDescriptor.source_key == "file_import")
                )
            ).scalar_one()  # seeded by migration 0001
            api_src = SourceDescriptor(
                source_key="platform_api", display_name="Platform API", kind="oauth_api"
            )
            session.add(api_src)
            await session.flush()

            ingest = IngestService(session)
            await ingest.ingest(
                str(athlete_id),
                file_src.source_descriptor_id,
                [_ride(native_id="file-1", watts=200.0, tier=Fidelity.RAW_STREAM)],
            )
            await ingest.ingest(
                str(athlete_id),
                api_src.source_descriptor_id,
                [_ride(native_id="api-1", watts=320.0, tier=Fidelity.PLATFORM_COMPUTED)],
            )
            await session.commit()
            before = float((await session.execute(select(Activity))).scalars().one().avg_power_w)

            # INSERT the override into the REAL migrated table, demoting file_import (the
            # current winner) so api_src now out-ranks it. This row goes through migration
            # 0005's FK -> athlete + source, its named Fidelity CHECK, and the
            # (athlete, source, channel) UNIQUE at runtime.
            session.add(
                AthleteSourcePreference(
                    athlete_id=athlete_id,
                    source_descriptor_id=file_src.source_descriptor_id,
                    channel="avg_power_w",
                    trust_tier=Fidelity.SUMMARY_ONLY,
                )
            )
            await session.commit()

            # ... then RE-INGEST file_import: resolution re-reads the override row and flips.
            await ingest.ingest(
                str(athlete_id),
                file_src.source_descriptor_id,
                [_ride(native_id="file-1", watts=200.0, tier=Fidelity.RAW_STREAM)],
            )
            await session.commit()
            after = float((await session.execute(select(Activity))).scalars().one().avg_power_w)
            return before, after
    finally:
        await engine.dispose()


def test_athlete_source_override_flips_winner_on_migrated_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A migrated ``athlete_source_preference`` row flips the contested-field winner (PRV-R7).

    End-to-end on the Alembic-migrated SQLite schema (NOT ``create_all``): two sources contest
    ``avg_power_w``; the RAW_STREAM source wins by adapter tier (200 W). After INSERTing a real
    override row through migration 0005's table — exercising its FK to athlete/source, its named
    Fidelity CHECK, and the (athlete, source, channel) UNIQUE at runtime — the resolver honours
    it and the 320 W source wins. Proves the override path works against the production schema,
    not just the live-ORM ``create_all`` schema the rest of the suite builds.
    """
    dsn = f"sqlite+aiosqlite:///{tmp_path / 'override.db'}"
    _upgrade_to_head(dsn, monkeypatch)
    before, after = asyncio.run(_override_flips_winner(dsn))
    assert before == pytest.approx(200.0), "no override -> RAW_STREAM source wins by adapter tier"
    assert after == pytest.approx(320.0), "the migrated override row flips the winner to api_src"


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
