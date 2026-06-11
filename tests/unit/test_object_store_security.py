"""Original-file encryption-at-rest + retention-window enforcement (RAW-T-R2(d)).

Original recording files are special-category data: at rest they are envelope-encrypted
(the bytes on disk are never the plaintext) and bounded by the configured retention
window (``retention.raw_file_days``; ``0`` retains indefinitely — the sweep itself is
:func:`wattwise_core.privacy.retention.purge_expired_original_files`, PRIV-R11.2). The
``object_ref`` stays the sha256 of the PLAINTEXT so content-addressing, the recorded
``content_hash`` round-trip, and the erasure path are identical to the plaintext store.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.domain.enums import ActivityFileFormat
from wattwise_core.persistence.models import Base
from wattwise_core.persistence.models.activity import ActivityFile
from wattwise_core.privacy.retention import purge_expired_original_files
from wattwise_core.security.crypto import EnvelopeCipher
from wattwise_core.storage import (
    EncryptedLocalObjectStore,
    LocalObjectStore,
    content_hash,
    create_object_store,
)

pytestmark = pytest.mark.unit

_PLAINTEXT = b"FIT-original-bytes: " + bytes(range(64)) * 4


def _cipher() -> EnvelopeCipher:
    return EnvelopeCipher(EnvelopeCipher.generate_root_key())


def test_encrypted_store_never_writes_plaintext_to_disk(tmp_path: Path) -> None:
    """At rest the stored object is ciphertext: the plaintext bytes appear NOWHERE on
    disk, while ``get`` round-trips them exactly (RAW-T-R2(d) encryption-at-rest)."""
    store = EncryptedLocalObjectStore(tmp_path, _cipher())
    ref = store.put(_PLAINTEXT, suffix=".fit")
    on_disk = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(on_disk) == 1
    raw = on_disk[0].read_bytes()
    assert _PLAINTEXT not in raw  # ciphertext only — never the original bytes
    assert raw != _PLAINTEXT
    assert store.get(ref) == _PLAINTEXT


def test_encrypted_ref_is_plaintext_hash_and_roundtrips(tmp_path: Path) -> None:
    """The ref stays the PLAINTEXT sha256 (+suffix): the recorded ``content_hash``
    equals the hash of the retrieved bytes (RAW-T-R2(b) under encryption)."""
    store = EncryptedLocalObjectStore(tmp_path, _cipher())
    ref = store.put(_PLAINTEXT, suffix=".fit")
    assert ref == content_hash(_PLAINTEXT) + ".fit"
    assert content_hash(store.get(ref)) == content_hash(_PLAINTEXT)


def test_encrypted_store_delete_removes_object(tmp_path: Path) -> None:
    """Erasure deletes the encrypted object too — no orphan ciphertext (RAW-T-R2(e))."""
    store = EncryptedLocalObjectStore(tmp_path, _cipher())
    ref = store.put(_PLAINTEXT)
    store.delete(ref)
    with pytest.raises(KeyError):
        store.get(ref)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def _file_row(object_ref: str, data: bytes, created_at: _dt.datetime) -> ActivityFile:
    row = ActivityFile(
        activity_id=uuid.uuid4(),
        athlete_id=uuid.uuid4(),
        object_ref=object_ref,
        format=ActivityFileFormat.FIT,
        byte_size=len(data),
        content_hash=content_hash(data),
        source_descriptor_id=uuid.uuid4(),
    )
    row.created_at = created_at
    return row


async def test_retention_sweep_composes_with_encrypted_store(tmp_path: Path) -> None:
    """The PRIV-R11.2 retention sweep purges ENCRYPTED originals too (RAW-T-R2(d)).

    ``purge_expired_original_files`` (the one boot-wired sweep) deletes the ciphertext
    object bytes and the reference row past the window, keeps a fresh original, and the
    ``0`` sentinel retains indefinitely — same contract as over the plaintext store, so
    encryption-at-rest composes with the retention window with NO second sweep.
    """
    store = EncryptedLocalObjectStore(tmp_path / "objects", _cipher())
    old_ref = store.put(_PLAINTEXT, suffix=".fit")
    new_ref = store.put(b"fresh-original", suffix=".gpx")
    now = _dt.datetime(2026, 6, 10, tzinfo=_dt.UTC)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'sweep.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        async with factory() as session:
            session.add(_file_row(old_ref, _PLAINTEXT, now - _dt.timedelta(days=40)))
            session.add(_file_row(new_ref, b"fresh-original", now - _dt.timedelta(days=1)))
            await session.commit()
            # Sentinel: 0 retains indefinitely — no sweep.
            assert (
                await purge_expired_original_files(
                    session, store, retention_days=0, now=lambda: now
                )
                == 0
            )
            purged = await purge_expired_original_files(
                session, store, retention_days=30, now=lambda: now
            )
            await session.commit()
            assert purged == 1
            with pytest.raises(KeyError):
                store.get(old_ref)
            assert store.get(new_ref) == b"fresh-original"
            remaining = (await session.execute(select(ActivityFile.object_ref))).scalars().all()
            assert remaining == [new_ref]
    finally:
        await engine.dispose()


@dataclass
class _StoreSettings:
    """The minimal settings surface ``create_object_store`` reads (duck-typed)."""

    object_store__kind: str
    object_store__local_root: Path
    encryption_root_key: object | None


class _Secret:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


def test_factory_encrypts_when_root_key_present(tmp_path: Path) -> None:
    """With an encryption root key configured (mandatory outside development), the
    factory returns the encrypted store — originals are never plaintext at rest."""
    settings = _StoreSettings("local", tmp_path, _Secret(EnvelopeCipher.generate_root_key()))
    store = create_object_store(settings)  # type: ignore[arg-type]  # duck-typed settings stub
    assert isinstance(store, EncryptedLocalObjectStore)
    ref = store.put(_PLAINTEXT)
    raw = (tmp_path / ref[:2] / ref).read_bytes()
    assert _PLAINTEXT not in raw


def test_factory_plaintext_only_in_keyless_development(tmp_path: Path) -> None:
    """Without a root key (development only — strict envs fail boot without one) the
    factory returns the plaintext local store."""
    settings = _StoreSettings("local", tmp_path, None)
    store = create_object_store(settings)  # type: ignore[arg-type]  # duck-typed settings stub
    assert type(store) is LocalObjectStore
