"""Original-file encryption-at-rest + retention-window enforcement (RAW-T-R2(d)).

Original recording files are special-category data: at rest they are envelope-encrypted
(the bytes on disk are never the plaintext) and bounded by the configured retention
window (``retention.raw_file_days``; ``0`` retains indefinitely). The ``object_ref``
stays the sha256 of the PLAINTEXT so content-addressing, the recorded ``content_hash``
round-trip, and the erasure path are identical to the plaintext store.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from wattwise_core.security.crypto import EnvelopeCipher
from wattwise_core.storage import (
    EncryptedLocalObjectStore,
    LocalObjectStore,
    content_hash,
    create_object_store,
    sweep_expired_originals,
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


def test_retention_sweep_purges_only_objects_older_than_window(tmp_path: Path) -> None:
    """A positive ``raw_file_days`` purges originals older than the window and keeps
    fresh ones (RAW-T-R2(d) retention bound is enforced, not config-only)."""
    store = LocalObjectStore(tmp_path)
    old_ref = store.put(b"old-original", suffix=".fit")
    new_ref = store.put(b"new-original", suffix=".gpx")
    now = _dt.datetime.now(_dt.UTC)
    aged = now - _dt.timedelta(days=40)
    old_path = tmp_path / old_ref[:2] / old_ref
    os.utime(old_path, (aged.timestamp(), aged.timestamp()))

    purged = sweep_expired_originals(store, retention_days=30, now=now)

    assert purged == (old_ref,)
    with pytest.raises(KeyError):
        store.get(old_ref)
    assert store.get(new_ref) == b"new-original"


def test_retention_zero_retains_indefinitely(tmp_path: Path) -> None:
    """``raw_file_days == 0`` means retain indefinitely: the sweep is a no-op."""
    store = LocalObjectStore(tmp_path)
    ref = store.put(b"keep-me", suffix=".fit")
    aged = _dt.datetime.now(_dt.UTC) - _dt.timedelta(days=3650)
    path = tmp_path / ref[:2] / ref
    os.utime(path, (aged.timestamp(), aged.timestamp()))

    assert sweep_expired_originals(store, retention_days=0) == ()
    assert store.get(ref) == b"keep-me"


def test_retention_sweep_works_on_encrypted_store(tmp_path: Path) -> None:
    """The retention bound holds for the encrypted store too (same sweep primitive)."""
    store = EncryptedLocalObjectStore(tmp_path, _cipher())
    ref = store.put(_PLAINTEXT, suffix=".fit")
    now = _dt.datetime.now(_dt.UTC)
    aged = now - _dt.timedelta(days=10)
    os.utime(tmp_path / ref[:2] / ref, (aged.timestamp(), aged.timestamp()))

    assert sweep_expired_originals(store, retention_days=7, now=now) == (ref,)
    with pytest.raises(KeyError):
        store.get(ref)


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
