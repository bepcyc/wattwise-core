"""Object store for verbatim original files (RAW-R1, GBO-R8d, tier 1).

The relational store keeps only an opaque ``object_ref`` handle; the verbatim
original ``.fit``/``.gpx``/``.tcx`` bytes live in an object store and are the
source-of-truth for idempotent re-derivation (RAW-R2). The OSS default is a local
directory; an S3-compatible store is the production option. Both sit behind one
:class:`ObjectStore` protocol so the rest of the engine never branches on which.

Original files are special-category data (RAW-R4): erasure deletes the object too;
any direct download is an authenticated, signed-URL artifact owned by the API layer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

from wattwise_core.config import Settings, get_settings


def content_hash(data: bytes) -> str:
    """Stable content hash of verbatim original bytes (RAW-R1 dedup/integrity)."""
    return hashlib.sha256(data).hexdigest()


@runtime_checkable
class ObjectStore(Protocol):
    """Opaque blob storage keyed by an ``object_ref`` handle."""

    def put(self, data: bytes, *, suffix: str = "") -> str:
        """Store ``data`` and return an opaque ``object_ref`` (content-addressed)."""
        ...

    def get(self, object_ref: str) -> bytes:
        """Retrieve the bytes for ``object_ref``; raises ``KeyError`` if absent."""
        ...

    def delete(self, object_ref: str) -> None:
        """Delete the object (erasure path, RAW-R4). Idempotent."""
        ...


class LocalObjectStore:
    """Content-addressed local-filesystem object store (OSS default).

    The ref is ``<sha256>[suffix]`` and files are sharded by the first two hex
    characters to avoid huge flat directories. Content-addressing makes a
    byte-identical re-upload a no-op (FIL-R5/UPS-R3).
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, object_ref: str) -> Path:
        return self._root / object_ref[:2] / object_ref

    def put(self, data: bytes, *, suffix: str = "") -> str:
        ref = content_hash(data) + suffix
        path = self._path(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return ref

    def get(self, object_ref: str) -> bytes:
        path = self._path(object_ref)
        if not path.is_file():
            raise KeyError(object_ref)
        return path.read_bytes()

    def delete(self, object_ref: str) -> None:
        self._path(object_ref).unlink(missing_ok=True)


def create_object_store(settings: Settings | None = None) -> ObjectStore:
    """Construct the configured object store (local in OSS; S3 is a commercial seam)."""
    settings = settings or get_settings()
    if settings.object_store__kind == "local":
        return LocalObjectStore(settings.object_store__local_root)
    raise NotImplementedError(
        "the S3 object-store backend is a commercial seam; OSS ships the local store"
    )


__all__ = ["LocalObjectStore", "ObjectStore", "content_hash", "create_object_store"]
