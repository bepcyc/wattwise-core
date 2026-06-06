"""Adapter registry — entry-point discovery of pluggable sources (ADP-R*, ROAD-R6).

Loads the installed ``wattwise_core.adapters`` entry points into typed
:class:`~wattwise_core.ingestion.base.SourceAdapter` instances, indexed by each
adapter's own ``source_key``. Adding a source is exactly **one adapter + one
entry-point line** (ROAD-R6): no consumer, sync, analytics, or agent change, and
NO source-name branch anywhere outside the adapter itself (ARCH-R2). The registry
discovers adapters by *data* (the entry-point group), never by importing a named
adapter module, so consumers select an adapter via :meth:`AdapterRegistry.get`
(by ``source_key``) and stay source-blind (Principle A).

Fail-closed (CLI-R2 / ADP-R*): an entry point that does not load, does not
instantiate, or whose object does not satisfy the :class:`SourceAdapter` Protocol
is rejected with a typed :class:`AdapterRegistryError` rather than silently
skipped — an unusable source must surface, never degrade to a wrong-but-plausible
no-op. A lookup for an unregistered ``source_key`` raises
:class:`UnknownSourceError` (the caller decides whether that is fatal).

Layer: this is the L3 ingestion registry (a neutral, source-agnostic adapter
module, NOT source-specific); it imports only the rankless adapter contract
(:mod:`wattwise_core.ingestion.base`) and the standard library.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Final

from wattwise_core.ingestion.base import SourceAdapter

#: The entry-point group every pluggable source registers under (ROAD-R6).
ADAPTER_ENTRY_POINT_GROUP: Final = "wattwise_core.adapters"


class AdapterRegistryError(RuntimeError):
    """An entry point could not be turned into a usable adapter (fail-closed, ADP-R*).

    Raised when an entry point fails to load/import, cannot be instantiated, or
    yields an object that does not satisfy the :class:`SourceAdapter` Protocol.
    Carries the offending entry-point name so the operator can fix the packaging
    without the registry guessing past a broken source.
    """


class UnknownSourceError(LookupError):
    """No registered adapter matches the requested ``source_key`` (fail-closed).

    Distinct from :class:`AdapterRegistryError`: the registry loaded cleanly, but
    the caller asked for a source that is not installed. The caller decides
    whether that is an error (an explicit sync request) or a skip.
    """


@dataclass(frozen=True, slots=True)
class AdapterRegistry:
    """Immutable lookup of installed source adapters, keyed by ``source_key``.

    Built once per process via :func:`load_registry`; treated as read-only config.
    """

    _by_source_key: dict[str, SourceAdapter]

    def get(self, source_key: str) -> SourceAdapter:
        """Return the adapter for ``source_key`` or raise :class:`UnknownSourceError`.

        The ONE sanctioned way a consumer obtains an adapter — by registered source
        identity, never by importing a named adapter module (ARCH-R2).
        """
        adapter = self._by_source_key.get(source_key)
        if adapter is None:
            raise UnknownSourceError(source_key)
        return adapter

    def has(self, source_key: str) -> bool:
        """True when an adapter is registered for ``source_key``."""
        return source_key in self._by_source_key

    def source_keys(self) -> tuple[str, ...]:
        """Every registered ``source_key`` (sorted; stable for deterministic listing)."""
        return tuple(sorted(self._by_source_key))


def _instantiate(entry_point: EntryPoint) -> SourceAdapter:
    """Load + instantiate one entry point into a verified adapter (fail-closed)."""
    try:
        factory = entry_point.load()
    except Exception as exc:  # any import/attr failure is fatal (fail-closed, ADP-R*)
        raise AdapterRegistryError(
            f"adapter entry point {entry_point.name!r} failed to load"
        ) from exc
    try:
        adapter = factory()
    except Exception as exc:  # a non-constructable adapter is fatal (fail-closed)
        raise AdapterRegistryError(
            f"adapter entry point {entry_point.name!r} could not be instantiated"
        ) from exc
    if not isinstance(adapter, SourceAdapter):
        raise AdapterRegistryError(
            f"adapter entry point {entry_point.name!r} does not satisfy SourceAdapter"
        )
    return adapter


def load_registry(*, group: str = ADAPTER_ENTRY_POINT_GROUP) -> AdapterRegistry:
    """Discover + load every installed adapter into an :class:`AdapterRegistry`.

    Indexes by each adapter's own ``source_key`` (the authoritative source
    identity), not the entry-point name, so a registry built from packaging data
    matches the ``source_key`` carried on connections and descriptors. Two
    adapters claiming the same ``source_key`` is a packaging error and fails closed
    (ADP-R*) rather than silently letting one shadow the other.
    """
    by_key: dict[str, SourceAdapter] = {}
    for entry_point in entry_points(group=group):
        adapter = _instantiate(entry_point)
        key = adapter.source_key
        if key in by_key:
            raise AdapterRegistryError(
                f"duplicate adapter source_key {key!r} (entry point {entry_point.name!r})"
            )
        by_key[key] = adapter
    return AdapterRegistry(_by_source_key=by_key)


def registry_from_adapters(adapters: Iterable[object]) -> AdapterRegistry:
    """Build a registry from explicit adapter instances (tests / embedding, ADP-R17).

    The offline counterpart to :func:`load_registry`: no entry-point scan, so a
    fake adapter can be registered without touching installed packaging. Each object
    is runtime-checked against the :class:`SourceAdapter` Protocol (fail-closed, the
    same gate :func:`_instantiate` applies to an entry-point object) — adapters
    legitimately declare their identity metadata as class attributes, so the gate is
    a runtime ``isinstance`` not a static assignment. Same duplicate-``source_key``
    guard as :func:`load_registry`.
    """
    by_key: dict[str, SourceAdapter] = {}
    for adapter in adapters:
        if not isinstance(adapter, SourceAdapter):
            raise AdapterRegistryError("object does not satisfy SourceAdapter")
        key = adapter.source_key
        if key in by_key:
            raise AdapterRegistryError(f"duplicate adapter source_key {key!r}")
        by_key[key] = adapter
    return AdapterRegistry(_by_source_key=by_key)


__all__ = [
    "ADAPTER_ENTRY_POINT_GROUP",
    "AdapterRegistry",
    "AdapterRegistryError",
    "UnknownSourceError",
    "load_registry",
    "registry_from_adapters",
]
