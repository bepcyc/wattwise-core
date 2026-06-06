"""Abstract seam-contract base-classes (GOLD-R5 §6.3a, OSS deliverable).

Each base-class encodes the invariants of one extension seam as reusable, impl-agnostic
pytest cases. A conforming implementation subclasses it and provides the impl via an
abstract fixture/method; the inherited cases then assert the seam's contract. The OSS
default implementations subclass these so the shipped contracts are real and green
against the bare OSS product.

This module ships only the contracts for the seams whose interfaces are stable OSS
deliverables: the source-adapter seam (ADP-R*), the dedup/conflict resolver seam
(CONF-R7/DEDUP-R6), and the entitlement seam (ENT-R*/DELIV-R6). The MemoryStore,
coach-config, sport-registry, and MCP-tool seam contracts live alongside their
implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from wattwise_core.domain.candidate import FieldCandidate
from wattwise_core.domain.enums import Fidelity
from wattwise_core.entitlement import EntitlementResolver
from wattwise_core.ingestion.base import SourceAdapter
from wattwise_core.ingestion.dedup import resolve_field


class SourceAdapterContract(ABC):
    """Invariants every source adapter MUST satisfy (ADP-R*, MAP-R1/R2).

    Subclass and implement :meth:`adapter`. The cases assert the adapter exposes the
    required identity metadata and a pure ``map`` whose output is canonical-only.
    """

    @abstractmethod
    def adapter(self) -> SourceAdapter:
        """Return the adapter under test."""

    def test_declares_identity_metadata(self) -> None:
        """An adapter declares the metadata the registry + connection flow need (ADP-R*)."""
        a = self.adapter()
        assert isinstance(a.source_key, str) and a.source_key
        assert a.auth_archetype is not None
        assert a.kind is not None
        assert isinstance(a.adapter_version, str)
        assert isinstance(a.mapping_version, str)

    def test_satisfies_protocol(self) -> None:
        """The adapter is structurally a SourceAdapter (typed seam, QUAL-R9c)."""
        assert isinstance(self.adapter(), SourceAdapter)


class ResolverContract:
    """Invariants every conflict resolver MUST satisfy (CONF-R2/R4/R5, DEDUP-R1).

    The OSS default resolver and any commercial replacement (DEDUP-R8) MUST pass these.
    Subclasses override :meth:`resolve` to point at the resolver under test; the default
    points at the shipped :func:`resolve_field`.
    """

    def resolve(self, candidates: list[FieldCandidate]) -> object | None:
        out = resolve_field(candidates)
        return None if out is None else out.value

    def test_no_contributor_is_typed_gap_not_zero(self) -> None:
        """No contributor -> None (a typed gap), never a fabricated 0 (CONF-R5)."""
        assert self.resolve([]) is None

    def test_highest_fidelity_wins(self) -> None:
        """Trust tier is the primary key of resolution (CONF-R2 step 1)."""
        raw = FieldCandidate(1.0, Fidelity.RAW_STREAM, "b")
        summary = FieldCandidate(2.0, Fidelity.SUMMARY_ONLY, "a")
        assert self.resolve([summary, raw]) == 1.0

    def test_deterministic_regardless_of_order(self) -> None:
        """Same candidate set -> same winner regardless of order (CONF-R4)."""
        cs = [
            FieldCandidate(1.0, Fidelity.MODELED, "m"),
            FieldCandidate(2.0, Fidelity.RAW_STREAM, "r"),
        ]
        assert self.resolve(cs) == self.resolve(list(reversed(cs)))

    def test_stable_tiebreak_lowest_source_id(self) -> None:
        """All-equal candidates resolve by lowest source id (byte-reproducible, CONF-R2)."""
        a = FieldCandidate(5.0, Fidelity.SUMMARY_ONLY, "aaa")
        b = FieldCandidate(6.0, Fidelity.SUMMARY_ONLY, "bbb")
        assert self.resolve([b, a]) == 5.0


class EntitlementResolverContract(ABC):
    """Invariants every entitlement resolver MUST satisfy (ENT-R*, DELIV-R6).

    The OSS all-permissive default and any commercial metered resolver MUST pass these.
    """

    @abstractmethod
    def resolver(self) -> EntitlementResolver:
        """Return the entitlement resolver under test."""

    def test_resolves_for_owner(self) -> None:
        """Resolving for the owner yields an entitlements object (resolve->attach->check)."""
        ent = self.resolver().resolve("athlete-1")
        assert ent is not None

    def test_satisfies_protocol(self) -> None:
        """The resolver is structurally an EntitlementResolver (typed seam, QUAL-R9c)."""
        assert isinstance(self.resolver(), EntitlementResolver)


__all__ = [
    "EntitlementResolverContract",
    "ResolverContract",
    "SourceAdapterContract",
]
