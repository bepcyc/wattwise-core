"""OSS default implementations pass their seam contracts (GOLD-R5 §6.3a, COMM-R16).

The shipped seam-contract base-classes are real and green against the bare OSS product:
the default conservative resolver (DEDUP-R7) passes the resolver + single-count
contracts, the config-loaded all-permissive entitlement resolver (DELIV-R6) passes the
entitlement finite-ceiling contract, the shipped deterministic grounding gate passes the
fail-closed grounding contract, the shipped bundle loader passes the coach-config
load-validation contract, and the in-memory saver over the production graph passes the
HITL resume contract.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from wattwise_core.config import load_settings
from wattwise_core.entitlement import EntitlementResolver, OssEntitlementResolver
from wattwise_core.testing.seam_contracts import (
    CoachConfigLoadValidationContract,
    EntitlementResolverContract,
    FailClosedGroundingContract,
    HitlResumeContract,
    ResolverContract,
    SingleCountContract,
)


class TestDefaultResolverConformance(ResolverContract):
    """The OSS default resolver (resolve_field) satisfies the resolver seam contract."""

    # Inherits the contract cases; the base `resolve` already targets resolve_field.


class TestDefaultSingleCountConformance(SingleCountContract):
    """The OSS conservative resolver (DEDUP-R7) satisfies the single-count contract."""

    # Inherits the contract cases; the base methods already target the shipped resolver.


class TestOssEntitlementConformance(EntitlementResolverContract):
    """The OSS config-loaded entitlement resolver satisfies the entitlement contract."""

    def resolver(self) -> EntitlementResolver:
        """The OSS default resolver with the config-loaded plan (CFG-R1a, finite bounds)."""
        settings = load_settings(
            app__environment="development",
            database_dsn="sqlite+aiosqlite:///:memory:",
            token_signing_key="unit-test-signing-key-not-a-real-secret",
        )
        return OssEntitlementResolver.from_settings(settings)


class TestShippedGroundingFailClosedConformance(FailClosedGroundingContract):
    """The shipped deterministic grounding gate satisfies the fail-closed contract."""

    # Inherits the contract cases; the base `ground_draft` already targets ground().


class TestShippedCoachConfigLoadValidation(CoachConfigLoadValidationContract):
    """The shipped bundle loader (load_manifest) satisfies COACH-CFG-R4 load validation."""

    # Inherits the contract cases; the base `load_bundle` already targets load_manifest.


class TestHitlResumeConformance(HitlResumeContract):
    """The production graph over a langgraph saver satisfies the HITL resume contract."""

    def checkpointer(self) -> InMemorySaver:
        """An in-memory saver: the durable sqlite/postgres savers are gated elsewhere."""
        return InMemorySaver()
