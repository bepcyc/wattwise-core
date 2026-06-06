"""OSS default implementations pass their seam contracts (GOLD-R5 §6.3a).

The shipped seam-contract base-classes are real and green against the bare OSS
product: the default conservative resolver (DEDUP-R7) passes the resolver contract,
and the all-permissive default entitlement resolver (DELIV-R6) passes the entitlement
contract.
"""

from __future__ import annotations

from wattwise_core.entitlement import EntitlementResolver, OssEntitlementResolver
from wattwise_core.testing.seam_contracts import (
    EntitlementResolverContract,
    ResolverContract,
)


class TestDefaultResolverConformance(ResolverContract):
    """The OSS default resolver (resolve_field) satisfies the resolver seam contract."""

    # Inherits the contract cases; the base `resolve` already targets resolve_field.


class TestOssEntitlementConformance(EntitlementResolverContract):
    """The OSS all-permissive entitlement resolver satisfies the entitlement contract."""

    def resolver(self) -> EntitlementResolver:
        """The bare OSS default entitlement resolver."""
        return OssEntitlementResolver()
