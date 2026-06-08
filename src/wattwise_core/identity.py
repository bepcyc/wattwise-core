"""The OSS single-owner identity anchor (GBO-R13 — exactly one athlete).

OSS is single-tenant: there is exactly ONE athlete (GBO-R13), provisioned with this
fixed, deterministic id (seeded by the initial migration) and used as the server-derived
subject of every first-party access token (AUTH-R3/R18). A STABLE id — not a random one
minted per boot — is load-bearing: the token ``sub`` claim, the seeded ``athlete`` row,
and every canonical foreign key must agree across restarts and across a re-deploy without
any out-of-band provisioning step. A random per-boot id would orphan all previously
ingested canonical data the moment the process restarted.

The id is derived deterministically (a UUIDv5 over a fixed name) so it is reproducible
from source with no stored state, and is the SAME value the migration seeds and the auth
layer signs. Tenancy is NOT expressed by this id (AUTH-R18): it is a referential anchor,
never an isolation key — the commercial layer that adds real tenants supplies per-tenant
ids through the same seam without changing this OSS default.
"""

from __future__ import annotations

import uuid
from typing import Final

#: The fixed canonical id of the single OSS owner athlete (GBO-R13). Deterministic so the
#: seeded ``athlete`` row, the token subject, and every FK agree across boots (AUTH-R18).
OWNER_ATHLETE_ID: Final = uuid.uuid5(uuid.NAMESPACE_DNS, "owner.oss.wattwise.invalid")

#: The owner id as the string ``sub`` claim carried by every first-party token (AUTH-R3).
OWNER_SUBJECT: Final = str(OWNER_ATHLETE_ID)


__all__ = ["OWNER_ATHLETE_ID", "OWNER_SUBJECT"]
