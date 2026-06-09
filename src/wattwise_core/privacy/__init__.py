"""Data-subject-rights executors (PRIV-1, PRIV-R8/-R11, MEM-R3).

The whole-athlete erasure EXECUTOR (the "right to be forgotten" fulfilment path) lives
here. The API layer (``DELETE /v1/users/me``) records an asynchronous deletion request;
the background erasure path drives :func:`erase_athlete`, which deletes every
athlete-scoped record across BOTH the canonical GBO store and the dedicated agent-state
store in one pass and returns an auditable :class:`ErasureReceipt` (PRIV-R8: "produce an
auditable record that erasure completed"). This package owns no HTTP/auth concerns and is
wired by the API composition root (ARCH-R22) — it is a clean, identity-scoped callable.
"""

from __future__ import annotations

from wattwise_core.privacy.erasure import (
    ErasureReceipt,
    StoreErasureReport,
    erase_athlete,
)

__all__ = ["ErasureReceipt", "StoreErasureReport", "erase_athlete"]
