"""Secret/crypto + credential-storage layer for wattwise-core.

This package owns the at-rest protection of source secrets (SEC-R7): envelope
encryption (:mod:`.crypto`) and the credential store that persists ONLY the
opaque, envelope-encrypted ciphertext and resolves an opaque ``credential_ref``
to a held-in-memory :class:`pydantic.SecretStr` (:mod:`.credentials`). The raw
credential is exchanged-then-discarded by the adapter and is never persisted,
logged, or returned in plaintext through any list/serialize path
(IDS / AUT-R2 / AUTH-R16).
"""

from __future__ import annotations

from wattwise_core.security.credentials import (
    CredentialNotFoundError,
    CredentialStore,
    InMemoryCredentialStore,
)
from wattwise_core.security.crypto import (
    CryptoError,
    EnvelopeCipher,
    EnvelopeToken,
)

__all__ = [
    "CredentialNotFoundError",
    "CredentialStore",
    "CryptoError",
    "EnvelopeCipher",
    "EnvelopeToken",
    "InMemoryCredentialStore",
]
