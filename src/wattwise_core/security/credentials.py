"""Credential store: opaque ``credential_ref`` over envelope-encrypted secrets.

Lifecycle (IDS / AUT-R2 / AUT-R7 / AUTH-R16, SEC-R7): an adapter exchanges the
athlete's raw credential server-side for a refreshable session/token and
IMMEDIATELY discards the raw value. The resulting token is handed here; this
store envelope-encrypts it (:mod:`.crypto`) and persists ONLY the ciphertext,
returning an **opaque** ``credential_ref``. The canonical data model
(``connection.credential_ref``) holds only that reference, never the secret. To
use the secret, an adapter calls :meth:`CredentialStore.resolve`, which decrypts
in-memory at the point of use and returns a :class:`pydantic.SecretStr` so the
plaintext is never accidentally rendered into a log line, repr, or trace
(PRIV-R5 / LOG-R5). No ``list``/serialize/``__repr__`` path on this store or its
records ever exposes the plaintext OR the wrapped ciphertext material.

Persistence seam: the OSS default :class:`InMemoryCredentialStore` keeps records
in-process (suitable for tests and a pluggable seam until the persistence package
is wired). A canonical SQLAlchemy table model — :class:`StoredCredential` — is
defined HERE (under ``security/``) so a future migration can pick it up WITHOUT
this module importing or editing ``wattwise_core.persistence``.

NOTE FOR THE PERSISTENCE OWNER: ``StoredCredential`` (table ``source_credential``)
is the at-rest home for the envelope-encrypted token; wire it into the canonical
metadata/migrations and add a DB-backed ``CredentialStore`` implementation that
reuses :class:`EnvelopeCipher` exactly as ``InMemoryCredentialStore`` does. The
``credential_ref`` column is the opaque pointer ``connection.credential_ref``
(doc 20) resolves to; ``wrapped_token`` stores ``EnvelopeToken.to_wire()`` only.
"""

from __future__ import annotations

import secrets
import uuid
from typing import Protocol, runtime_checkable

from pydantic import SecretStr
from sqlalchemy import LargeBinary, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from wattwise_core.security.crypto import EnvelopeCipher, EnvelopeToken


class CredentialNotFoundError(LookupError):
    """Raised when a ``credential_ref`` resolves to no stored credential.

    Carries only the opaque ref (never any secret material) so an unknown ref is
    distinguishable from a present one without leaking plaintext (SEC-R7).
    """


@runtime_checkable
class CredentialStore(Protocol):
    """Protocol for persisting/resolving envelope-encrypted source secrets.

    The OSS default and any commercial/DB-backed implementation share this exact
    shape so the resolving call-site (adapters' ``ensure_authorized``) never
    branches on the backend (SEC-R7 extension seam).
    """

    def store(self, raw_secret: bytes | str) -> str:
        """Envelope-encrypt ``raw_secret`` and persist ONLY the ciphertext.

        Returns an opaque, unguessable ``credential_ref``. The raw secret is not
        retained after this call returns and is never written in plaintext.
        """
        ...

    def resolve(self, credential_ref: str) -> SecretStr:
        """Resolve a ``credential_ref`` to its plaintext as a :class:`SecretStr`.

        Decrypts in-memory at the point of use (SEC-R7). Raises
        :class:`CredentialNotFoundError` for an unknown ref (fail-closed).
        """
        ...

    def delete(self, credential_ref: str) -> None:
        """Erase the stored ciphertext for ``credential_ref`` (PRIV-R8 erasure).

        Idempotent: deleting an unknown ref is a no-op (the desired end state —
        the secret is absent — already holds).
        """
        ...


def _new_credential_ref() -> str:
    """Return an opaque, unguessable, non-enumerable credential reference.

    A random URL-safe token (not a sequential id) so a ref cannot be guessed or
    enumerated (cf. PRIV-R11.1 non-enumerability) and reveals nothing about the
    secret it points to.
    """
    return f"cred_{secrets.token_urlsafe(24)}"


class InMemoryCredentialStore:
    """OSS-default, pluggable in-process credential store (SEC-R7).

    Holds ONLY :class:`EnvelopeToken` ciphertext per ``credential_ref`` — never
    the plaintext. Implements :class:`CredentialStore`. Suitable for tests and as
    the seam until a DB-backed store (using :class:`StoredCredential`) is wired.
    """

    __slots__ = ("_cipher", "_records")

    def __init__(self, cipher: EnvelopeCipher) -> None:
        self._cipher = cipher
        # ref -> sealed token (ciphertext only; no plaintext ever held here).
        self._records: dict[str, EnvelopeToken] = {}

    def store(self, raw_secret: bytes | str) -> str:
        # Source secrets are text (OAuth/API/session tokens). Accept bytes for
        # caller convenience but require valid UTF-8 so resolve() round-trips
        # losslessly into a SecretStr — fail-closed on a non-text secret rather
        # than persisting something that cannot be resolved later.
        if isinstance(raw_secret, str):
            plaintext = raw_secret.encode("utf-8")
        else:
            try:
                plaintext = bytes(raw_secret).decode("utf-8").encode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("raw_secret bytes must be valid UTF-8 text") from exc
        token = self._cipher.encrypt(plaintext)
        # Best-effort scrub of our local copy; the caller is responsible for not
        # retaining the raw credential beyond the exchange (AUTH-R16).
        del plaintext
        ref = _new_credential_ref()
        # Astronomically unlikely collision; loop keeps refs unique regardless.
        while ref in self._records:
            ref = _new_credential_ref()
        self._records[ref] = token
        return ref

    def resolve(self, credential_ref: str) -> SecretStr:
        token = self._records.get(credential_ref)
        if token is None:
            raise CredentialNotFoundError(credential_ref)
        plaintext = self._cipher.decrypt(token)
        # SecretStr keeps the value out of reprs/logs/tracebacks by construction.
        return SecretStr(plaintext.decode("utf-8"))

    def delete(self, credential_ref: str) -> None:
        self._records.pop(credential_ref, None)

    def __repr__(self) -> str:
        # Count only; never the refs' contents or any token material.
        return f"InMemoryCredentialStore(<{len(self._records)} sealed records>)"


class _CredentialBase(DeclarativeBase):
    """Local declarative base for the security-owned credential table.

    Kept separate from the persistence package's metadata on purpose: this module
    MUST NOT import ``wattwise_core.persistence`` (owned by another agent). The
    persistence owner may re-home :class:`StoredCredential` onto the canonical
    metadata when wiring migrations (see the NOTE FOR THE PERSISTENCE OWNER in the
    module docstring).
    """


class StoredCredential(_CredentialBase):
    """At-rest row for one envelope-encrypted source secret (SEC-R7, AUT-R2).

    Stores ONLY the opaque ``credential_ref`` and the wrapped ciphertext
    (``EnvelopeToken.to_wire()``) — never the plaintext, never the unwrapped data
    key. ``connection.credential_ref`` (doc 20) points at ``credential_ref``.
    """

    __tablename__ = "source_credential"

    # Opaque, unguessable pointer the canonical store references; the PK.
    credential_ref: Mapped[str] = mapped_column(String(64), primary_key=True)
    # The envelope-encrypted token, base64 wire form (ciphertext + nonces only).
    wrapped_token: Mapped[str] = mapped_column(String, nullable=False)
    # Optional opaque athlete/connection scoping handles (no PII); a DB-backed
    # store sets these for per-(athlete, source) scoping (AUT-R2) and erasure
    # (PRIV-R8). Left nullable so the schema is additive for the persistence owner.
    athlete_ref: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    connection_ref: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    def __repr__(self) -> str:
        # Never render wrapped_token (ciphertext) — only the opaque ref.
        return f"StoredCredential(credential_ref={self.credential_ref!r}, <sealed>)"


def new_uuid_ref() -> str:
    """Return a UUID4 string (helper for DB-backed stores that prefer UUID refs)."""
    return uuid.uuid4().hex
