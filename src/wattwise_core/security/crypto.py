"""Envelope encryption for source secrets at rest (SEC-R7, PRIV-R3).

A single service **root key** (the encryption master key, supplied ONLY via the
environment / a secret manager — BOOT-R4 / SEC-R12, never baked into source,
config, or images) wraps a fresh **per-record data key** for every encryption.
The plaintext is sealed with that per-record data key; the data key is then
sealed (wrapped) with the root key. Only the two ciphertexts and their nonces
are ever persisted — the plaintext and the unwrapped data key live in memory
only at the point of use (SEC-R7).

Cipher choice — **AES-256-GCM** (``cryptography`` AESGCM) rather than Fernet:

- AES-GCM is a true AEAD: it authenticates the ciphertext, so tampering is
  detected on decrypt and surfaces as an error (fail-closed), and it lets us
  bind an immutable, per-token *associated data* header (the wire-format
  version) so a token cannot be re-interpreted under a different layout.
- A misuse-resistant API: there is **no** code path that returns or stores
  plaintext alongside a token. :func:`EnvelopeCipher.encrypt` takes ``bytes``
  and returns an opaque :class:`EnvelopeToken` that carries ONLY ciphertext;
  :meth:`EnvelopeToken.__repr__`/``__str__`` never render the wrapped material.
  A fresh 96-bit nonce is drawn per encryption for both the payload seal and the
  data-key wrap (never reused), and a fresh 256-bit data key is drawn per record.
- Fail-closed construction: a missing root key, or a root key that is not
  exactly 32 bytes (256 bits) of base64-decoded material, raises at construction
  (SEC-R3 entropy floor analogue / RUN-R4.1) — the cipher refuses to exist in an
  insecure state rather than silently using a weak/zero key.

Fernet would also be authenticated, but it hardcodes AES-128-CBC+HMAC, hides the
key-wrap layer we need for true *envelope* encryption (per-record data keys), and
does not expose associated data for version binding; AES-256-GCM is the better
fit for the SEC-R7 envelope shape.
"""

from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Wire-format version, bound as AES-GCM associated data on BOTH the data-key wrap
# and the payload seal. Bumping this (a future algorithm/layout change) makes old
# tokens fail authentication rather than be silently misread.
_FORMAT_VERSION = b"wattwise-envelope-v1"

# AES-256 key length (the root key and every per-record data key) and the
# GCM nonce length, in bytes. 96-bit nonces are the GCM-recommended size.
_KEY_BYTES = 32
_NONCE_BYTES = 12


class CryptoError(Exception):
    """A fail-closed cryptographic error.

    Raised when the root key is missing/malformed (refuse to operate, SEC-R3 /
    RUN-R4.1) or when a token fails authentication on decrypt (tampered or wrong
    key, SEC-R7). It never carries key or plaintext material in its message.
    """


@dataclass(frozen=True, slots=True)
class EnvelopeToken:
    """An opaque, immutable envelope-encrypted token (SEC-R7).

    Carries ONLY ciphertext: the root-key-wrapped per-record data key
    (``wrapped_key`` + its ``key_nonce``) and the data-key-sealed payload
    (``ciphertext`` + its ``payload_nonce``). There is no field, property, or
    serialization path that exposes the plaintext or the unwrapped data key.
    ``__repr__``/``__str__`` deliberately redact the wrapped material so the
    token cannot leak via a log line or an f-string (PRIV-R5 / LOG-R5).
    """

    key_nonce: bytes
    wrapped_key: bytes
    payload_nonce: bytes
    ciphertext: bytes

    def __repr__(self) -> str:
        # Never render the wrapped material; only its presence and size.
        return f"EnvelopeToken(<sealed {len(self.ciphertext)} bytes>)"

    __str__ = __repr__

    def to_wire(self) -> str:
        """Serialize to a single URL-safe base64 string for at-rest storage.

        The encoded bytes are ONLY ciphertext + nonces — never plaintext — so
        persisting this string in the canonical/credential store leaks nothing
        usable without the root key (SEC-R7, PRIV-R3).
        """
        blob = b"".join(
            (
                len(self.key_nonce).to_bytes(1, "big"),
                self.key_nonce,
                len(self.wrapped_key).to_bytes(2, "big"),
                self.wrapped_key,
                len(self.payload_nonce).to_bytes(1, "big"),
                self.payload_nonce,
                self.ciphertext,
            )
        )
        return base64.urlsafe_b64encode(blob).decode("ascii")

    @classmethod
    def from_wire(cls, wire: str) -> EnvelopeToken:
        """Parse a :meth:`to_wire` string back into a token (no decryption).

        Raises :class:`CryptoError` (fail-closed) on any malformed/truncated
        input rather than returning a partially built token.
        """
        try:
            blob = base64.urlsafe_b64decode(wire.encode("ascii"))
        except (binascii.Error, ValueError) as exc:
            raise CryptoError("malformed envelope token (base64)") from exc
        try:
            pos = 0
            kn_len = blob[pos]
            pos += 1
            key_nonce = blob[pos : pos + kn_len]
            pos += kn_len
            wk_len = int.from_bytes(blob[pos : pos + 2], "big")
            pos += 2
            wrapped_key = blob[pos : pos + wk_len]
            pos += wk_len
            pn_len = blob[pos]
            pos += 1
            payload_nonce = blob[pos : pos + pn_len]
            pos += pn_len
            ciphertext = blob[pos:]
        except IndexError as exc:
            raise CryptoError("malformed envelope token (truncated)") from exc
        if (
            len(key_nonce) != kn_len
            or len(wrapped_key) != wk_len
            or len(payload_nonce) != pn_len
            or not ciphertext
        ):
            raise CryptoError("malformed envelope token (length mismatch)")
        return cls(
            key_nonce=key_nonce,
            wrapped_key=wrapped_key,
            payload_nonce=payload_nonce,
            ciphertext=ciphertext,
        )


def _decode_root_key(root_key_b64: str | None) -> bytes:
    """Decode + validate the base64 root key, fail-closed (SEC-R3 / RUN-R4.1)."""
    if not root_key_b64:
        # No in-code fallback default (SEC-R12): refuse to operate without a key.
        raise CryptoError(
            "encryption root key is missing; it MUST be supplied via the "
            "environment / a secret manager (BOOT-R4)"
        )
    try:
        raw = base64.b64decode(root_key_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CryptoError("encryption root key is not valid base64") from exc
    if len(raw) != _KEY_BYTES:
        # Short/over-long key = insufficient entropy for AES-256 → refuse.
        raise CryptoError(
            f"encryption root key must decode to exactly {_KEY_BYTES} bytes "
            f"(256 bits); got {len(raw)} bytes"
        )
    return raw


class EnvelopeCipher:
    """Envelope-encrypt/decrypt arbitrary byte payloads under a root key (SEC-R7).

    Construct ONCE from the base64 root key (from config / secret manager) and
    reuse: every :meth:`encrypt` draws its own per-record data key + nonces, so
    the same instance safely serves many records.
    """

    __slots__ = ("_root",)

    def __init__(self, root_key_b64: str | None) -> None:
        # Validate-and-hold the AESGCM wrapper for the root key. The raw key
        # bytes are not retained as an attribute beyond AESGCM's internal use.
        self._root = AESGCM(_decode_root_key(root_key_b64))

    @classmethod
    def generate_root_key(cls) -> str:
        """Return a fresh, correctly sized base64 root key (for tests/bootstrap).

        This is a key *generator* for an operator to seed their secret manager;
        it does NOT read or persist anything and is never a fallback default.
        """
        return base64.b64encode(AESGCM.generate_key(bit_length=256)).decode("ascii")

    def encrypt(self, plaintext: bytes) -> EnvelopeToken:
        """Envelope-encrypt ``plaintext`` (SEC-R7).

        Draws a fresh 256-bit data key and two fresh 96-bit nonces, seals the
        plaintext under the data key, then wraps the data key under the root key.
        Returns an opaque :class:`EnvelopeToken` (ciphertext only).
        """
        if not isinstance(plaintext, (bytes, bytearray)):  # defensive, typed API
            raise TypeError("plaintext must be bytes")
        data_key = AESGCM.generate_key(bit_length=256)
        payload_nonce = _random_nonce()
        ciphertext = AESGCM(data_key).encrypt(payload_nonce, bytes(plaintext), _FORMAT_VERSION)
        key_nonce = _random_nonce()
        wrapped_key = self._root.encrypt(key_nonce, data_key, _FORMAT_VERSION)
        return EnvelopeToken(
            key_nonce=key_nonce,
            wrapped_key=wrapped_key,
            payload_nonce=payload_nonce,
            ciphertext=ciphertext,
        )

    def decrypt(self, token: EnvelopeToken) -> bytes:
        """Authenticate + decrypt a token, returning the plaintext bytes (SEC-R7).

        Unwraps the data key with the root key, then opens the payload. Any
        tampering with either ciphertext, nonce, or the bound version header
        fails authentication and raises :class:`CryptoError` (fail-closed) — the
        method NEVER returns unauthenticated bytes.
        """
        try:
            data_key = self._root.decrypt(token.key_nonce, token.wrapped_key, _FORMAT_VERSION)
        except InvalidTag as exc:
            raise CryptoError("envelope key unwrap failed (tampered or wrong root key)") from exc
        try:
            return AESGCM(data_key).decrypt(token.payload_nonce, token.ciphertext, _FORMAT_VERSION)
        except InvalidTag as exc:
            raise CryptoError("envelope payload decryption failed (tampered ciphertext)") from exc


def _random_nonce() -> bytes:
    """Return a fresh, never-reused 96-bit GCM nonce from the OS CSPRNG."""
    return os.urandom(_NONCE_BYTES)
