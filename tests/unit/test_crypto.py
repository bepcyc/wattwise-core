"""Unit tests for envelope encryption (SEC-R7, PRIV-R3).

Proves: encrypt -> decrypt round-trips; ciphertext != plaintext; tampering with
any ciphertext/nonce fails authentication (fail-closed); fresh nonces + data keys
per encryption; fail-closed construction on a missing/short root key; and that the
token never renders its sealed material in repr/str (PRIV-R5 / LOG-R5).
"""

from __future__ import annotations

import base64

import pytest

from wattwise_core.security.crypto import (
    CryptoError,
    EnvelopeCipher,
    EnvelopeToken,
)


@pytest.fixture
def cipher() -> EnvelopeCipher:
    return EnvelopeCipher(EnvelopeCipher.generate_root_key())


def test_encrypt_decrypt_round_trip(cipher: EnvelopeCipher) -> None:
    plaintext = b"oauth-refresh-token-abc123"
    token = cipher.encrypt(plaintext)
    assert cipher.decrypt(token) == plaintext


def test_ciphertext_differs_from_plaintext(cipher: EnvelopeCipher) -> None:
    plaintext = b"super-secret-session-token"
    token = cipher.encrypt(plaintext)
    # The sealed payload must not contain the plaintext anywhere.
    assert plaintext not in token.ciphertext
    assert plaintext not in token.wrapped_key
    assert token.ciphertext != plaintext


def test_round_trip_empty_and_unicode(cipher: EnvelopeCipher) -> None:
    for plaintext in (b"", "пароль-🔑".encode(), bytes(range(256))):
        assert cipher.decrypt(cipher.encrypt(plaintext)) == plaintext


def test_nonces_and_data_keys_are_fresh_per_encryption(cipher: EnvelopeCipher) -> None:
    a = cipher.encrypt(b"same plaintext")
    b = cipher.encrypt(b"same plaintext")
    # Same plaintext, different ciphertext (fresh nonce + fresh data key each time).
    assert a.payload_nonce != b.payload_nonce
    assert a.key_nonce != b.key_nonce
    assert a.ciphertext != b.ciphertext
    assert a.wrapped_key != b.wrapped_key


def test_tampered_ciphertext_fails_closed(cipher: EnvelopeCipher) -> None:
    token = cipher.encrypt(b"do not tamper")
    flipped = bytearray(token.ciphertext)
    flipped[0] ^= 0x01
    tampered = EnvelopeToken(
        key_nonce=token.key_nonce,
        wrapped_key=token.wrapped_key,
        payload_nonce=token.payload_nonce,
        ciphertext=bytes(flipped),
    )
    with pytest.raises(CryptoError):
        cipher.decrypt(tampered)


def test_tampered_wrapped_key_fails_closed(cipher: EnvelopeCipher) -> None:
    token = cipher.encrypt(b"do not tamper key")
    flipped = bytearray(token.wrapped_key)
    flipped[-1] ^= 0x80
    tampered = EnvelopeToken(
        key_nonce=token.key_nonce,
        wrapped_key=bytes(flipped),
        payload_nonce=token.payload_nonce,
        ciphertext=token.ciphertext,
    )
    with pytest.raises(CryptoError):
        cipher.decrypt(tampered)


def test_wrong_root_key_cannot_decrypt(cipher: EnvelopeCipher) -> None:
    token = cipher.encrypt(b"sealed under key A")
    other = EnvelopeCipher(EnvelopeCipher.generate_root_key())
    with pytest.raises(CryptoError):
        other.decrypt(token)


def test_missing_root_key_fails_closed() -> None:
    with pytest.raises(CryptoError):
        EnvelopeCipher(None)
    with pytest.raises(CryptoError):
        EnvelopeCipher("")


def test_short_root_key_fails_closed() -> None:
    short = base64.b64encode(b"too-short-key").decode("ascii")  # 13 bytes, not 32
    with pytest.raises(CryptoError):
        EnvelopeCipher(short)


def test_non_base64_root_key_fails_closed() -> None:
    with pytest.raises(CryptoError):
        EnvelopeCipher("not valid base64 !!!")


def test_wire_round_trip(cipher: EnvelopeCipher) -> None:
    token = cipher.encrypt(b"persist me at rest")
    wire = token.to_wire()
    # Wire form leaks no plaintext.
    assert b"persist me at rest" not in base64.urlsafe_b64decode(wire)
    restored = EnvelopeToken.from_wire(wire)
    assert cipher.decrypt(restored) == b"persist me at rest"


def test_malformed_wire_fails_closed() -> None:
    with pytest.raises(CryptoError):
        EnvelopeToken.from_wire("!!!not-base64!!!")
    with pytest.raises(CryptoError):
        EnvelopeToken.from_wire(base64.urlsafe_b64encode(b"\x05ab").decode("ascii"))


def test_token_repr_does_not_leak_sealed_material(cipher: EnvelopeCipher) -> None:
    token = cipher.encrypt(b"secret-bytes")
    text = repr(token) + str(token)
    assert "secret-bytes" not in text
    # The raw ciphertext bytes must not appear in the human rendering.
    assert token.ciphertext.hex() not in text
    assert "REDACTED" not in text  # it simply omits, by size summary
    assert "sealed" in text
