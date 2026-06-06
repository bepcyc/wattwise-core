"""Unit tests for the credential store (SEC-R7, AUT-R2, AUTH-R16).

Proves: ``store`` persists ONLY ciphertext and returns an opaque, unguessable
``credential_ref``; ``resolve`` round-trips the secret as a SecretStr without ever
revealing plaintext through repr/serialize; an unknown ref fails closed; and
``delete`` (PRIV-R8 erasure) removes the ciphertext.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from wattwise_core.security.credentials import (
    CredentialNotFoundError,
    CredentialStore,
    InMemoryCredentialStore,
    StoredCredential,
)
from wattwise_core.security.crypto import EnvelopeCipher, EnvelopeToken


@pytest.fixture
def store() -> InMemoryCredentialStore:
    return InMemoryCredentialStore(EnvelopeCipher(EnvelopeCipher.generate_root_key()))


def test_in_memory_store_satisfies_protocol(store: InMemoryCredentialStore) -> None:
    assert isinstance(store, CredentialStore)


def test_store_returns_opaque_ref_and_resolve_round_trips(
    store: InMemoryCredentialStore,
) -> None:
    raw = "garmin-session-token-xyz"
    ref = store.store(raw)
    # Opaque, prefixed, not the secret itself.
    assert ref.startswith("cred_")
    assert raw not in ref
    resolved = store.resolve(ref)
    assert isinstance(resolved, SecretStr)
    assert resolved.get_secret_value() == raw


def test_store_accepts_bytes(store: InMemoryCredentialStore) -> None:
    # Source secrets are text (OAuth/API/session tokens); bytes input is accepted
    # and round-trips as the equivalent UTF-8 string via SecretStr.
    ref = store.store(b"utf8-bytes-token")
    assert store.resolve(ref).get_secret_value() == "utf8-bytes-token"
    ref2 = store.store("simple")
    assert store.resolve(ref2).get_secret_value() == "simple"


def test_refs_are_unique_and_unguessable(store: InMemoryCredentialStore) -> None:
    refs = {store.store(f"secret-{i}") for i in range(50)}
    assert len(refs) == 50  # all distinct
    # Not sequential / enumerable.
    assert all(not r.removeprefix("cred_").isdigit() for r in refs)


def test_credential_ref_never_reveals_plaintext_via_store_repr(
    store: InMemoryCredentialStore,
) -> None:
    secret = "TOP-SECRET-REFRESH-TOKEN"
    ref = store.store(secret)
    # Neither the store repr nor the ref expose the plaintext.
    assert secret not in repr(store)
    assert secret not in ref
    # The persisted record holds ciphertext only — plaintext is absent.
    sealed: EnvelopeToken = store._records[ref]  # white-box: ciphertext-only check
    assert secret.encode() not in sealed.ciphertext
    assert secret.encode() not in sealed.wrapped_key


def test_secretstr_does_not_leak_in_repr(store: InMemoryCredentialStore) -> None:
    ref = store.store("hidden-value-123")
    resolved = store.resolve(ref)
    assert "hidden-value-123" not in repr(resolved)
    assert "hidden-value-123" not in str(resolved)


def test_unknown_ref_fails_closed(store: InMemoryCredentialStore) -> None:
    with pytest.raises(CredentialNotFoundError):
        store.resolve("cred_does-not-exist")


def test_delete_erases_ciphertext(store: InMemoryCredentialStore) -> None:
    ref = store.store("erase-me")
    store.delete(ref)
    with pytest.raises(CredentialNotFoundError):
        store.resolve(ref)
    # Idempotent: deleting again is a no-op.
    store.delete(ref)


def test_stored_credential_model_repr_hides_ciphertext() -> None:
    row = StoredCredential(credential_ref="cred_abc", wrapped_token="c2VhbGVkLXdpcmU=")
    rendered = repr(row)
    assert "cred_abc" in rendered
    assert "c2VhbGVkLXdpcmU=" not in rendered  # wrapped ciphertext never rendered
    assert row.__tablename__ == "source_credential"
