"""Fuzz tests for the file-upload decoders (TIER-R5 T-FUZZ, CLI-R2, FIL fuzz).

Covers the FIT/GPX/TCX decoders.

The contract: arbitrary / corrupt / truncated bytes -> a TYPED
:class:`FileDecodeError`, NEVER an uncaught exception escape and NEVER a
wrong-but-plausible canonical record. This is the fail-closed boundary (ING-R3):
when a file cannot be parsed the adapter raises a typed error rather than coercing
garbage into an activity.

Three fuzz surfaces:

* arbitrary bytes through the format-dispatching :func:`decode`;
* each per-format decoder fed arbitrary bytes directly;
* byte-level mutations of a VALID FIT fixture (flipped/dropped bytes) — every
  outcome must be either a clean :class:`ActivityAsbo` or a typed
  :class:`FileDecodeError`, never anything else.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from wattwise_core.ingestion.adapters._asbo import ActivityAsbo, FileDecodeError
from wattwise_core.ingestion.adapters._decode_fit import decode_fit
from wattwise_core.ingestion.adapters._decode_gpx import decode_gpx
from wattwise_core.ingestion.adapters._decode_tcx import decode_tcx
from wattwise_core.ingestion.adapters.file_upload import decode

pytestmark = pytest.mark.fuzz  # exactly ONE tier marker (TIER-R3)

_FIXTURES = Path(__file__).resolve().parents[1] / "contract" / "fixtures" / "file_upload"
_VALID_FIT = (_FIXTURES / "ride.fit").read_bytes()

# A FIT-looking header so the bytes route to the FIT decoder (header size 12,
# ".FIT" signature at offset 8) — exercises the binary parser, not just rejection.
_FIT_HEADER = bytes([12, 0x10, 0x00, 0x00]) + b"\x00\x00\x00\x00" + b".FIT"


def _assert_typed_or_asbo(result: object) -> None:
    assert isinstance(result, ActivityAsbo)


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.binary(min_size=0, max_size=4096))
def test_decode_arbitrary_bytes_never_escapes(data: bytes) -> None:
    """Arbitrary bytes -> typed error OR a valid ASBO, never an uncaught exception."""
    try:
        result = decode(data, filename=None)
    except FileDecodeError:
        return
    _assert_typed_or_asbo(result)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.binary(min_size=0, max_size=2048))
def test_decode_fit_arbitrary_bytes_never_escapes(data: bytes) -> None:
    try:
        result = decode_fit(_FIT_HEADER + data)
    except FileDecodeError:
        return
    _assert_typed_or_asbo(result)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.binary(min_size=0, max_size=2048))
def test_decode_gpx_arbitrary_bytes_never_escapes(data: bytes) -> None:
    try:
        result = decode_gpx(data)
    except FileDecodeError:
        return
    _assert_typed_or_asbo(result)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.binary(min_size=0, max_size=2048))
def test_decode_tcx_arbitrary_bytes_never_escapes(data: bytes) -> None:
    try:
        result = decode_tcx(data)
    except FileDecodeError:
        return
    _assert_typed_or_asbo(result)


@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
@given(
    index=st.integers(min_value=0, max_value=len(_VALID_FIT) - 1),
    new_byte=st.integers(min_value=0, max_value=255),
)
def test_single_byte_mutation_of_valid_fit_never_escapes(index: int, new_byte: int) -> None:
    """A one-byte mutation of a valid FIT -> typed error OR a valid ASBO, never a crash."""
    mutated = bytearray(_VALID_FIT)
    mutated[index] = new_byte
    try:
        result = decode_fit(bytes(mutated))
    except FileDecodeError:
        return
    _assert_typed_or_asbo(result)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(cut=st.integers(min_value=0, max_value=len(_VALID_FIT)))
def test_truncation_of_valid_fit_never_escapes(cut: int) -> None:
    """Truncating a valid FIT at any offset -> typed error OR recovered ASBO."""
    try:
        result = decode_fit(_VALID_FIT[:cut])
    except FileDecodeError:
        return
    _assert_typed_or_asbo(result)


def test_empty_bytes_is_typed_error() -> None:
    for decoder in (decode_fit, decode_gpx, decode_tcx):
        with pytest.raises(FileDecodeError):
            decoder(b"")


def test_xml_xxe_payload_does_not_trigger_external_fetch() -> None:
    """An XXE-style entity in a TCX must not fetch a URL; it fails closed (no I/O)."""
    xxe = (
        b'<?xml version="1.0"?>'
        b'<!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]>'
        b"<TrainingCenterDatabase><Activities><Activity><Id>&e;</Id>"
        b"</Activity></Activities></TrainingCenterDatabase>"
    )
    # Either a typed decode error or an ASBO whose content never expanded the entity
    # (resolve_entities=False) — never an external read.
    try:
        result = decode_tcx(xxe)
    except FileDecodeError:
        return
    assert isinstance(result, ActivityAsbo)
    assert "root:" not in str(result.session)
