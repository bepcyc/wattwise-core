"""Opaque, signed cursor pagination for unbounded collections (PAGE-R1..R8).

Unbounded collections page by an OPAQUE, HMAC-SIGNED, server-issued cursor — never
an offset and never a client-constructible token (PAGE-R5). The cursor carries the
keyset position ``(start_time, id)`` AND a fingerprint of the originating
filter/sort/order params, signed with the engine ``token_signing_key``; a tampered or
re-keyed cursor fails signature verification (``invalid-cursor``), and a cursor
replayed against changed filters/sort is rejected ``cursor-parameter-mismatch``
(PAGE-R6). ``limit`` is clamped to ``[1, 200]`` (PAGE-R3); there is no offset paging.

Requirement IDs: PAGE-R1 (cursor paging), PAGE-R2 (typed sort/order), PAGE-R3 (limit
clamp), PAGE-R5 (opaque signed cursor; tamper -> ``invalid-cursor``), PAGE-R6 (cursor
bound to filter/sort -> ``cursor-parameter-mismatch``), PAGE-R7 (tie-broken keyset).
"""

from __future__ import annotations

import base64
import binascii
import datetime as _dt
import hashlib
import hmac
import json
from typing import Final

from wattwise_core.api.errors import ProblemError

#: The HMAC digest the cursor signature uses (truncated to keep the token compact).
_DIGEST: Final = hashlib.sha256

#: Bytes of the HMAC tag carried in the cursor (128-bit tag; ample for tamper-proofing).
_TAG_BYTES: Final = 16

#: PAGE-R3: the hard clamp on any page ``limit`` (never unbounded; never an offset).
MAX_PAGE_LIMIT: Final = 200
DEFAULT_PAGE_LIMIT: Final = 50


def clamp_limit(limit: int, *, default: int = DEFAULT_PAGE_LIMIT) -> int:
    """Clamp a requested page size into ``[1, MAX_PAGE_LIMIT]`` (PAGE-R3)."""
    if limit <= 0:
        return default
    return min(int(limit), MAX_PAGE_LIMIT)


def _fingerprint(params: dict[str, str]) -> str:
    """A stable, order-independent fingerprint of the originating filter/sort params."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _sign(payload: bytes, key: str) -> bytes:
    """The truncated HMAC tag binding ``payload`` to the server signing key (PAGE-R5)."""
    return hmac.new(key.encode(), payload, _DIGEST).digest()[:_TAG_BYTES]


def encode_cursor(
    start_time: _dt.datetime, item_id: str, *, params: dict[str, str], key: str
) -> str:
    """Encode a signed, filter-bound keyset cursor (PAGE-R5/R6).

    The payload carries the keyset position, the active filter/sort fingerprint, and an
    HMAC tag over both; it is base64url-encoded into one opaque token a client stores
    and returns verbatim but can neither read nor forge.
    """
    body = {"t": start_time.isoformat(), "id": item_id, "f": _fingerprint(params)}
    raw = json.dumps(body, separators=(",", ":")).encode()
    tag = _sign(raw, key)
    token = base64.urlsafe_b64encode(raw).decode() + "." + base64.urlsafe_b64encode(tag).decode()
    return token


def decode_cursor(cursor: str, *, params: dict[str, str], key: str) -> tuple[_dt.datetime, str]:
    """Verify + decode a signed cursor to its keyset position (PAGE-R5/R6).

    A malformed token or a bad/forged signature -> ``invalid-cursor`` (400); a valid
    cursor whose embedded filter/sort fingerprint disagrees with the current request ->
    ``cursor-parameter-mismatch`` (400). The keyset ``(start_time, id)`` is returned
    only after both checks pass.
    """
    try:
        raw_part, tag_part = cursor.split(".", 1)
        raw = base64.urlsafe_b64decode(raw_part.encode())
        tag = base64.urlsafe_b64decode(tag_part.encode())
    except (ValueError, binascii.Error) as exc:
        raise ProblemError("invalid-cursor") from exc
    if not hmac.compare_digest(tag, _sign(raw, key)):
        raise ProblemError("invalid-cursor")
    try:
        data = json.loads(raw)
        start_time = _dt.datetime.fromisoformat(data["t"])
        item_id = str(data["id"])
        fingerprint = str(data["f"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ProblemError("invalid-cursor") from exc
    if fingerprint != _fingerprint(params):
        raise ProblemError("cursor-parameter-mismatch")
    return start_time, item_id


__all__ = [
    "DEFAULT_PAGE_LIMIT",
    "MAX_PAGE_LIMIT",
    "clamp_limit",
    "decode_cursor",
    "encode_cursor",
]
