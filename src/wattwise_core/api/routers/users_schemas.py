"""Wire shapes for the ``/v1/users/me`` account surface (doc 60 §8 / retention §11).

Router-local schemas for the single owner's account self-service surface: the readable
account (its delivery/notification routes + the captured email + the email ``verified``
flag), the email-capture/verify body, and the async account-deletion acknowledgement.

Identity is NEVER a field on any of these shapes — the acting athlete is server-derived
from the bearer token (AUTH-R3 / AUTH-R18), so no body carries a caller-identity field; an
injected/forged ``athlete_id`` is rejected by ``additionalProperties:false`` (SCHEMA-R4).
No field is source-shaped or carries a provider name (AUTH-R15), and the account surface
carries no model/tier/catalog control (API-R38).

Requirement IDs: API-R51, AUTH-R3, AUTH-R15, AUTH-R18, GBO-R49, SCHEMA-R4, ERR-R6.
"""

from __future__ import annotations

import datetime as _dt
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from wattwise_core.domain.enums import DeliveryChannel

#: A pragmatic email-address shape (RFC-5321-ish local@domain.tld). A bare ``str`` with a
#: ``pattern`` constraint validates the wire value WITHOUT pulling in the optional
#: ``email-validator`` dependency ``pydantic.EmailStr`` requires; the constraint is the same
#: high-confidence shape the redaction layer recognises (mirrors ``redaction._EMAIL``). An
#: address failing it → ``422 validation-error`` (ERR-R6), never a silent accept.
EMAIL_PATTERN: Final[str] = r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"


class NotificationRouteOut(BaseModel):
    """One per-channel delivery binding on the account (GBO-R49).

    Projects the canonical :class:`NotificationRoute`: the ``channel``, whether it is
    ``enabled`` and ``verified``, and (for non-``web`` channels) the masked address hint.
    The raw address is NOT echoed verbatim — only a redaction-safe presence/mask — so a
    full email never leaves on a read where the channel hint suffices (API-R19 / ERR-R5).
    ``web`` is always-on and verified by construction (no address).
    """

    channel: DeliveryChannel
    enabled: bool
    verified: bool
    address_hint: str | None = None


class UserAccount(BaseModel):
    """The single owner's readable account (``GET /v1/users/me``, doc 60 §8).

    Surfaces the account's delivery/notification routes (GBO-R49), the captured ``email``
    (the address bound to the ``email`` channel, or ``null`` when none is captured yet), and
    the email ``verified`` flag that GATES the digest email channel — a digest e-mail is
    delivered only when the email route is BOTH verified AND enabled (GBO-R49). No source
    name, no model/tier (AUTH-R15 / API-R38).
    """

    email: str | None = Field(default=None, json_schema_extra={"format": "email"})
    verified: bool = False
    notification_routes: list[NotificationRouteOut] = Field(default_factory=list)


class EmailCaptureRequest(BaseModel):
    """``PATCH /v1/users/me`` body — capture/verify the digest email (gates the channel).

    Carries ONLY the email address (validated as an address shape) and no caller-identity
    field (AUTH-R3); ``additionalProperties:false`` (SCHEMA-R4) rejects any unknown/forged
    property, including an injected ``athlete_id`` or a spoofed ``verified`` flag — the
    ``verified`` state is SERVER-controlled (set by the verification flow), never accepted
    from the client. Capturing a NEW/changed address resets ``verified`` to ``false`` so an
    unverified address can never gate the digest email channel open (fail-closed, GBO-R49).
    """

    model_config = ConfigDict(extra="forbid")

    email: str = Field(
        min_length=3, max_length=320, pattern=EMAIL_PATTERN,
        json_schema_extra={"format": "email"},
    )


class AccountDeletionAck(BaseModel):
    """``DELETE /v1/users/me`` acknowledgement — the async erasure request was accepted.

    The DELETE does NOT hard-delete inline (retention §11): it records an account-deletion
    request that a background erasure path fulfils (PRIV-R8 right-to-be-forgotten across
    every store). This body is the durable acknowledgement: ``status=pending_deletion`` and
    the server-stamped ``requested_at``. The erasure itself is asynchronous; the client polls
    no field here beyond the accepted acknowledgement.
    """

    status: str = "pending_deletion"
    requested_at: _dt.datetime


__all__ = [
    "AccountDeletionAck",
    "EmailCaptureRequest",
    "NotificationRouteOut",
    "UserAccount",
]
