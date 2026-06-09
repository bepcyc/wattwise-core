"""Wire shapes for the ``/v1/athlete`` profile + read surface (doc 60 §8.1).

Extracted from the athlete router so the route logic stays within the module-size
ceiling (QUAL-R9(a)). Every field is source-agnostic + number-typed and carries no
provider name (AUTH-R15); the effective FTP/threshold signature and its versioned
history read typed canonical :class:`~wattwise_core.persistence.models.FitnessSignature`
rows (GBO-R26/R27). The change-sport request validates its registry-backed sport code
in the router against the runtime sport registry (GBO-R16a / API-R40).

Requirement IDs: API-R40, AUTH-R3, AUTH-R15, GBO-R16a, GBO-R26, GBO-R27, PAGE-R1,
SCHEMA-R4.
"""

from __future__ import annotations

import datetime as _dt

from pydantic import BaseModel, ConfigDict, Field

from wattwise_core.api.activity_schemas import Page
from wattwise_core.domain.enums import SignatureOrigin


class FitnessSignatureOut(BaseModel):
    """The effective FTP/threshold signature surfaced on the profile (GBO-R26).

    Source-agnostic and number-typed; carries no provider name (AUTH-R15). ``null``
    everywhere when the owner has not yet set a signature for the current sport.
    """

    signature_type: str
    effective_date: _dt.date
    ftp_w: float | None = None
    cp_w: float | None = None
    w_prime_j: float | None = None
    threshold_hr_bpm: int | None = None
    max_hr_bpm: int | None = None
    resting_hr_bpm: int | None = None
    origin: SignatureOrigin


class FitnessSignatureHistory(BaseModel):
    """The cursor-paginated signature history response (doc 60 §8.1, GBO-R26/R27).

    Each item is one effective-dated :class:`FitnessSignature` row in the owner's
    versioned threshold series (newest effective date first); the series is unbounded
    over the athlete's lifetime so it pages by the opaque signed cursor (PAGE-R1/R5),
    never an offset. Source-agnostic + number-typed; carries no provider name (AUTH-R15).
    """

    data: list[FitnessSignatureOut]
    page: Page


class ChangeSportRequest(BaseModel):
    """``POST /v1/athlete/change-sport`` body — the explicit change-sport action (API-R40).

    ``additionalProperties:false`` (SCHEMA-R4) rejects any unknown/forged property (e.g.
    an injected ``athlete_id`` — identity is server-derived, AUTH-R3). The single required
    ``sport`` is a registry-backed code (GBO-R16a); the router validates it against the
    runtime sport registry and rejects an unregistered code ``422 unknown_sport``. Changing
    the current sport is a hint update — it rewrites NO historical activity (API-R40).
    """

    model_config = ConfigDict(extra="forbid")

    sport: str = Field(min_length=1, max_length=64)


__all__ = [
    "ChangeSportRequest",
    "FitnessSignatureHistory",
    "FitnessSignatureOut",
]
