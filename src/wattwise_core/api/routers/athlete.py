"""Athlete profile router — the single-owner ``/v1/athlete`` profile + signature surface.

Serves the one owner's profile (doc 60 §8.1): the readable profile (sex, reference
timezone, current sport, the effective FTP fitness signature), a guarded profile update
(sex / reference timezone / current sport — the change-sport path, API-R40), and the
critical **set-FTP-signature** write that the whole power-analytics stack grounds on
(CTL/TSS/NP/IF/CP all read the effective :class:`FitnessSignature`, GBO-R26/R27 → doc 40).
Without it the analytics surface has no threshold to compute against and degrades to
typed-unavailable; this router is the only first-party way the owner provides it.

Boundary contract enforced here:

- **AUTH-R3 / AUTH-R18** the acting athlete identity is server-derived from the verified
  bearer token (never read from the body/query/path); every read and write acts ONLY on
  that one server-derived id. There is no writable caller-identity field on any request.
- **AUTH-R11** the readable profile requires the ``read`` scope; every mutation
  (``PUT /v1/athlete``, ``PUT /v1/athlete/signature``) requires the ``write`` scope — a
  token without it is ``403 insufficient-scope`` (AUTH-R7), never a silent accept.
- **API-R40** ``current_sport`` is a registry-backed sport code (GBO-R16a), validated
  against the runtime :class:`Sport` registry; an unregistered code → ``422
  validation-error`` with ``errors[].code = "unknown_sport"`` (no new problem type). A
  change-sport is append-only and rewrites NO historical activity (the column is a hint).
- **API-R51** the owner profile exists (it is seeded), so a read returns ``200``; a
  ``signature_type`` naming an unregistered sport is the same ``unknown_sport`` ``422``.

The identity/scope/session dependencies are override seams the app factory wires (FastAPI
``dependency_overrides``), mirroring the performance/activities routers. No field is
source-shaped or carries a provider name (AUTH-R15).

Requirement IDs: API-R40, API-R51, AUTH-R3, AUTH-R7, AUTH-R11, AUTH-R18, GBO-R13,
GBO-R16a, GBO-R26, SCHEMA-R4, ERR-R6, ERR-R8.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.api.errors import FieldError, ProblemError
from wattwise_core.domain.enums import Sex, SignatureOrigin
from wattwise_core.persistence.models import Athlete, FitnessSignature, Sport

router = APIRouter(prefix="/v1/athlete", tags=["athlete"])


# --- dependency seams (overridden by the app factory) ---------------------------


def require_read_scope() -> None:
    """Gate on the ``read`` scope (AUTH-R11); the app factory overrides it (fail-closed)."""
    raise ProblemError("insufficient-scope")  # pragma: no cover - replaced by the app factory


def require_write_scope() -> None:
    """Gate on the ``write`` scope (AUTH-R11); the app factory overrides it (fail-closed)."""
    raise ProblemError("insufficient-scope")  # pragma: no cover - replaced by the app factory


def current_athlete_id() -> str:
    """Server-derived acting athlete id (AUTH-R3); app factory overrides it (fail-closed)."""
    raise ProblemError("unauthenticated")  # pragma: no cover - replaced by the app factory


def current_session() -> AsyncSession:
    """Request-scoped DB session seam; the app factory overrides it (fail-closed)."""
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


_Read = Depends(require_read_scope)
_Write = Depends(require_write_scope)
AthleteId = Annotated[str, Depends(current_athlete_id)]
Session = Annotated[AsyncSession, Depends(current_session)]


# --- wire shapes ----------------------------------------------------------------


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


class AthleteProfile(BaseModel):
    """The single owner's readable profile (doc 60 §8.1).

    ``current_sport`` is a registry-backed code (GBO-R16a), NOT a closed enum;
    ``fitness_signature`` is the effective FTP/threshold for that sport, or ``null``
    when none is set yet (the analytics stack then degrades to typed-unavailable).
    """

    sex: Sex
    reference_timezone: str
    current_sport: str | None = None
    default_response_length: str | None = None
    default_training_load_model: str | None = None
    fitness_signature: FitnessSignatureOut | None = None


class AthleteProfileUpdate(BaseModel):
    """``PUT /v1/athlete`` body — the settable profile fields (API-R40).

    Identity is NOT a field here — it is server-derived (AUTH-R3); a client cannot name
    the athlete it acts as. ``additionalProperties:false`` (SCHEMA-R4) rejects any unknown
    or forged body property (e.g. an injected ``athlete_id``) with a ``422``. Every field
    is optional: a ``PUT`` patches only the fields present (an omitted field is untouched),
    so the same body shape serves a partial update.
    """

    model_config = ConfigDict(extra="forbid")

    sex: Sex | None = None
    reference_timezone: str | None = Field(default=None, min_length=1, max_length=64)
    current_sport: str | None = Field(default=None, min_length=1, max_length=64)


class FitnessSignatureIn(BaseModel):
    """``PUT /v1/athlete/signature`` body — the owner-entered FTP/threshold (GBO-R26).

    ``additionalProperties:false`` (SCHEMA-R4) rejects any unknown/forged property. The
    written row is stamped ``origin = user_entered`` server-side (NOT a client field) so
    provenance can never be spoofed. ``signature_type`` defaults to the owner's current
    sport when omitted, and is validated against the runtime sport registry. The effective
    date defaults to today (UTC) when omitted.
    """

    model_config = ConfigDict(extra="forbid")

    ftp_w: float = Field(gt=0.0, le=2000.0)
    signature_type: str | None = Field(default=None, min_length=1, max_length=64)
    effective_date: _dt.date | None = None
    cp_w: float | None = Field(default=None, gt=0.0, le=2000.0)
    w_prime_j: float | None = Field(default=None, gt=0.0, le=200000.0)
    threshold_hr_bpm: int | None = Field(default=None, gt=0, le=260)
    max_hr_bpm: int | None = Field(default=None, gt=0, le=260)
    resting_hr_bpm: int | None = Field(default=None, gt=0, le=200)


# --- helpers --------------------------------------------------------------------


def _unknown_sport(value: str) -> ProblemError:
    """A ``422 validation-error`` for an unregistered sport code (API-R40; no new type)."""
    return ProblemError(
        "validation-error",
        errors=[FieldError(code="unknown_sport", message="", pointer="/sport")],
    )


async def _load_owner(session: AsyncSession, athlete_id: str) -> Athlete:
    """Load the one server-derived owner row, or fail closed (AUTH-R18 / API-R51).

    Identity is the server-derived id (never client input); the owner is seeded, so a
    miss is an operator-state error surfaced as a generic ``internal-error`` (no leak),
    never a client-facing ``404`` the caller could probe.
    """
    owner = await session.get(Athlete, _uid(athlete_id))
    if owner is None:
        raise ProblemError("internal-error")  # pragma: no cover - the owner is always seeded
    return owner


def _uid(value: str) -> uuid.UUID:
    """Coerce the server-derived athlete id; an unparsable id is an internal error."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError) as exc:  # pragma: no cover - server-derived id is valid
        raise ProblemError("internal-error") from exc


async def _sport_exists(session: AsyncSession, sport_code: str) -> bool:
    """Whether ``sport_code`` is a registered sport (GBO-R16a runtime registry)."""
    found = await session.get(Sport, sport_code)
    return found is not None


async def _effective_signature(
    session: AsyncSession, athlete_id: str, sport: str | None
) -> FitnessSignature | None:
    """The latest-effective signature for ``sport`` today (the analytics resolution, GBO-R27)."""
    if sport is None:
        return None
    stmt = (
        select(FitnessSignature)
        .where(
            FitnessSignature.athlete_id == _uid(athlete_id),
            FitnessSignature.signature_type == sport,
            FitnessSignature.effective_date <= _dt.datetime.now(tz=_dt.UTC).date(),
        )
        .order_by(FitnessSignature.effective_date.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


def _profile(owner: Athlete, signature: FitnessSignature | None) -> AthleteProfile:
    """Project the owner row (+ effective signature) onto the readable profile shape."""
    return AthleteProfile(
        sex=owner.sex,
        reference_timezone=owner.reference_timezone,
        current_sport=owner.current_sport,
        default_response_length=owner.default_response_length,
        default_training_load_model=owner.default_training_load_model,
        fitness_signature=_signature_out(signature),
    )


def _signature_out(sig: FitnessSignature | None) -> FitnessSignatureOut | None:
    if sig is None:
        return None
    return FitnessSignatureOut(
        signature_type=sig.signature_type,
        effective_date=sig.effective_date,
        ftp_w=_f(sig.ftp_w),
        cp_w=_f(sig.cp_w),
        w_prime_j=_f(sig.w_prime_j),
        threshold_hr_bpm=sig.threshold_hr_bpm,
        max_hr_bpm=sig.max_hr_bpm,
        resting_hr_bpm=sig.resting_hr_bpm,
        origin=sig.origin,
    )


def _f(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


# --- §8.1 profile ---------------------------------------------------------------


@router.get(
    "", response_model=AthleteProfile, operation_id="getAthleteProfile", dependencies=[_Read]
)
async def get_profile(session: Session, athlete_id: AthleteId) -> AthleteProfile:
    """Read the one owner's profile + effective FTP signature (doc 60 §8.1)."""
    owner = await _load_owner(session, athlete_id)
    sig = await _effective_signature(session, athlete_id, owner.current_sport)
    return _profile(owner, sig)


@router.put(
    "", response_model=AthleteProfile, operation_id="updateAthleteProfile", dependencies=[_Write]
)
async def update_profile(
    body: AthleteProfileUpdate, session: Session, athlete_id: AthleteId
) -> AthleteProfile:
    """Set sex / reference timezone / current sport (the change-sport path, API-R40).

    A ``current_sport`` is validated against the runtime sport registry (GBO-R16a); an
    unregistered code is rejected ``422 unknown_sport`` BEFORE any write (no partial
    mutation). Changing the current sport is a hint update — it rewrites NO historical
    activity (each :class:`Activity` keeps its own recorded sport, API-R40). Acts ONLY on
    the server-derived owner id (AUTH-R3).
    """
    owner = await _load_owner(session, athlete_id)
    if body.current_sport is not None and not await _sport_exists(session, body.current_sport):
        raise _unknown_sport(body.current_sport)
    if body.sex is not None:
        owner.sex = body.sex
    if body.reference_timezone is not None:
        owner.reference_timezone = body.reference_timezone
    if body.current_sport is not None:
        owner.current_sport = body.current_sport
    await session.flush()
    sig = await _effective_signature(session, athlete_id, owner.current_sport)
    return _profile(owner, sig)


@router.put(
    "/signature",
    response_model=AthleteProfile,
    operation_id="setAthleteSignature",
    dependencies=[_Write],
)
async def set_signature(
    body: FitnessSignatureIn, session: Session, athlete_id: AthleteId
) -> AthleteProfile:
    """Write the owner-entered FTP fitness signature the power analytics ground on (GBO-R26).

    This is the load-bearing write: CTL/TSS/NP/IF/CP all resolve the effective
    :class:`FitnessSignature` for the activity's sport (doc 40), so without it the power
    surface degrades to typed-unavailable. The row is stamped ``origin = user_entered``
    SERVER-side (never a client field) and keyed by ``(athlete_id, effective_date,
    signature_type)``: re-setting the SAME effective date for the SAME sport UPDATES that
    row in place rather than violating the uniqueness key. ``signature_type`` defaults to
    the owner's current sport and is validated against the sport registry (GBO-R16a); an
    unregistered code → ``422 unknown_sport``. Acts ONLY on the server-derived owner id.
    """
    owner = await _load_owner(session, athlete_id)
    sport = body.signature_type or owner.current_sport
    if sport is None:
        raise ProblemError(
            "validation-error",
            errors=[
                FieldError(
                    code="signature_type_required", message="", pointer="/signature_type"
                )
            ],
        )
    if not await _sport_exists(session, sport):
        raise _unknown_sport(sport)
    effective = body.effective_date or _dt.datetime.now(tz=_dt.UTC).date()
    existing = await _exact_signature(session, athlete_id, sport, effective)
    if existing is None:
        session.add(_new_signature(athlete_id, sport, effective, body))
    else:
        _apply_signature(existing, body)
    await session.flush()
    sig = await _effective_signature(session, athlete_id, owner.current_sport)
    return _profile(owner, sig)


async def _exact_signature(
    session: AsyncSession, athlete_id: str, sport: str, effective: _dt.date
) -> FitnessSignature | None:
    """The signature with the EXACT natural key, if any (the idempotent upsert target)."""
    stmt = select(FitnessSignature).where(
        FitnessSignature.athlete_id == _uid(athlete_id),
        FitnessSignature.signature_type == sport,
        FitnessSignature.effective_date == effective,
    )
    return (await session.execute(stmt)).scalar_one_or_none()


def _new_signature(
    athlete_id: str, sport: str, effective: _dt.date, body: FitnessSignatureIn
) -> FitnessSignature:
    return FitnessSignature(
        athlete_id=_uid(athlete_id),
        signature_type=sport,
        effective_date=effective,
        ftp_w=body.ftp_w,
        cp_w=body.cp_w,
        w_prime_j=body.w_prime_j,
        threshold_hr_bpm=body.threshold_hr_bpm,
        max_hr_bpm=body.max_hr_bpm,
        resting_hr_bpm=body.resting_hr_bpm,
        origin=SignatureOrigin.USER_ENTERED,
    )


def _apply_signature(sig: FitnessSignature, body: FitnessSignatureIn) -> None:
    """Overwrite an existing same-key signature with the new owner-entered values."""
    sig.ftp_w = body.ftp_w
    sig.cp_w = body.cp_w
    sig.w_prime_j = body.w_prime_j
    sig.threshold_hr_bpm = body.threshold_hr_bpm
    sig.max_hr_bpm = body.max_hr_bpm
    sig.resting_hr_bpm = body.resting_hr_bpm
    sig.origin = SignatureOrigin.USER_ENTERED


__all__ = [
    "AthleteProfile",
    "AthleteProfileUpdate",
    "FitnessSignatureIn",
    "FitnessSignatureOut",
    "current_athlete_id",
    "current_session",
    "require_read_scope",
    "require_write_scope",
    "router",
]
