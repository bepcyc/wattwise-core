"""User-settings router — the athlete-facing ``/v1/user-settings`` preferences surface.

Serves the one owner's own preferences (doc 60 §8.10): training **zones**, the **default
training-load model**, the persisted **language**, and the persisted **answer-length**
(``response_length``) default the coaching agent applies when an ``AgentAskRequest`` gives
no per-request override (API-R11f/API-R37). Each setting has a ``GET`` (``read`` scope) and
a ``PUT`` (``write`` scope). These are athlete-facing preferences ONLY: there is NO model /
tier / token-budget control on any of them (API-R38) — ``response_length`` expresses only
*how much detail the athlete wants back*, never a verbosity-machinery knob (API-R11c).

Persistence (API-R32 — no orphan writes; every PUT backs a real entity):

- ``zones`` → the canonical effective-dated :class:`TrainingZoneSet` (GBO-R13d).
- ``language`` → the athlete profile ``primary_locale`` column (master data, API-R37).
- ``default-load-model`` → the athlete profile ``default_training_load_model`` column,
  validated against the doc-40 LOAD-R2 closed set ``{power_tss, hr_load, hr_load_zonal}``.
- ``response-length`` → the athlete profile ``default_response_length`` column (the durable
  agent-interaction preference, doc 50 VOICE-R8 / API-R11f) — an unsupported value → ``422``.

Boundary contract: identity is server-derived from the bearer token (AUTH-R3/R18) and every
read/write acts ONLY on that one owner id — no writable caller-identity field exists on any
request body (SCHEMA-R4). Reads require ``read``; writes require ``write`` (AUTH-R11), so a
token without ``write`` is ``403 insufficient-scope`` (AUTH-R7). No field is source-shaped or
carries a provider name (AUTH-R15), and no response carries a model/tier/catalog (API-R38).

Requirement IDs: API-R11c, API-R11f, API-R32, API-R37, API-R38, AUTH-R3, AUTH-R7, AUTH-R11,
AUTH-R18, GBO-R13d, LOAD-R2, SCHEMA-R4, ERR-R6.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import Annotated, Final, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.api.errors import FieldError, ProblemError
from wattwise_core.domain.enums import ZoneBasis, ZoneKind
from wattwise_core.persistence.models import Athlete, TrainingZoneSet

router = APIRouter(prefix="/v1/user-settings", tags=["user-settings"])

#: The athlete-facing answer-length tokens (API-R11f); mirrors the agent ResponseLength.
ResponseLength = Literal["short", "standard", "detailed"]
_RESPONSE_LENGTHS: Final[frozenset[str]] = frozenset({"short", "standard", "detailed"})

#: The languages this surface persists/serves (API-R37); EN/DE/RU out of the box.
Language = Literal["en", "de", "ru"]

#: The doc-40 LOAD-R2 closed selectable load-model set; a PUT validates against exactly it.
_LOAD_MODELS: Final[frozenset[str]] = frozenset({"power_tss", "hr_load", "hr_load_zonal"})


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


class ZoneBoundary(BaseModel):
    """One ordered, contiguous training-zone boundary (GBO-R13d)."""

    model_config = ConfigDict(extra="forbid")

    zone_index: int = Field(ge=0)
    label: str = Field(min_length=1, max_length=64)
    lower: float
    upper: float | None = None


class ZonesSettings(BaseModel):
    """The athlete's power/HR zone definitions (GBO-R13d).

    ``kind`` selects the zone family (``power``/``hr``); ``basis`` whether the boundaries
    are absolute (watts/bpm) or relative (a fraction of FTP/threshold). ``boundaries`` is
    the ordered, non-overlapping list. ``additionalProperties:false`` (SCHEMA-R4) rejects a
    forged property (e.g. an injected ``athlete_id``).
    """

    model_config = ConfigDict(extra="forbid")

    kind: ZoneKind = ZoneKind.POWER
    basis: ZoneBasis = ZoneBasis.ABSOLUTE
    boundaries: list[ZoneBoundary] = Field(default_factory=list)


class LanguageSettings(BaseModel):
    """The athlete's persisted language preference (API-R37)."""

    model_config = ConfigDict(extra="forbid")

    language: Language


class ResponseLengthSettings(BaseModel):
    """The athlete's persisted answer-length preference (API-R11f).

    Expresses ONLY how much detail the athlete wants back (short/standard/detailed) — it is
    NOT a model/tier/budget/verbosity-machinery control (API-R11c/API-R38).
    """

    model_config = ConfigDict(extra="forbid")

    response_length: ResponseLength


class DefaultLoadModelSettings(BaseModel):
    """The athlete's default training-load **analysis** model (a sports-science choice).

    A member of the doc-40 LOAD-R2 set — NOT an LLM/model-tier choice (API-R38). ``null``
    means the engine's automatic per-activity selection (LOAD-R3) applies.
    """

    model_config = ConfigDict(extra="forbid")

    default_load_model: str | None = Field(default=None, max_length=64)


# --- helpers --------------------------------------------------------------------


def _uid(value: str) -> uuid.UUID:
    """Coerce the server-derived athlete id; an unparsable id is an internal error."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError) as exc:  # pragma: no cover - server-derived id is valid
        raise ProblemError("internal-error") from exc


async def _load_owner(session: AsyncSession, athlete_id: str) -> Athlete:
    """Load the one server-derived owner row, or fail closed (AUTH-R18 / API-R51)."""
    owner = await session.get(Athlete, _uid(athlete_id))
    if owner is None:
        raise ProblemError("internal-error")  # pragma: no cover - the owner is always seeded
    return owner


def _unsupported(field: str, code: str) -> ProblemError:
    """A ``422 validation-error`` for an unsupported enum value (ERR-R6)."""
    return ProblemError(
        "validation-error",
        errors=[FieldError(code=code, message="", pointer=f"/{field}")],
    )


# --- §8.10 zones ----------------------------------------------------------------


@router.get(
    "/zones", response_model=ZonesSettings, operation_id="getUserZones", dependencies=[_Read]
)
async def get_zones(session: Session, athlete_id: AthleteId) -> ZonesSettings:
    """Read the owner's latest-effective power zones (GBO-R13d); empty when none set."""
    latest = await _latest_zone_set(session, athlete_id)
    if latest is None:
        return ZonesSettings()
    return ZonesSettings(
        kind=latest.zone_kind, basis=latest.basis, boundaries=_boundaries_out(latest.boundaries)
    )


@router.put(
    "/zones", response_model=ZonesSettings, operation_id="putUserZones", dependencies=[_Write]
)
async def put_zones(
    body: ZonesSettings, session: Session, athlete_id: AthleteId
) -> ZonesSettings:
    """Write the owner's zone definitions as a today-effective :class:`TrainingZoneSet`.

    Backs a real canonical entity (API-R32): a new effective interval for ``(athlete_id,
    zone_kind, today)`` — re-setting the SAME kind today UPDATES that row rather than
    violating its uniqueness key. Acts ONLY on the server-derived owner id (AUTH-R3).
    """
    await _load_owner(session, athlete_id)
    today = _dt.datetime.now(tz=_dt.UTC).date()
    existing = await _exact_zone_set(session, athlete_id, body.kind, today)
    boundaries = [b.model_dump() for b in body.boundaries]
    if existing is None:
        session.add(
            TrainingZoneSet(
                athlete_id=_uid(athlete_id), zone_kind=body.kind, effective_date=today,
                basis=body.basis, boundaries=boundaries,
            )
        )
    else:
        existing.basis = body.basis
        existing.boundaries = boundaries
    await session.flush()
    return ZonesSettings(kind=body.kind, basis=body.basis, boundaries=body.boundaries)


async def _latest_zone_set(
    session: AsyncSession, athlete_id: str
) -> TrainingZoneSet | None:
    """The most-recent effective zone set for the owner (any kind), or ``None``."""
    stmt = (
        select(TrainingZoneSet)
        .where(TrainingZoneSet.athlete_id == _uid(athlete_id))
        .order_by(TrainingZoneSet.effective_date.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def _exact_zone_set(
    session: AsyncSession, athlete_id: str, kind: ZoneKind, effective: _dt.date
) -> TrainingZoneSet | None:
    """The zone set with the EXACT natural key, if any (the idempotent upsert target)."""
    stmt = select(TrainingZoneSet).where(
        TrainingZoneSet.athlete_id == _uid(athlete_id),
        TrainingZoneSet.zone_kind == kind,
        TrainingZoneSet.effective_date == effective,
    )
    return (await session.execute(stmt)).scalar_one_or_none()


def _boundaries_out(raw: list[dict[str, object]]) -> list[ZoneBoundary]:
    return [ZoneBoundary.model_validate(b) for b in raw]


# --- §8.10 language -------------------------------------------------------------


@router.get(
    "/language",
    response_model=LanguageSettings,
    operation_id="getUserLanguage",
    dependencies=[_Read],
)
async def get_language(session: Session, athlete_id: AthleteId) -> LanguageSettings:
    """Read the owner's persisted language; defaults to ``en`` when unset (API-R37)."""
    owner = await _load_owner(session, athlete_id)
    lang = owner.primary_locale if owner.primary_locale in ("en", "de", "ru") else "en"
    return LanguageSettings(language=lang)


@router.put(
    "/language",
    response_model=LanguageSettings,
    operation_id="putUserLanguage",
    dependencies=[_Write],
)
async def put_language(
    body: LanguageSettings, session: Session, athlete_id: AthleteId
) -> LanguageSettings:
    """Persist the owner's language preference on the profile (API-R37/API-R32)."""
    owner = await _load_owner(session, athlete_id)
    owner.primary_locale = body.language
    await session.flush()
    return LanguageSettings(language=body.language)


# --- §8.10 response-length ------------------------------------------------------


@router.get(
    "/response-length",
    response_model=ResponseLengthSettings,
    operation_id="getUserResponseLength",
    dependencies=[_Read],
)
async def get_response_length(
    session: Session, athlete_id: AthleteId
) -> ResponseLengthSettings:
    """Read the owner's persisted answer-length; defaults to ``standard`` (API-R11f)."""
    owner = await _load_owner(session, athlete_id)
    stored = owner.default_response_length
    value = stored if stored in _RESPONSE_LENGTHS else "standard"
    return ResponseLengthSettings(response_length=value)


@router.put(
    "/response-length",
    response_model=ResponseLengthSettings,
    operation_id="putUserResponseLength",
    dependencies=[_Write],
)
async def put_response_length(
    body: ResponseLengthSettings, session: Session, athlete_id: AthleteId
) -> ResponseLengthSettings:
    """Persist the owner's answer-length default on the profile (API-R11f/API-R32).

    This is the default applied to every agent answer/deliverable when an
    ``AgentAskRequest`` gives no per-request ``response_length``; a per-request value
    overrides for that one call WITHOUT mutating this stored default.
    """
    owner = await _load_owner(session, athlete_id)
    owner.default_response_length = body.response_length
    await session.flush()
    return ResponseLengthSettings(response_length=body.response_length)


# --- §8.10 default training-load model ------------------------------------------


@router.get(
    "/default-load-model",
    response_model=DefaultLoadModelSettings,
    operation_id="getUserDefaultLoadModel",
    dependencies=[_Read],
)
async def get_default_load_model(
    session: Session, athlete_id: AthleteId
) -> DefaultLoadModelSettings:
    """Read the owner's default training-load model; ``null`` = automatic (LOAD-R3)."""
    owner = await _load_owner(session, athlete_id)
    return DefaultLoadModelSettings(default_load_model=owner.default_training_load_model)


@router.put(
    "/default-load-model",
    response_model=DefaultLoadModelSettings,
    operation_id="putUserDefaultLoadModel",
    dependencies=[_Write],
)
async def put_default_load_model(
    body: DefaultLoadModelSettings, session: Session, athlete_id: AthleteId
) -> DefaultLoadModelSettings:
    """Persist the owner's default load model (a sports-science choice, NOT an LLM tier).

    Validated against the doc-40 LOAD-R2 closed set ``{power_tss, hr_load, hr_load_zonal}``;
    any other token → ``422 validation-error`` (``unsupported_load_model``). ``null`` clears
    the preference so the engine's automatic LOAD-R3 selection applies. Acts ONLY on the
    server-derived owner id (AUTH-R3).
    """
    owner = await _load_owner(session, athlete_id)
    if body.default_load_model is not None and body.default_load_model not in _LOAD_MODELS:
        raise _unsupported("default_load_model", "unsupported_load_model")
    owner.default_training_load_model = body.default_load_model
    await session.flush()
    return DefaultLoadModelSettings(default_load_model=body.default_load_model)


__all__ = [
    "DefaultLoadModelSettings",
    "LanguageSettings",
    "ResponseLengthSettings",
    "ZoneBoundary",
    "ZonesSettings",
    "current_athlete_id",
    "current_session",
    "require_read_scope",
    "require_write_scope",
    "router",
]
