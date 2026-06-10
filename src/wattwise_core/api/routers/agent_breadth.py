"""Agent BREADTH router — diagnose, the weekly-digest subscription, and the memory seam.

The focused sibling of :mod:`wattwise_core.api.routers.agent_routes` (QUAL-R9 size split) that owns
the remaining ``/v1/agent`` surfaces doc 60 §6/§7 specifies beyond ``/ask`` / ``/readiness`` /
``/decision``. ``agent_routes`` mounts this router onto its own ``/v1/agent`` router so the single
``include_router(agent_routes.router)`` the app factory performs picks up every agent endpoint, and
re-exports the new seams so the factory wires them exactly as before. NO grounding, NO model call,
and NO graph topology lives here — every surface reaches the injected engine ONLY through the typed
:class:`BreadthEngine` seam (ARCH-R21) and shapes the request / enforces the boundary contract.

Endpoints (all server-derived identity, AUTH-R3 / ARCH-R16 — never a client field):

- **``POST /v1/agent/diagnose``** (scope ``agent``, API-R15) — the DETERMINISTIC data-quality /
  coverage narration. Fails closed over the canonical coverage envelope (no fabrication, GROUND-R7).
- **``POST /v1/agent/digest/subscribe``** / **``GET …/digest/list``** / **``DELETE
  …/digest/subscribe/{id}``** (API-R14) — manage the owner's ONE standing weekly-digest schedule
  over the canonical :class:`~wattwise_core.persistence.models.notify.DigestSubscription`. The
  ``email`` channel is GATED: a subscription that names ``email`` before the owner's email is
  verified (the ``/v1/users/me`` ``email`` :class:`NotificationRoute` ``verified`` flag, GBO-R49) is
  refused ``422`` — fail-closed so an unverified address can never gate the channel open.
- **``GET …/digest/last``** (API-R14) — the most-recent grounded weekly digest body, server-side
  sanitized (API-R13); abstains visibly (``degraded`` + caveat) on a week with no canonical inputs.
- **``GET /v1/agent/memory``** / **``GET …/memory/{id}``** / **``DELETE …/memory/{id}``**
  (API-R15a / MEM-R3 MUST) — the athlete-scoped per-item memory read + erase. NON-LLM and OUTSIDE
  the agent cost gate; a per-item erase is a privacy MUST (PRIV-R8) and a re-GET of an erased /
  foreign / unknown id is ``404`` (never disclosed). Scope ``agent`` (the memory surface is part of
  the agent product, AUTH-R13).

Requirement IDs: API-R14, API-R15, API-R15a, AUTH-R3, AUTH-R13, AUTH-R18, ARCH-R16, ARCH-R21,
GBO-R46, GBO-R47, GBO-R49, MEM-R1, MEM-R3, PRIV-R8, LIMIT-R2, SCHEMA-R4, API-R11c, API-R13, API-R51.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import Sequence
from typing import Annotated, Protocol, runtime_checkable

from fastapi import APIRouter, Depends, Header, Path, Query, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.agent.deliverables import Digest
from wattwise_core.agent.diagnose_deliverable import AgentDiagnosis
from wattwise_core.agent.memory import RecalledItem
from wattwise_core.api.errors import ProblemError, resolve_trace_id
from wattwise_core.api.problems import not_found
from wattwise_core.api.ratelimit import LimitClass, RateLimiter
from wattwise_core.api.routers.agent_routes import (
    agent_engine,
    attached_entitlement,
    current_athlete_id,
    rate_limiter,
    require_agent_scope,
)
from wattwise_core.api.routers.agent_schemas import (
    AgentDiagnosisResponse,
    DigestBody,
    DigestSubscribeRequest,
    DigestSubscriptionList,
    DigestSubscriptionOut,
    MemoryEraseAck,
    MemoryItemList,
    MemoryItemOut,
    memory_item_out,
    render_diagnosis,
    render_digest,
)
from wattwise_core.domain.enums import (
    DeliveryChannel,
    DigestCadence,
    DigestStatus,
    Weekday,
)
from wattwise_core.entitlement import Entitlements
from wattwise_core.persistence.models import DigestSubscription, NotificationRoute

# No prefix: this router is mounted ONTO the ``/v1/agent`` router in :mod:`agent_routes` (which
# prepends that prefix), so the route paths below are relative to ``/v1/agent``.
router = APIRouter(tags=["agent"])

#: The languages this surface localizes athlete-facing copy into (API-R37).
_SUPPORTED_LOCALES = frozenset({"en", "de", "ru"})


# --- engine seam (the breadth methods; reached only through this Protocol, ARCH-R21) ---


@runtime_checkable
class BreadthEngine(Protocol):
    """The diagnose / digest / memory engine seam this router drives (ARCH-R21).

    The concrete engine is the SAME ``GraphAgentEngine`` the ``/v1/agent`` surface drives; this
    router reaches it ONLY through this typed seam. ``diagnose`` is DETERMINISTIC over canonical
    coverage (API-R15); ``digest`` projects the grounded weekly review (API-R14); the memory methods
    are the athlete-scoped per-item read + erase over the dedicated agent-state store (MEM-R3).
    Every ``athlete_id`` is passed server-derived (AUTH-R3) and never trusted from a client.
    """

    async def diagnose(self, *, athlete_id: str, locale: str) -> AgentDiagnosis: ...

    async def digest(
        self, *, athlete_id: str, week_end: str, entitlement: Entitlements | None = None
    ) -> Digest: ...

    async def list_memory(
        self, *, athlete_id: str, limit: int, offset: int
    ) -> Sequence[RecalledItem]: ...

    async def get_memory(self, *, athlete_id: str, memory_item_id: str) -> RecalledItem | None: ...

    async def delete_memory(self, *, athlete_id: str, memory_item_id: str) -> bool: ...


# --- dependency seams ------------------------------------------------------------
# Identity / scope / engine / rate-limiter are the SAME fail-closed seams agent_routes
# declares (the app factory's existing overrides apply to both routers); only the DB
# session seam is new here (digest persistence + the email-verified gate).


def current_session() -> AsyncSession:
    """Request-scoped canonical DB session seam; the app factory overrides it (fail-closed)."""
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


def _breadth_engine(engine: Annotated[object, Depends(agent_engine)]) -> BreadthEngine:
    """Type the shared injected engine as the :class:`BreadthEngine` seam (ARCH-R21).

    Reuses ``agent_routes.agent_engine`` (the app factory's single override) so the breadth surfaces
    drive the SAME concrete engine the rest of ``/v1/agent`` does, surfaced through the typed
    diagnose/digest/memory seam without a second wiring point.
    """
    return engine  # type: ignore[return-value]


_Agent = Depends(require_agent_scope)
AthleteId = Annotated[str, Depends(current_athlete_id)]
Engine = Annotated[BreadthEngine, Depends(_breadth_engine)]
Limiter = Annotated[RateLimiter, Depends(rate_limiter)]
Session = Annotated[AsyncSession, Depends(current_session)]


def _header_locale(accept_language: str | None) -> str:
    """The first supported ``Accept-Language`` tag (en/de/ru), else the default ``en`` (API-R37)."""
    if accept_language:
        for part in accept_language.split(","):
            tag = part.split(";", 1)[0].strip().lower()[:2]
            if tag in _SUPPORTED_LOCALES:
                return tag
    return "en"


def _uid(value: str) -> uuid.UUID:
    """Coerce the server-derived athlete id; an unparsable id reads as a not-found scope."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise not_found() from exc


# --- POST /v1/agent/diagnose — deterministic coverage narration (API-R15) --------


@router.post(
    "/diagnose",
    response_model=AgentDiagnosisResponse,
    dependencies=[_Agent],
    operation_id="agentDiagnose",
)
async def agent_diagnose(
    request: Request,
    engine: Engine,
    athlete_id: AthleteId,
    limiter: Limiter,
    accept_language: Annotated[str | None, Header()] = None,
) -> AgentDiagnosisResponse:
    """Narrate the athlete's canonical data-quality / coverage, fail-closed (API-R15).

    Requires the ``agent`` scope (AUTH-R13) and debits the per-athlete ``agent`` rate bucket
    (LIMIT-R2) keyed by the server-derived id (AUTH-R3). DETERMINISTIC: the engine probes each
    canonical input and projects the typed ``present|stale|missing`` envelope with NO model call
    and nothing to fabricate (GROUND-R7). ``completed`` when at least one input is present, else
    ``degraded`` with the typed ``no_canonical_coverage`` caveat (OUTCOME-R3). Carries NO
    athlete-facing numbers (VOICE-R7) and NO billing/model machinery (API-R11c).
    """
    limiter.check(athlete_id, LimitClass.AGENT)
    trace_id = resolve_trace_id(request)
    locale = _header_locale(accept_language)
    diagnosis = await engine.diagnose(athlete_id=athlete_id, locale=locale)
    return render_diagnosis(diagnosis, trace_id)


# --- /v1/agent/digest/* — the standing weekly-digest schedule (API-R14) ----------


def _last_week_end(today: _dt.date) -> str:
    """The most-recent COMPLETED week-end (the last Sunday on/before today), ISO (API-R14).

    The digest reviews a trailing calendar week; ``GET …/digest/last`` with no explicit week
    resolves to the latest completed Sun-ending week so the body is deterministic.
    """
    return (today - _dt.timedelta(days=(today.weekday() + 1) % 7)).isoformat()


@router.get(
    "/digest/last",
    response_model=DigestBody,
    dependencies=[_Agent],
    operation_id="agentDigestLast",
)
async def agent_digest_last(
    request: Request,
    engine: Engine,
    athlete_id: AthleteId,
    limiter: Limiter,
    accept_language: Annotated[str | None, Header()] = None,
    week_end: Annotated[str | None, Query()] = None,
) -> DigestBody:
    """Read the most-recent grounded weekly digest (API-R14); sanitized + fail-closed.

    Requires the ``agent`` scope (AUTH-R13) and debits the ``agent`` rate bucket (LIMIT-R2). With no
    ``week_end`` the latest completed Sun-ending week is used. The grounded body is server-side
    sanitized (API-R13) and abstains VISIBLY (``degraded`` + localized caveat) on a week with no
    canonical inputs (OUTCOME-R3) — never a guessed number (GROUND-R7). Identity is server-derived
    (AUTH-R3); carries NO billing/model machinery (API-R11c).
    """
    limiter.check(athlete_id, LimitClass.AGENT)
    trace_id = resolve_trace_id(request)
    locale = _header_locale(accept_language)
    resolved = week_end or _last_week_end(_dt.datetime.now(_dt.UTC).date())
    digest = await engine.digest(
        athlete_id=athlete_id, week_end=resolved, entitlement=attached_entitlement(request)
    )
    return render_digest(digest, trace_id, locale)


async def _email_unverified(session: AsyncSession, athlete_uuid: uuid.UUID) -> bool:
    """True iff the owner's ``email`` channel is NOT verified (the digest-email gate, GBO-R49).

    Reads the canonical ``email`` :class:`NotificationRoute`; a digest e-mail is delivered only when
    the address is verified (GBO-R49). A missing route OR an unverified one fails the gate, so a
    subscription naming ``email`` before verification is refused — an unverified address can NEVER
    gate the email channel open (fail-closed).
    """
    stmt = select(NotificationRoute).where(
        NotificationRoute.athlete_id == athlete_uuid,
        NotificationRoute.channel == DeliveryChannel.EMAIL,
    )
    route = (await session.execute(stmt)).scalar_one_or_none()
    return route is None or not route.verified


def _subscription_out(row: DigestSubscription) -> DigestSubscriptionOut:
    """Project a canonical digest subscription onto the wire shape (API-R14 / GBO-R46)."""
    return DigestSubscriptionOut(
        subscription_id=str(row.subscription_id),
        cadence=row.cadence.value,
        weekday=row.weekday.value if row.weekday is not None else None,
        hour_local=row.hour_local,
        channels=list(row.channels),
        status=row.status.value,
    )


@router.post(
    "/digest/subscribe",
    response_model=DigestSubscriptionOut,
    dependencies=[_Agent],
    operation_id="agentDigestSubscribe",
    status_code=status.HTTP_200_OK,
)
async def agent_digest_subscribe(
    body: DigestSubscribeRequest,
    athlete_id: AthleteId,
    session: Session,
) -> DigestSubscriptionOut:
    """Create the owner's standing weekly-digest schedule (API-R14 / GBO-R46).

    Requires the ``agent`` scope (AUTH-R13). Persists ONE standing canonical
    :class:`DigestSubscription` for the server-derived owner (AUTH-R3) — identity is never a body
    field (SCHEMA-R4). The schedule is bounded to ONE active standing row (GBO-R46): re-subscribing
    UPDATES the owner's existing active row in place rather than creating a duplicate (so two POSTs
    never leave two active schedules). The ``email`` channel is GATED: if the subscription names
    ``email`` while the owner's email is unverified (GBO-R49), it is refused ``422``
    ``validation-error`` so an unverified address can never gate the channel open (fail-closed).
    ``hour_local`` is the athlete-LOCAL firing hour (GBO-R47), never UTC.
    """
    athlete_uuid = _uid(athlete_id)
    if "email" in body.channels and await _email_unverified(session, athlete_uuid):
        raise ProblemError("validation-error")
    row = await _active_subscription(session, athlete_uuid)
    if row is None:
        row = DigestSubscription(athlete_id=athlete_uuid, status=DigestStatus.ACTIVE)
        session.add(row)
    row.cadence = DigestCadence(body.cadence)
    row.weekday = Weekday(body.weekday) if body.weekday is not None else None
    row.hour_local = body.hour_local
    row.channels = list(body.channels)
    row.status = DigestStatus.ACTIVE
    await session.flush()
    return _subscription_out(row)


async def _active_subscription(
    session: AsyncSession, athlete_uuid: uuid.UUID
) -> DigestSubscription | None:
    """The owner's single ACTIVE standing digest schedule, if one exists (GBO-R46).

    The schedule is bounded to ONE active row per owner (GBO-R46), so a re-subscribe UPDATES this
    row in place instead of inserting a duplicate (the portable "ONE standing schedule" guarantee —
    an UPDATE-then-INSERT, since the surrogate-PK table carries no natural unique key to upsert on).
    Newest-first so even a pre-existing duplicate (from before this guard) resolves to a single
    deterministic active row to update.
    """
    stmt = (
        select(DigestSubscription)
        .where(
            DigestSubscription.athlete_id == athlete_uuid,
            DigestSubscription.status == DigestStatus.ACTIVE,
        )
        .order_by(DigestSubscription.created_at.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


@router.get(
    "/digest/list",
    response_model=DigestSubscriptionList,
    dependencies=[_Agent],
    operation_id="agentDigestList",
)
async def agent_digest_list(athlete_id: AthleteId, session: Session) -> DigestSubscriptionList:
    """List the owner's standing digest subscriptions (API-R14).

    Requires the ``agent`` scope (AUTH-R13). Reads ONLY the server-derived owner's subscriptions
    (AUTH-R3), newest first; never another athlete's. The owner's digest schedule is bounded (ONE
    standing schedule, GBO-R46) so the list is small and unpaginated by design.
    """
    stmt = (
        select(DigestSubscription)
        .where(DigestSubscription.athlete_id == _uid(athlete_id))
        .order_by(DigestSubscription.created_at.desc())
    )
    rows = (await session.execute(stmt)).scalars().all()
    return DigestSubscriptionList(data=[_subscription_out(r) for r in rows])


@router.delete(
    "/digest/subscribe/{subscription_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[_Agent],
    operation_id="agentDigestUnsubscribe",
)
async def agent_digest_unsubscribe(
    athlete_id: AthleteId,
    session: Session,
    subscription_id: Annotated[str, Path()],
) -> None:
    """Cancel the owner's standing digest subscription by id (API-R14 / GBO-R47).

    Requires the ``agent`` scope (AUTH-R13). Closing sets the terminal ``cancelled`` status
    (GBO-R47) on the owner's row ONLY (scoped to the server-derived id, AUTH-R3): an unknown /
    foreign / non-UUID id is ``404`` ``not-found`` (API-R51), never disclosed. Returns ``204``.
    """
    row = await _owned_subscription(session, athlete_id, subscription_id)
    if row is None:
        raise not_found()
    row.status = DigestStatus.CANCELLED
    await session.flush()


async def _owned_subscription(
    session: AsyncSession, athlete_id: str, subscription_id: str
) -> DigestSubscription | None:
    """The owner's subscription by id, scoped to the server-derived athlete, else ``None``.

    Looks up by BOTH the id AND the owner ``athlete_id`` (AUTH-R3): a foreign / unknown / non-UUID
    id matches no row the caller owns and reads as absent (the router maps that to a 404, never
    disclosing a foreign row).
    """
    try:
        sub_uuid = uuid.UUID(subscription_id)
    except (ValueError, AttributeError):
        return None
    stmt = select(DigestSubscription).where(
        DigestSubscription.subscription_id == sub_uuid,
        DigestSubscription.athlete_id == _uid(athlete_id),
    )
    return (await session.execute(stmt)).scalar_one_or_none()


# --- /v1/agent/memory/* — the per-item read + erase seam (API-R15a / MEM-R3) -----


@router.get(
    "/memory",
    response_model=MemoryItemList,
    dependencies=[_Agent],
    operation_id="agentMemoryList",
)
async def agent_memory_list(
    athlete_id: AthleteId,
    engine: Engine,
    *,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> MemoryItemList:
    """List the owner's durable memory rows, newest first (API-R15a / MEM-R3).

    Requires the ``agent`` scope (AUTH-R13). NON-LLM and OUTSIDE the agent cost gate (a memory read
    debits no coaching budget). Scoped STRICTLY to the server-derived owner (AUTH-R3 / MEM-R3) —
    another athlete's rows are never listed. Returns personalization context only, never a canonical
    number (MEM-R1). ``limit`` is bounded ``[1, 200]`` and ``offset`` pages the newest-first list.
    """
    rows = await engine.list_memory(athlete_id=athlete_id, limit=limit, offset=offset)
    return MemoryItemList(data=[memory_item_out(r) for r in rows])


@router.get(
    "/memory/{memory_item_id}",
    response_model=MemoryItemOut,
    dependencies=[_Agent],
    operation_id="agentMemoryGet",
)
async def agent_memory_get(
    athlete_id: AthleteId,
    engine: Engine,
    memory_item_id: Annotated[str, Path()],
) -> MemoryItemOut:
    """Fetch ONE durable memory row by id, scoped to the owner (API-R15a / MEM-R3, fail-closed).

    Requires the ``agent`` scope (AUTH-R13). NON-LLM / outside the cost gate. Looks up by BOTH the
    id AND the server-derived ``athlete_id`` (AUTH-R3): a foreign / unknown / non-UUID id is
    ``404`` ``not-found`` (API-R51), indistinguishable from truly absent and never disclosed
    (MEM-R3). Returns personalization context only, never a canonical number (MEM-R1).
    """
    item = await engine.get_memory(athlete_id=athlete_id, memory_item_id=memory_item_id)
    if item is None:
        raise not_found()
    return memory_item_out(item)


@router.delete(
    "/memory/{memory_item_id}",
    response_model=MemoryEraseAck,
    dependencies=[_Agent],
    operation_id="agentMemoryErase",
)
async def agent_memory_erase(
    athlete_id: AthleteId,
    engine: Engine,
    memory_item_id: Annotated[str, Path()],
) -> MemoryEraseAck:
    """Erase ONE durable memory row by id, scoped to the owner (API-R15a / MEM-R3 MUST / PRIV-R8).

    Requires the ``agent`` scope (AUTH-R13). NON-LLM / outside the cost gate. The guarded delete
    matches BOTH the id AND the server-derived ``athlete_id`` (AUTH-R3): a cross-athlete / unknown /
    non-UUID id erases nothing and is ``404`` ``not-found`` (API-R51), never disclosed. A successful
    erase removes the residual row entirely (PRIV-R8) so a re-GET of the id is ``404``. Per-item
    erase is a privacy MUST (MEM-R3).
    """
    erased = await engine.delete_memory(athlete_id=athlete_id, memory_item_id=memory_item_id)
    if not erased:
        raise not_found()
    return MemoryEraseAck(memory_item_id=memory_item_id)


__all__ = [
    "BreadthEngine",
    "current_session",
    "router",
]
