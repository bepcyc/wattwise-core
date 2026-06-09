"""Agent router — the grounded ``POST /v1/agent/ask`` surface + its SSE stream.

Serves the Phase-1 coaching agent over HTTP: a natural-language question in, a
grounded, sanitized, status-discriminated answer out (doc 60 §7). It is the thin API
projection of the agent deliverables (the :func:`answer_question` projection of the
``deliverables`` module, reached ONLY through the injected :class:`AgentEngine` seam,
ARCH-R21) — this router owns NO grounding, NO model call, and NO graph topology; it
shapes the request, enforces the boundary contract, and renders the engine's terminal
outcome.

Boundary contract enforced here:

- **AUTH-R13** the endpoint requires the ``agent`` scope and is request-rate-limited
  in the ``agent`` class (LIMIT-R2; the per-minute ceiling is the config-loaded
  entitlement request-rate bound, not a code literal) keyed by the server-derived athlete id.
- **API-R11a / API-R12a** the response is a status-discriminated union on ``status``; OSS
  surfaces ``completed``, ``degraded``, AND ``awaiting_approval`` (an approval-gated multi-day
  PLAN paused at the durable interrupt-gate, carrying the ``interrupt_id`` the decision endpoint
  consumes + the grounded plan body, CKPT-R9). Only ``budget_exceeded`` remains commercial-only
  and is never produced by the OSS engine.
- **API-R12a** ``POST /v1/agent/threads/{thread_id}/decision`` resumes a paused plan with the
  athlete's ``approve``/``reject``/``edit`` verdict over the DURABLE saver (no recompute,
  CKPT-R2). The atomic single-consume guarantees exactly one decision wins: an unknown/foreign
  interrupt is ``404`` ``not-found``, an already-decided one is ``409`` ``decision-conflict``
  (CKPT-R9); an ``edit`` re-grounds the edited plan before resume (GROUND-R3).
- **API-R11c** the athlete-facing response carries NO billing/budget/model machinery
  (no ``usage``/``cost_*``/token counts/``model_tier``/``reasoning``/model name).
- **API-R12** a run that cannot ground fails closed with ``422`` ``agent-grounding-failed``
  — never a ``completed`` answer with ``grounding.grounded == false``.
- **API-R13 / SCHEMA-R7** ``answer_html`` is server-side sanitized before return.
- **API-R22** ``stream:true`` returns an SSE ``text/event-stream`` of typed events
  (``token``/``progress``/``tool``/``status``/``error``/``done``); a terminal
  ``done`` (or ``error``) event is ALWAYS emitted; an aborted stream is cancellation-safe
  (PERF-R10(b)) — a client disconnect neither spins the loop nor leaks the run.

The identity/scope/engine dependencies are override seams the app factory wires
(FastAPI ``dependency_overrides``), mirroring the performance router. No field is
source-shaped or carries a provider name (AUTH-R15).

Requirement IDs: API-R11, API-R11a, API-R11c, API-R12, API-R13, API-R22, AUTH-R3,
AUTH-R13, SCHEMA-R7, LIMIT-R2, PERF-R10(b), ERR-R8, ERR-R9.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from wattwise_core.agent.deliverables import AgentAnswer, Plan, Readiness
from wattwise_core.agent.engine import DecisionRefused
from wattwise_core.api.agent_stream import (
    SSE_HEADERS,
    SSE_TERMINAL_DONE,
    SSE_TERMINAL_ERROR,
    heartbeat_until,
    problem_event,
    sse_event,
)
from wattwise_core.api.errors import ProblemError, resolve_trace_id
from wattwise_core.api.ratelimit import LimitClass, RateLimiter
from wattwise_core.api.routers.agent_request import (
    header_locale,
    resolve_locale,
    resolve_response_length,
    validate_request,
)
from wattwise_core.api.routers.agent_schemas import (
    AgentAskRequest,
    AgentAskResponse,
    AgentDecisionRequest,
    AgentDecisionResponse,
    DecisionKind,
    ReadinessResponse,
    ResponseLength,
    grounded_flag,
    render_decision,
    render_readiness,
    render_response,
)
from wattwise_core.api.security import attached_entitlement
from wattwise_core.entitlement import Entitlements

router: APIRouter = APIRouter(prefix="/v1/agent", tags=["agent"])


# --- engine seam (injected; reached only through this Protocol, ARCH-R21) --------


@runtime_checkable
class AgentEngine(Protocol):
    """The grounded-answer seam this router drives (the a6 deliverables projection).

    The concrete engine wires the LangGraph coach + fail-closed grounding behind the
    :func:`wattwise_core.agent.deliverables.answer_question` projection. This router
    reaches it ONLY through this typed seam (ARCH-R21): it never imports the in-flight
    graph. ``athlete_id`` is passed server-derived (AUTH-R3) and never trusted from the
    model. The returned :class:`AgentAnswer` already carries the engine's grounded body,
    stable-id observations, surviving citations, and (for ``degraded``) the typed caveat.
    """

    async def answer(
        self,
        *,
        athlete_id: str,
        question: str | None,
        thread_id: str | None,
        response_length: ResponseLength,
        follow_up: dict[str, Any] | None,
        locale: str,
        entitlement: Entitlements | None = None,
    ) -> AgentAnswer: ...

    async def readiness(
        self,
        *,
        athlete_id: str,
        locale: str,
        response_length: ResponseLength,
    ) -> Readiness: ...

    async def decision(
        self,
        *,
        athlete_id: str,
        thread_id: str,
        interrupt_id: str,
        decision: DecisionKind,
        edited_plan: str | None,
        entitlement: Entitlements | None = None,
    ) -> Plan:
        """Resume a paused approval-gated PLAN with the athlete's HITL verdict (API-R12a/CKPT-R9).

        Atomically CONSUMES the live interrupt then drives ``Command(resume)`` through the durable
        saver (no recompute); ``edit`` re-grounds ``edited_plan`` first (GROUND-R3). A
        double-decision / unknown / cross-athlete attempt consumes no row and raises
        :class:`~wattwise_core.agent.engine.DecisionRefused` — the router classifies it 404/409 via
        :meth:`interrupt_status`. ``athlete_id`` is server-derived (AUTH-R3); the ``thread_id`` is
        the path-bound durable scope.
        """
        ...

    async def interrupt_status(
        self, *, athlete_id: str, thread_id: str, interrupt_id: str
    ) -> Literal["unknown", "live", "consumed"]:
        """Classify a refused decision into ``unknown`` (404) vs ``consumed`` (409) (API-R12a).

        A read-only, side-effect-free probe the router consults ONLY after :meth:`decision` fails
        closed, to choose the HTTP status: ``unknown`` (no live/consumed row the caller owns —
        unknown thread/interrupt OR another athlete's, never disclosed) -> ``404``; ``consumed``
        (the caller's row was already decided) -> ``409``. ``live`` is the never-decided race case
        (also ``409`` — a concurrent decision won). Scoped to the server-derived ``athlete_id``
        (CKPT-R3): a foreign row reads as ``unknown``.
        """
        ...


# --- dependency seams (overridden by the app factory) ----------------------------


def require_agent_scope() -> None:
    """Gate the endpoint on the ``agent`` scope (AUTH-R13); app factory overrides it.

    The unwired default fails closed with ``403 insufficient-scope`` so a router
    mounted without its security wiring never serves the agent ungated.
    """
    raise HTTPException(  # pragma: no cover - replaced by the app factory
        status_code=status.HTTP_403_FORBIDDEN, detail="insufficient-scope"
    )


def current_athlete_id() -> str:
    """Server-derived acting athlete id (AUTH-R3); the app factory overrides it.

    Never read from the client. The unwired default fails closed with ``401`` so the
    agent identity is never silently absent.
    """
    raise HTTPException(  # pragma: no cover - replaced by the app factory
        status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthenticated"
    )


def agent_engine() -> AgentEngine:
    """Provide the request-scoped :class:`AgentEngine`; the app factory overrides it."""
    raise HTTPException(  # pragma: no cover - replaced by the app factory
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="internal-error"
    )


def rate_limiter() -> RateLimiter:
    """Provide the process-wide :class:`RateLimiter`; the app factory overrides it."""
    raise HTTPException(  # pragma: no cover - replaced by the app factory
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="internal-error"
    )


_Agent = Depends(require_agent_scope)
AthleteId = Annotated[str, Depends(current_athlete_id)]
Engine = Annotated[AgentEngine, Depends(agent_engine)]
Limiter = Annotated[RateLimiter, Depends(rate_limiter)]


async def _run_engine(
    engine: AgentEngine,
    athlete_id: str,
    body: AgentAskRequest,
    locale: str,
    *,
    entitlement: Entitlements | None = None,
) -> AgentAnswer:
    """Drive the injected engine for ``body`` and enforce fail-closed grounding (API-R12).

    Passes the server-derived ``athlete_id`` (AUTH-R3) — never a client value — and the
    resolved ``locale`` (API-R37) and ``response_length`` (API-R11f). The per-request resolved
    ``entitlement`` (MED-2) is threaded so the engine reads its bounds FROM the attached plan. A
    terminal outcome that is not grounded raises ``422`` ``agent-grounding-failed`` (API-R12 /
    ERR-R9): the API never returns a ``completed`` answer with ``grounding.grounded == false``.
    """
    answer = await engine.answer(
        athlete_id=athlete_id,
        question=body.question,
        thread_id=body.thread_id,
        response_length=resolve_response_length(body),
        follow_up=body.follow_up.model_dump() if body.follow_up else None,
        locale=locale,
        entitlement=entitlement,
    )
    if not grounded_flag(answer):
        raise ProblemError("agent-grounding-failed")
    return answer


# --- SSE streaming (API-R22 / API-R22a) — framing lives in api.agent_stream -------


async def _stream_answer(
    request: Request,
    engine: AgentEngine,
    athlete_id: str,
    body: AgentAskRequest,
    trace_id: str,
    locale: str,
    last_event_id: str | None,
) -> AsyncIterator[str]:
    """Yield the SSE event sequence for one agent run (API-R22/R22a), terminal-safe.

    Emits a ``status`` start frame, interleaves periodic ``:``-comment heartbeats while
    awaiting the engine (~15s, so idle connections survive proxies, API-R22a), then the
    terminal ``done`` (grounded) or ``error`` (grounding failed / engine error) frame —
    a terminal frame is ALWAYS emitted so a client deterministically detects stream end
    (API-R22). On reconnect with a ``Last-Event-ID`` already at the terminal ``done``, a
    ``restart`` first event tells the client the prior run is gone and a fresh one began
    (API-R22a resume). Cancellation-safe per PERF-R10(b): a client disconnect mid-run
    cancels the awaited engine coroutine cleanly — no busy-loop, no foreign cancellation
    into a shared tool session, no leaked run.
    """
    if last_event_id == SSE_TERMINAL_DONE:
        yield sse_event("restart", {"status": "restarting"}, event_id="restart")
    yield sse_event("status", {"status": "working"}, event_id="0")
    entitlement = attached_entitlement(request)
    try:
        if await request.is_disconnected():
            return
        run = asyncio.ensure_future(
            _run_engine(engine, athlete_id, body, locale, entitlement=entitlement)
        )
        async for frame in heartbeat_until(run, request):
            yield frame
        answer = run.result()
    except ProblemError as exc:
        yield sse_event(SSE_TERMINAL_ERROR, problem_event(exc, request), event_id="error")
        return
    response = render_response(answer, trace_id, locale)
    yield sse_event(SSE_TERMINAL_DONE, response.model_dump(), event_id="done")


# --- the endpoint ----------------------------------------------------------------


@router.post(
    "/ask",
    response_model=AgentAskResponse,
    dependencies=[_Agent],
    operation_id="agentAsk",
)
async def agent_ask(
    request: Request,
    body: AgentAskRequest,
    engine: Engine,
    athlete_id: AthleteId,
    limiter: Limiter,
    accept_language: Annotated[str | None, Header()] = None,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> Any:
    """Submit a question to the coaching agent (API-R11); JSON or SSE per ``stream``.

    Requires the ``agent`` scope (AUTH-R13) and debits the per-athlete ``agent`` rate
    bucket (``20/min``, LIMIT-R2) keyed by the server-derived id (AUTH-R3). The 200 body
    is the named status-discriminated :class:`AgentAskResponse` (SCHEMA-R1/API-R11a) with
    a server-sanitized ``answer_html`` (API-R13) and no billing/model machinery
    (API-R11c); a run that cannot ground fails closed ``422`` ``agent-grounding-failed``
    (API-R12). The response language resolves body ``language`` -> ``Accept-Language`` ->
    ``en`` (API-R37). With ``stream:true`` it returns a cancellation-safe SSE stream
    (heartbeats + ``Last-Event-ID`` resume, API-R22a) whose terminal ``done`` carries the
    identical union (API-R22 / PERF-R10(b)).
    """
    limiter.check(athlete_id, LimitClass.AGENT)
    validate_request(body)
    trace_id = resolve_trace_id(request)
    locale = resolve_locale(body, accept_language)
    if body.stream:
        return StreamingResponse(
            _stream_answer(request, engine, athlete_id, body, trace_id, locale, last_event_id),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )
    answer = await _run_engine(
        engine, athlete_id, body, locale, entitlement=attached_entitlement(request)
    )
    return render_response(answer, trace_id, locale)


@router.get(
    "/readiness",
    response_model=ReadinessResponse,
    dependencies=[_Agent],
    operation_id="agentReadiness",
)
async def agent_readiness(
    request: Request,
    engine: Engine,
    athlete_id: AthleteId,
    limiter: Limiter,
    accept_language: Annotated[str | None, Header()] = None,
) -> ReadinessResponse:
    """Read the athlete's readiness/form state (API-R41); a typed verdict, never a number.

    Requires the ``agent`` scope (AUTH-R13) and debits the per-athlete ``agent`` rate
    bucket (LIMIT-R2) keyed by the server-derived id (AUTH-R3). The 200 body is the typed
    :class:`ReadinessResponse`: a verdict ``go|maintain|ease|rest`` (or ``null`` when there
    is insufficient grounded data, GROUND-R6) with NO numeric readiness KPI (API-R41 /
    COACH-R7), a state-first ``summary_text``, a server-sanitized ``summary_html``
    (API-R13), and the form/HRV numbers demoted to on-demand grounded ``citations``
    (GROUND-R5/R7). When no LLM is configured the engine returns the same typed shape with
    a ``null`` verdict and a graceful "not switched on" state sentence (RUN-R4.1) — the
    endpoint never errors on an unconfigured agent. The verdict the API renders is ALWAYS
    the deterministic, metric-consistent one (COACH-R3 / EVAL-R5).
    """
    limiter.check(athlete_id, LimitClass.AGENT)
    trace_id = resolve_trace_id(request)
    locale = header_locale(accept_language)
    readiness = await engine.readiness(
        athlete_id=athlete_id, locale=locale, response_length="standard"
    )
    return render_readiness(readiness, trace_id)


async def _decision_problem(
    engine: AgentEngine, athlete_id: str, thread_id: str, interrupt_id: str
) -> ProblemError:
    """Classify a refused decision into the right RFC 9457 problem (API-R12a / CKPT-R9).

    Consulted ONLY after :meth:`AgentEngine.decision` failed closed (the atomic consume matched no
    live row). The read-only :meth:`interrupt_status` probe disambiguates: ``unknown`` (no row the
    caller owns — unknown thread/interrupt OR a foreign athlete's, never disclosed) -> ``404``
    ``not-found``; ``consumed``/``live`` (the caller's row was already decided, or a concurrent
    decision won the race) -> ``409`` ``decision-conflict``. Identity is the server-derived
    ``athlete_id`` (AUTH-R3 / CKPT-R3). If a deployed engine predates the probe, we fail closed to
    ``409`` (the contract maps a ``rowcount==0`` refusal to already-decided) rather than guessing a
    ``404`` — never resumed twice either way.
    """
    probe = getattr(engine, "interrupt_status", None)
    if probe is None:
        return ProblemError("decision-conflict")
    state = await probe(athlete_id=athlete_id, thread_id=thread_id, interrupt_id=interrupt_id)
    if state == "unknown":
        return ProblemError("not-found")
    return ProblemError("decision-conflict")


@router.post(
    "/threads/{thread_id}/decision",
    response_model=AgentDecisionResponse,
    dependencies=[_Agent],
    operation_id="agentDecision",
)
async def agent_decision(
    request: Request,
    thread_id: str,
    body: AgentDecisionRequest,
    engine: Engine,
    athlete_id: AthleteId,
    limiter: Limiter,
) -> AgentDecisionResponse:
    """Decide a paused approval-gated PLAN: approve / reject / edit (API-R12a / CKPT-R9).

    Requires the ``agent`` scope (AUTH-R13) and debits the per-athlete ``agent`` rate bucket
    (LIMIT-R2) keyed by the server-derived id (AUTH-R3). The ``(athlete_id, conversation_id)``
    durable scope is resolved by the engine FROM the ``thread_id`` path param — identity is never
    a body field (SCHEMA-R4). The engine atomically consumes the live interrupt then resumes the
    durable thread (no recompute, CKPT-R2); ``edit`` re-grounds ``edited_plan`` first (GROUND-R3).
    A refused decision fails closed: an unknown/foreign interrupt is ``404``, an already-decided
    one is ``409`` (CKPT-R9) — the run is NEVER resumed twice.
    """
    limiter.check(athlete_id, LimitClass.AGENT)
    trace_id = resolve_trace_id(request)
    try:
        plan = await engine.decision(
            athlete_id=athlete_id,
            thread_id=thread_id,
            interrupt_id=body.interrupt_id,
            decision=body.decision,
            edited_plan=body.edited_plan,
            entitlement=attached_entitlement(request),
        )
    except DecisionRefused as exc:
        raise await _decision_problem(
            engine, athlete_id, thread_id, body.interrupt_id
        ) from exc
    return render_decision(plan, body.decision, trace_id)


# The remaining doc-60 §6/§7 agent surfaces (diagnose / digest-subscription / memory) live in the
# focused :mod:`agent_breadth` sibling (QUAL-R9 size split); it is mounted onto THIS router so the
# app factory's single ``include_router(agent_routes.router)`` picks up every agent endpoint. The
# import sits at module END (after every seam above is defined) so the breadth module can import
# these seams without an import cycle; ``router`` is fully typed (the ``has-type`` is the cycle, not
# a real ambiguity). ``current_session`` is re-exported so the factory wires it on this module.
from wattwise_core.api.routers import agent_breadth as _breadth  # noqa: E402

router.include_router(_breadth.router)  # type: ignore[has-type]
current_session = _breadth.current_session

__all__ = [
    "AgentAskRequest",
    "AgentAskResponse",
    "AgentDecisionRequest",
    "AgentDecisionResponse",
    "AgentEngine",
    "ReadinessResponse",
    "agent_engine",
    "attached_entitlement",
    "current_athlete_id",
    "current_session",
    "rate_limiter",
    "require_agent_scope",
    "router",
]
