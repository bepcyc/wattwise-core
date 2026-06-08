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
  in the ``agent`` class (``20/min``, LIMIT-R2) keyed by the server-derived athlete id.
- **API-R11a** the response is a status-discriminated union on ``status``; OSS surfaces
  the ``completed`` and ``degraded`` members (``awaiting_approval``/``budget_exceeded``
  are later/commercial and never produced by the OSS engine).
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
from collections.abc import AsyncIterator, Sequence
from typing import Annotated, Any, Final, Literal, Protocol, runtime_checkable

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from wattwise_core.agent.contracts import RunStatus
from wattwise_core.agent.deliverables import AgentAnswer, Citation, Readiness
from wattwise_core.api.agent_stream import (
    SSE_HEADERS,
    SSE_TERMINAL_DONE,
    SSE_TERMINAL_ERROR,
    heartbeat_until,
    problem_event,
    sse_event,
)
from wattwise_core.api.errors import FieldError, ProblemError, resolve_trace_id
from wattwise_core.api.ratelimit import LimitClass, RateLimiter
from wattwise_core.api.routers.agent_schemas import (
    AgentAskRequest,
    AgentAskResponse,
    CitationOut,
    DegradedOut,
    GroundingOut,
    ObservationOut,
    ReadinessResponse,
    ResponseLength,
    SuggestedFollowupOut,
)
from wattwise_core.api.sanitize import sanitize_html

router = APIRouter(prefix="/v1/agent", tags=["agent"])

#: The per-language warm reason_text for a degraded outcome (API-R11a / API-R37). The
#: structured ``coverage_caveat`` carries the machine basis; this is its human gloss in
#: the athlete's selected language (en/de/ru), externalized as catalog copy (QUAL-R13).
_DEGRADED_REASON_BY_LOCALE: Final[dict[str, str]] = {
    "en": "I built this with what we have — a source is offline.",
    "de": "Ich habe das mit den vorhandenen Daten erstellt — eine Quelle ist offline.",
    "ru": "Я собрал это из того, что есть — один источник недоступен.",
}


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
    ) -> AgentAnswer: ...

    async def readiness(
        self,
        *,
        athlete_id: str,
        locale: str,
        response_length: ResponseLength,
    ) -> Readiness: ...


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


# --- projection helpers ----------------------------------------------------------


def _grounded_flag(answer: AgentAnswer) -> bool:
    """True iff the engine produced a grounded terminal outcome (API-R12).

    A ``completed`` or ``degraded`` outcome is grounded (degraded is partial-coverage
    grounded, never fabricated). Any other/absent status is treated as ungrounded so
    the endpoint fails closed rather than emitting an ungrounded answer.
    """
    return answer.status in (RunStatus.COMPLETED, RunStatus.DEGRADED)


def _citations_out(citations: Sequence[Citation]) -> list[CitationOut]:
    """Project surviving grounded citations into the wire shape (API-R11d).

    Shared by the answer and readiness renders — both project the same canonical
    ``{metric, value, as_of}`` + record-id citation shape, so neither carries an
    external provider name (API-R13 / AUTH-R15).
    """
    return [
        CitationOut(citation_id=cit.record_id, metric=cit.metric, value=cit.value, as_of=cit.as_of)
        for cit in citations
    ]


def _observations_out(answer: AgentAnswer) -> list[ObservationOut]:
    """Project the stable-id observations into the wire shape (API-R11e)."""
    return [
        ObservationOut(observation_id=obs.observation_id, text=obs.text)
        for obs in answer.observations
    ]


def _followups_out(answer: AgentAnswer) -> list[SuggestedFollowupOut]:
    """Project the engine's jargon-free follow-up prompts into reveal-numbers chips.

    The deliverables seam carries follow-ups as plain athlete-native labels; we surface
    each as an ``expand`` chip (the safe default "tell me more"), since OSS does not
    bind a label to a specific target. Empty when the engine offered none.
    """
    return [
        SuggestedFollowupOut(kind="expand", label=label)
        for label in answer.suggested_followups
    ]


def _degraded_out(answer: AgentAnswer, locale: str) -> DegradedOut | None:
    """Build the ``degraded`` member payload, else ``None`` (API-R11a / API-R37).

    Present only for a ``degraded`` outcome: the human ``reason_text`` in the athlete's
    selected language (en/de/ru, API-R37) plus the typed ``coverage_caveat``
    (source-agnostic missing/substituted/stale state). The caveat is the engine's typed
    structure; we pass it through without inventing a number.
    """
    if answer.status is not RunStatus.DEGRADED:
        return None
    caveat = dict(answer.coverage_caveat) if answer.coverage_caveat is not None else None
    reason = _DEGRADED_REASON_BY_LOCALE.get(locale, _DEGRADED_REASON_BY_LOCALE["en"])
    return DegradedOut(reason_text=reason, coverage_caveat=caveat)


def _render_response(answer: AgentAnswer, trace_id: str, locale: str) -> AgentAskResponse:
    """Render a grounded :class:`AgentAnswer` into the sanitized response union.

    ``answer_html`` is sanitized HERE (API-R13 / SCHEMA-R7) before it leaves the API —
    the client is never trusted to sanitize. Maps the OSS terminal status to the
    union's closed member; only ``completed``/``degraded`` are reachable in OSS. The
    degraded human caveat is localized to ``locale`` (API-R37).
    """
    member: Literal["completed", "degraded"] = (
        "degraded" if answer.status is RunStatus.DEGRADED else "completed"
    )
    return AgentAskResponse(
        status=member,
        thread_id=answer.thread_id,
        trace_id=trace_id,
        answer_html=sanitize_html(answer.answer_html),
        answer_text=answer.answer_text,
        observations=_observations_out(answer),
        grounding=GroundingOut(grounded=True, citations=_citations_out(answer.citations)),
        suggested_followups=_followups_out(answer),
        degraded=_degraded_out(answer, locale),
    )


def _readiness_followups_out(readiness: Readiness) -> list[SuggestedFollowupOut]:
    """Project the jargon-free reveal-the-numbers chips (API-R11e / VOICE-R9)."""
    return [
        SuggestedFollowupOut(kind="reveal_numbers", label=label)
        for label in readiness.suggested_followups
    ]


def _render_readiness(readiness: Readiness, trace_id: str) -> ReadinessResponse:
    """Render the readiness deliverable into the sanitized typed response (API-R41).

    ``summary_html`` is sanitized HERE (API-R13 / SCHEMA-R7) before it leaves the API. The
    verdict is the StrEnum value (or ``None`` when the deliverable abstained); there is no
    numeric readiness field. ``coverage`` passes through the engine's typed map unchanged
    (no invented number).
    """
    coverage = dict(readiness.coverage) if readiness.coverage is not None else None
    return ReadinessResponse(
        verdict=readiness.verdict.value if readiness.verdict is not None else None,
        as_of=readiness.as_of,
        trace_id=trace_id,
        summary_html=sanitize_html(readiness.summary_html),
        summary_text=readiness.summary_text,
        observations=[
            ObservationOut(observation_id=obs.observation_id, text=obs.text)
            for obs in readiness.observations
        ],
        citations=_citations_out(readiness.citations),
        coverage=coverage,
        suggested_followups=_readiness_followups_out(readiness),
    )


def _validate_request(body: AgentAskRequest) -> None:
    """Enforce the API-R11/R11e body invariants beyond pydantic types.

    ``question`` is REQUIRED unless a ``follow_up`` is present (API-R11e); a request
    with neither is a semantic ``422`` ``validation-error`` (ERR-R6), not a model call.
    The human copy comes from the catalog title (API-R21); the machine-readable cause
    is the ``errors[]`` code clients branch on (ERR-R3), not an inline sentence.
    """
    if body.question is None and body.follow_up is None:
        raise ProblemError(
            "validation-error",
            errors=[FieldError(code="question_required", message="", pointer="/question")],
        )


#: The languages this surface localizes athlete-facing copy into (API-R37).
_SUPPORTED_LOCALES: Final[frozenset[str]] = frozenset({"en", "de", "ru"})


def _header_locale(accept_language: str | None) -> str:
    """The first supported ``Accept-Language`` tag (en/de/ru), else the default ``en``.

    The single header-scan both locale resolvers share (API-R37): it reads the first
    supported two-letter language tag from the comma-separated header, ignoring quality
    weights, and falls back to ``en`` (the commercial layer inserts the persisted
    per-athlete language between the header and this default).
    """
    if accept_language:
        for part in accept_language.split(","):
            tag = part.split(";", 1)[0].strip().lower()[:2]
            if tag in _SUPPORTED_LOCALES:
                return tag
    return "en"


def resolve_locale(body: AgentAskRequest, accept_language: str | None) -> str:
    """Resolve the response language: body ``language`` -> Accept-Language -> ``en`` (API-R37).

    The body ``language`` field takes precedence over the ``Accept-Language`` header when
    both are present; otherwise the first supported language tag in the header is used,
    falling back to the default ``en``. (OSS has no persisted per-athlete language
    setting; the commercial layer inserts it between the header and the default.)
    """
    if body.language is not None:
        return body.language
    return _header_locale(accept_language)


def _resolve_response_length(body: AgentAskRequest) -> ResponseLength:
    """Apply the persisted response-length default when omitted, else ``standard`` (API-R11f).

    OSS has no persisted per-athlete response-length store, so an omitted value resolves
    to ``standard``; the commercial layer resolves the athlete's saved preference here.
    """
    return body.response_length or "standard"


async def _run_engine(
    engine: AgentEngine, athlete_id: str, body: AgentAskRequest, locale: str
) -> AgentAnswer:
    """Drive the injected engine for ``body`` and enforce fail-closed grounding (API-R12).

    Passes the server-derived ``athlete_id`` (AUTH-R3) — never a client value — and the
    resolved ``locale`` (API-R37) and ``response_length`` (API-R11f). A terminal outcome
    that is not grounded raises ``422`` ``agent-grounding-failed`` (API-R12 / ERR-R9):
    the API never returns a ``completed`` answer with ``grounding.grounded == false``.
    """
    answer = await engine.answer(
        athlete_id=athlete_id,
        question=body.question,
        thread_id=body.thread_id,
        response_length=_resolve_response_length(body),
        follow_up=body.follow_up.model_dump() if body.follow_up else None,
        locale=locale,
    )
    if not _grounded_flag(answer):
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
    try:
        if await request.is_disconnected():
            return
        run = asyncio.ensure_future(_run_engine(engine, athlete_id, body, locale))
        async for frame in heartbeat_until(run, request):
            yield frame
        answer = run.result()
    except ProblemError as exc:
        yield sse_event(SSE_TERMINAL_ERROR, problem_event(exc, request), event_id="error")
        return
    response = _render_response(answer, trace_id, locale)
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
    _validate_request(body)
    trace_id = resolve_trace_id(request)
    locale = resolve_locale(body, accept_language)
    if body.stream:
        return StreamingResponse(
            _stream_answer(request, engine, athlete_id, body, trace_id, locale, last_event_id),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )
    answer = await _run_engine(engine, athlete_id, body, locale)
    return _render_response(answer, trace_id, locale)


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
    locale = _header_locale(accept_language)
    readiness = await engine.readiness(
        athlete_id=athlete_id, locale=locale, response_length="standard"
    )
    return _render_readiness(readiness, trace_id)


__all__ = [
    "AgentAskRequest",
    "AgentAskResponse",
    "AgentEngine",
    "ReadinessResponse",
    "agent_engine",
    "current_athlete_id",
    "rate_limiter",
    "require_agent_scope",
    "router",
]
