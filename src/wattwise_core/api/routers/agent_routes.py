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

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any, Final, Literal, Protocol, runtime_checkable

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from wattwise_core.agent.contracts import RunStatus
from wattwise_core.agent.deliverables import AgentAnswer
from wattwise_core.api.errors import FieldError, ProblemError, resolve_trace_id
from wattwise_core.api.ratelimit import LimitClass, RateLimiter
from wattwise_core.api.sanitize import sanitize_html

router = APIRouter(prefix="/v1/agent", tags=["agent"])

#: SSE event names (the API-R22 closed ``sse_event`` discriminator, SCHEMA-R3). The
#: terminal members are ``done``/``error``; one of them is ALWAYS emitted last.
SSE_TERMINAL_DONE: Final = "done"
SSE_TERMINAL_ERROR: Final = "error"

#: Externalized athlete-facing copy for this surface (QUAL-R13: user copy is a named
#: catalog value, never an inline literal in logic). The warm human translation of a
#: degraded outcome's typed coverage caveat (API-R11a / API-R21); the structured
#: ``coverage_caveat`` carries the machine-readable basis a client badges on.
_DEGRADED_REASON_TEXT: Final = "I built this with what we have — a source is offline."

#: The athlete-facing answer-length enum (API-R11f); default ``standard``.
ResponseLength = Literal["short", "standard", "detailed"]
FollowUpKind = Literal["expand", "drill", "reveal_numbers"]


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


# --- wire shapes -----------------------------------------------------------------


class FollowUp(BaseModel):
    """Typed conversational follow-up over the durable thread (API-R11e)."""

    kind: FollowUpKind
    target_ref: str | None = None


class AgentAskRequest(BaseModel):
    """``POST /v1/agent/ask`` request body (API-R11).

    ``question`` is required UNLESS a ``follow_up`` is present (API-R11e); it is bounded
    to 2000 chars (LIMIT-R5). Identity is NOT a field here — it is server-derived
    (AUTH-R3); a client cannot name the athlete it acts as.
    """

    question: str | None = Field(default=None, min_length=1, max_length=2000)
    thread_id: str | None = None
    response_length: ResponseLength = "standard"
    follow_up: FollowUp | None = None
    language: Literal["en", "de", "ru"] | None = None
    stream: bool = False


class CitationOut(BaseModel):
    """One on-demand grounded citation (API-R11d): ``{metric, value, as_of}`` only.

    References a canonical metric value/date; NEVER an external provider name
    (API-R13 / AUTH-R15). ``source_kind`` is the fixed ``canonical`` marker.
    """

    citation_id: str
    metric: str | None = None
    value: float | None = None
    as_of: str | None = None
    source_kind: Literal["canonical"] = "canonical"


class ObservationOut(BaseModel):
    """One athlete-facing observation with its stable expand/drill handle (API-R11e)."""

    observation_id: str
    text: str


class GroundingOut(BaseModel):
    """The grounding block: the grounded flag + on-demand citations (API-R11d)."""

    grounded: bool
    citations: list[CitationOut]


class SuggestedFollowupOut(BaseModel):
    """An optional athlete-native drill-down chip (API-R11e); jargon-free copy."""

    kind: FollowUpKind
    label: str
    target_ref: str | None = None


class DegradedOut(BaseModel):
    """The ``degraded`` member payload: human caveat + typed coverage caveat (API-R11a)."""

    reason_text: str
    coverage_caveat: dict[str, Any] | None = None


class AgentAskResponse(BaseModel):
    """The status-discriminated ``AgentAskResponse`` union (API-R11a).

    OSS surfaces ``completed`` and ``degraded`` (the engine never produces
    ``awaiting_approval``/``budget_exceeded`` in OSS). Carries NO billing/budget/model
    machinery (API-R11c): there is deliberately no ``usage``/``cost_*``/token/
    ``model_tier``/``reasoning`` field on this schema. ``answer_html`` is already
    sanitized (API-R13).
    """

    status: Literal["completed", "degraded"]
    thread_id: str
    trace_id: str
    answer_html: str
    answer_text: str
    observations: list[ObservationOut]
    grounding: GroundingOut
    suggested_followups: list[SuggestedFollowupOut] = Field(default_factory=list)
    degraded: DegradedOut | None = None


# --- projection helpers ----------------------------------------------------------


def _grounded_flag(answer: AgentAnswer) -> bool:
    """True iff the engine produced a grounded terminal outcome (API-R12).

    A ``completed`` or ``degraded`` outcome is grounded (degraded is partial-coverage
    grounded, never fabricated). Any other/absent status is treated as ungrounded so
    the endpoint fails closed rather than emitting an ungrounded answer.
    """
    return answer.status in (RunStatus.COMPLETED, RunStatus.DEGRADED)


def _citations_out(answer: AgentAnswer) -> list[CitationOut]:
    """Project the surviving grounded citations into the wire shape (API-R11d)."""
    return [
        CitationOut(
            citation_id=cit.record_id,
            metric=cit.metric,
            value=cit.value,
            as_of=cit.as_of,
        )
        for cit in answer.citations
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


def _degraded_out(answer: AgentAnswer) -> DegradedOut | None:
    """Build the ``degraded`` member payload, else ``None`` (API-R11a).

    Present only for a ``degraded`` outcome: the human ``reason_text`` plus the typed
    ``coverage_caveat`` (source-agnostic missing/substituted/stale state). The caveat is
    the engine's typed structure; we pass it through without inventing a number.
    """
    if answer.status is not RunStatus.DEGRADED:
        return None
    caveat = dict(answer.coverage_caveat) if answer.coverage_caveat is not None else None
    return DegradedOut(reason_text=_DEGRADED_REASON_TEXT, coverage_caveat=caveat)


def _render_response(answer: AgentAnswer, trace_id: str) -> AgentAskResponse:
    """Render a grounded :class:`AgentAnswer` into the sanitized response union.

    ``answer_html`` is sanitized HERE (API-R13 / SCHEMA-R7) before it leaves the API —
    the client is never trusted to sanitize. Maps the OSS terminal status to the
    union's closed member; only ``completed``/``degraded`` are reachable in OSS.
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
        grounding=GroundingOut(grounded=True, citations=_citations_out(answer)),
        suggested_followups=_followups_out(answer),
        degraded=_degraded_out(answer),
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


async def _run_engine(engine: AgentEngine, athlete_id: str, body: AgentAskRequest) -> AgentAnswer:
    """Drive the injected engine for ``body`` and enforce fail-closed grounding (API-R12).

    Passes the server-derived ``athlete_id`` (AUTH-R3) — never a client value. A
    terminal outcome that is not grounded raises ``422`` ``agent-grounding-failed``
    (API-R12 / ERR-R9): the API never returns a ``completed`` answer with
    ``grounding.grounded == false``.
    """
    answer = await engine.answer(
        athlete_id=athlete_id,
        question=body.question,
        thread_id=body.thread_id,
        response_length=body.response_length,
        follow_up=body.follow_up.model_dump() if body.follow_up else None,
        locale=body.language or "en",
    )
    if not _grounded_flag(answer):
        raise ProblemError("agent-grounding-failed")
    return answer


# --- SSE streaming (API-R22) -----------------------------------------------------


def _sse_event(event: str, data: dict[str, Any], *, event_id: str | None = None) -> str:
    """Encode one SSE frame: the typed ``event:`` name + a JSON ``data:`` line (API-R22).

    The ``event`` is the closed ``sse_event`` discriminator carried on the wire
    ``event:`` line (the client dispatches on it). An optional ``id:`` makes the event
    resumable (API-R22a). The frame ends with the blank line SSE requires.
    """
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, separators=(',', ':'))}")
    return "\n".join(lines) + "\n\n"


def _terminal_envelope(response: AgentAskResponse) -> dict[str, Any]:
    """The streamed ``done`` envelope: the SAME athlete-facing union as non-streamed.

    Carries the identical status-discriminated payload (API-R11a/R17) — including the
    stable observation/citation handles — and NO billing/model machinery (API-R11c).
    """
    return response.model_dump()


async def _stream_answer(
    request: Request,
    engine: AgentEngine,
    athlete_id: str,
    body: AgentAskRequest,
    trace_id: str,
) -> AsyncIterator[str]:
    """Yield the SSE event sequence for one agent run (API-R22), terminal-safe.

    Emits a ``status`` start frame, then the terminal ``done`` (grounded) or ``error``
    (grounding failed / engine error) frame — a terminal frame is ALWAYS emitted so a
    client deterministically detects stream end (API-R22). Cancellation-safe per
    PERF-R10(b): a client disconnect mid-run raises ``CancelledError`` here, which
    propagates to cancel the awaited engine coroutine cleanly — no busy-loop, no foreign
    cancellation injected into a shared tool session, no leaked run.
    """
    yield _sse_event("status", {"status": "working"}, event_id="0")
    try:
        if await request.is_disconnected():
            return
        answer = await _run_engine(engine, athlete_id, body)
    except ProblemError as exc:
        yield _sse_event(
            SSE_TERMINAL_ERROR,
            _problem_event(exc, request),
            event_id="error",
        )
        return
    response = _render_response(answer, trace_id)
    yield _sse_event(SSE_TERMINAL_DONE, _terminal_envelope(response), event_id="done")


def _problem_event(exc: ProblemError, request: Request) -> dict[str, Any]:
    """Render a :class:`ProblemError` as the streamed ``error`` body (API-R16).

    The streamed ``error`` carries the SAME RFC 9457 problem body as the non-streamed
    contract (§6 / API-R16); the stream never terminates silently on failure.
    """
    return {
        "type": exc.problem_type.uri,
        "title": exc.problem_type.title,
        "status": exc.problem_type.status,
        "detail": exc.detail if exc.detail is not None else exc.problem_type.title,
        "instance": request.url.path,
        "trace_id": resolve_trace_id(request),
    }


#: Anti-buffering + keep-alive headers so SSE works through proxies/CDNs (API-R22a).
_SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# --- the endpoint ----------------------------------------------------------------


@router.post("/ask", dependencies=[_Agent], operation_id="agentAsk")
async def agent_ask(
    request: Request,
    body: AgentAskRequest,
    engine: Engine,
    athlete_id: AthleteId,
    limiter: Limiter,
) -> Any:
    """Submit a question to the coaching agent (API-R11); JSON or SSE per ``stream``.

    Requires the ``agent`` scope (AUTH-R13) and debits the per-athlete ``agent`` rate
    bucket (``20/min``, LIMIT-R2) keyed by the server-derived id (AUTH-R3). Returns the
    status-discriminated :class:`AgentAskResponse` (API-R11a) with a server-sanitized
    ``answer_html`` (API-R13) and no billing/model machinery (API-R11c); a run that
    cannot ground fails closed ``422`` ``agent-grounding-failed`` (API-R12). With
    ``stream:true`` it returns a cancellation-safe SSE stream of typed events whose
    terminal ``done`` carries the identical union (API-R22 / PERF-R10(b)).
    """
    limiter.check(athlete_id, LimitClass.AGENT)
    _validate_request(body)
    trace_id = resolve_trace_id(request)
    if body.stream:
        return StreamingResponse(
            _stream_answer(request, engine, athlete_id, body, trace_id),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    answer = await _run_engine(engine, athlete_id, body)
    return _render_response(answer, trace_id)


__all__ = [
    "AgentAskRequest",
    "AgentAskResponse",
    "AgentEngine",
    "agent_engine",
    "current_athlete_id",
    "rate_limiter",
    "require_agent_scope",
    "router",
]
