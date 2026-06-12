"""Integration tests for the secure agent API surface (A-E3, doc 60 §7).

Drives ``POST /v1/agent/ask`` end-to-end over the assembled ASGI app with a FAKE
:class:`AgentEngine` injected through the router's override seams, asserting the
boundary contract the router owns:

- **SCHEMA-R7 / API-R13** ``answer_html`` is server-side sanitized — an injection
  payload comes back inert (no ``<script>``, no event handler, no ``javascript:``).
- **LIMIT-R2 / LIMIT-R3** the ``agent`` class is rate-limited to ``20/min`` per
  athlete; the 21st call in a window is ``429`` ``rate-limited`` with ``Retry-After``
  and ``RateLimit-*`` headers.
- **API-R22** ``stream:true`` returns an SSE ``text/event-stream`` that ALWAYS emits a
  terminal ``done`` (or ``error``) event.
- **API-R11c** no athlete-facing response (JSON or streamed) carries a billing/model/
  token machinery field.
- **API-R12** a run that cannot ground fails closed ``422`` ``agent-grounding-failed``.

The engine is a stand-in for the a6 deliverables projection (ARCH-R21): the router is
the unit under test, not the grounding engine.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from starlette.requests import Request
from starlette.testclient import TestClient

from wattwise_core.agent.contracts import RunStatus
from wattwise_core.agent.deliverables import AgentAnswer, Citation, Observation, Plan, Readiness
from wattwise_core.agent.engine import DecisionRefused
from wattwise_core.api.agent_stream import problem_event
from wattwise_core.api.app import create_app
from wattwise_core.api.auth import Scope, issue_access_token
from wattwise_core.api.errors import FieldError, ProblemError
from wattwise_core.api.ratelimit import LimitClass, RateLimiter
from wattwise_core.api.redaction import contains_pii, redact_payload, redact_text
from wattwise_core.api.routers import agent_routes, agent_schemas
from wattwise_core.api.sanitize import is_inert, sanitize_html
from wattwise_core.config import Environment, load_settings
from wattwise_core.domain.enums import ReadinessVerdict


def _fake_request(path: str = "/v1/agent/ask") -> Request:
    """A minimal Starlette request for unit-testing the SSE problem renderer."""
    scope = {"type": "http", "path": path, "headers": [], "query_string": b""}
    return Request(scope)


pytestmark = pytest.mark.integration

#: The forbidden billing/budget/model machinery fields (API-R11c) — none may appear
#: on any athlete-facing agent response, streamed or not.
FORBIDDEN_FIELDS = (
    "usage",
    "cost_remaining_usd",
    "cost_reset_at",
    "cost_usd_estimate",
    "input_tokens",
    "output_tokens",
    "model_tier",
    "reasoning",
    "model",
)


class _FakeEngine:
    """A controllable stand-in for the grounded-answer engine (ARCH-R21 seam).

    Returns a preset :class:`AgentAnswer` so the router's boundary behavior — not the
    grounding — is what is exercised. ``athlete_id`` is recorded to assert the router
    passes the server-derived id, never a client value.
    """

    def __init__(self, answer: AgentAnswer, readiness: Readiness | None = None) -> None:
        self._answer = answer
        self._readiness = readiness
        self.seen_athlete_id: str | None = None
        self.seen_entitlement: Any = None

    async def answer(
        self,
        *,
        athlete_id: str,
        question: str | None,
        thread_id: str | None,
        response_length: str,
        follow_up: dict[str, Any] | None,
        locale: str,
        entitlement: Any = None,  # MED-2: the route threads the per-request resolved plan
        coach_numeric_detail_level: int | None = None,
    ) -> AgentAnswer:
        self.seen_athlete_id = athlete_id
        self.seen_entitlement = entitlement
        self.seen_locale = locale
        return self._answer

    async def readiness(self, *, athlete_id: str, locale: str, response_length: str) -> Readiness:
        self.seen_athlete_id = athlete_id
        assert self._readiness is not None, "test did not script a readiness deliverable"
        return self._readiness


def _grounded_answer(
    *,
    answer_html: str = "<p>You're fresh and ready.</p>",
    status: RunStatus = RunStatus.COMPLETED,
) -> AgentAnswer:
    """A grounded :class:`AgentAnswer` with a stable observation + citation."""
    return AgentAnswer(
        status=status,
        thread_id="01THREAD",
        answer_html=answer_html,
        answer_text="You're fresh and ready.",
        observations=(Observation(observation_id="01OBS", text="You're recovered."),),
        citations=(Citation(record_id="01CIT", metric="tsb", value=6.2, as_of="2026-06-05"),),
        suggested_followups=("Tell me more",),
    )


def _readiness(
    *,
    verdict: ReadinessVerdict | None = ReadinessVerdict.REST,
    summary_text: str = "You're deep in fatigue, so today is for rest.",
    coverage: dict[str, Any] | None = None,
    citations: tuple[Citation, ...] = (),
) -> Readiness:
    """A typed readiness deliverable: a state-first verdict with no numeric readiness field."""
    return Readiness(
        verdict=verdict,
        status=RunStatus.COMPLETED,
        as_of="2026-06-08",
        summary_html=f"<p>{summary_text}</p>",
        summary_text=summary_text,
        citations=citations,
        coverage=coverage or {"inputs_used": ["form"], "inputs_unavailable": ["hrv"]},
        suggested_followups=("Show me the numbers behind that",) if citations else (),
    )


def _build_app(
    answer: AgentAnswer,
    *,
    limiter: RateLimiter | None = None,
    readiness: Readiness | None = None,
) -> tuple[FastAPI, _FakeEngine]:
    """Assemble the app with the agent router mounted and its seams overridden.

    Wires the fake engine, a fixed server-derived athlete id, an always-pass ``agent``
    scope gate, and a rate limiter (a fresh one unless the test supplies its own).
    """
    settings = load_settings(
        app__environment=Environment.DEVELOPMENT,
        database_dsn="sqlite+aiosqlite:///:memory:",
        token_signing_key="test-signing-key-0123456789abcdef",
    )
    app = create_app(settings)
    app.include_router(agent_routes.router)
    engine = _FakeEngine(answer, readiness)
    bucket = limiter or RateLimiter()
    app.dependency_overrides[agent_routes.require_agent_scope] = lambda: None
    app.dependency_overrides[agent_routes.current_athlete_id] = lambda: "owner"
    app.dependency_overrides[agent_routes.agent_engine] = lambda: engine
    app.dependency_overrides[agent_routes.rate_limiter] = lambda: bucket
    return app, engine


def _token(app: FastAPI) -> str:
    """Mint a valid bearer token for the single owner (the scope gate is overridden)."""
    settings = app.state.settings
    tokens = issue_access_token(settings, subject="owner", scopes=(Scope.AGENT,))
    return tokens.access_token


def _auth(app: FastAPI) -> dict[str, str]:
    return {"Authorization": f"Bearer {_token(app)}"}


def test_answer_html_is_sanitized_inert() -> None:
    """An injection payload in the engine's HTML returns inert (SCHEMA-R7 / API-R13)."""
    payload = "<p>Hi</p><script>alert(1)</script><img src=x onerror=alert(2)>"
    app, _ = _build_app(_grounded_answer(answer_html=payload))
    with TestClient(app) as client:
        resp = client.post("/v1/agent/ask", json={"question": "How am I?"}, headers=_auth(app))
    assert resp.status_code == 200
    html = resp.json()["answer_html"]
    lowered = html.lower()
    assert "<script" not in lowered
    assert "onerror" not in lowered
    assert "javascript:" not in lowered
    assert "alert" not in lowered or "<" not in html  # no executable markup survives
    assert "<p>hi</p>" in lowered


def test_completed_response_has_no_forbidden_fields() -> None:
    """No billing/model/token machinery on the athlete-facing response (API-R11c)."""
    app, _ = _build_app(_grounded_answer())
    with TestClient(app) as client:
        resp = client.post("/v1/agent/ask", json={"question": "Ready?"}, headers=_auth(app))
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "completed"
    assert body["grounding"]["grounded"] is True
    flat = json.dumps(body)
    for field in FORBIDDEN_FIELDS:
        assert f'"{field}"' not in flat, f"forbidden field {field!r} leaked (API-R11c)"


def test_grounding_failure_is_fail_closed_422() -> None:
    """An ungrounded terminal outcome -> 422 agent-grounding-failed (API-R12)."""
    app, _ = _build_app(_grounded_answer(status=RunStatus.AWAITING_APPROVAL))
    with TestClient(app) as client:
        resp = client.post("/v1/agent/ask", json={"question": "Ready?"}, headers=_auth(app))
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/agent-grounding-failed")


def test_degraded_member_carries_typed_caveat() -> None:
    """A degraded outcome surfaces the typed coverage caveat (API-R11a)."""
    answer = AgentAnswer(
        status=RunStatus.DEGRADED,
        thread_id="01T",
        answer_html="<p>Working with what we have.</p>",
        answer_text="Working with what we have.",
        coverage_caveat={"inputs": [{"input": "hrv", "state": "missing"}]},
    )
    app, _ = _build_app(answer)
    with TestClient(app) as client:
        resp = client.post("/v1/agent/ask", json={"question": "Ready?"}, headers=_auth(app))
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "degraded"
    assert body["degraded"]["coverage_caveat"]["inputs"][0]["input"] == "hrv"


def test_rate_limit_returns_429_with_headers() -> None:
    """The 21st agent call in a window is 429 with Retry-After + RateLimit-* (LIMIT-R2/R3)."""
    limiter = RateLimiter()
    app, _ = _build_app(_grounded_answer(), limiter=limiter)
    with TestClient(app) as client:
        headers = _auth(app)
        last = None
        for _ in range(21):
            last = client.post("/v1/agent/ask", json={"question": "Ready?"}, headers=headers)
        assert last is not None
        assert last.status_code == 429
        assert last.json()["type"].endswith("/rate-limited")
        assert "Retry-After" in last.headers
        assert int(last.headers["Retry-After"]) >= 1
        assert last.headers["RateLimit-Limit"] == "20"
        assert last.headers["RateLimit-Remaining"] == "0"
        assert "RateLimit-Reset" in last.headers


def test_stream_emits_terminal_done_event() -> None:
    """stream:true yields an SSE stream that ends with a terminal done event (API-R22)."""
    app, _ = _build_app(_grounded_answer())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/ask",
            json={"question": "Ready?", "stream": True},
            headers=_auth(app),
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
    events = _sse_events(body)
    assert events[-1] == "done", f"stream must end with a terminal done (API-R22); got {events}"
    assert "status" in events
    # the done envelope carries the same union and no forbidden machinery (API-R11c/R17)
    done_data = _last_data_block(body)
    assert done_data["status"] == "completed"
    assert done_data["grounding"]["grounded"] is True
    flat = json.dumps(done_data)
    for field in FORBIDDEN_FIELDS:
        assert f'"{field}"' not in flat


def test_stream_grounding_failure_emits_terminal_error() -> None:
    """An ungrounded stream emits a terminal error event, never silent end (API-R16/R22)."""
    app, _ = _build_app(_grounded_answer(status=RunStatus.AWAITING_APPROVAL))
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/ask",
            json={"question": "Ready?", "stream": True},
            headers=_auth(app),
        )
        body = resp.text
    events = _sse_events(body)
    assert events[-1] == "error"
    err = _last_data_block(body)
    assert err["type"].endswith("/agent-grounding-failed")


def test_missing_question_without_follow_up_is_422() -> None:
    """A request with neither question nor follow_up is 422 validation-error (API-R11e)."""
    app, _ = _build_app(_grounded_answer())
    with TestClient(app) as client:
        resp = client.post("/v1/agent/ask", json={}, headers=_auth(app))
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/validation-error")


def test_stream_emits_id_lines_for_resume() -> None:
    """SSE frames carry ``id:`` lines so a client can resume with Last-Event-ID (API-R22a)."""
    app, _ = _build_app(_grounded_answer())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/ask", json={"question": "Ready?", "stream": True}, headers=_auth(app)
        )
        body = resp.text
    ids = [line.split(":", 1)[1].strip() for line in body.splitlines() if line.startswith("id:")]
    assert "0" in ids and "done" in ids  # the status frame and the terminal frame are addressable


def test_stream_reconnect_at_done_emits_restart_first() -> None:
    """A reconnect already at the terminal ``done`` gets a ``restart`` first event (API-R22a)."""
    app, _ = _build_app(_grounded_answer())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/ask",
            json={"question": "Ready?", "stream": True},
            headers={**_auth(app), "Last-Event-ID": "done"},
        )
        body = resp.text
    events = _sse_events(body)
    assert events[0] == "restart"
    assert events[-1] == "done"


def test_degraded_reason_is_localized_by_accept_language() -> None:
    """A degraded answer's reason_text is in the selected language (API-R11a / API-R37)."""
    answer = AgentAnswer(
        status=RunStatus.DEGRADED,
        thread_id="01T",
        answer_html="<p>x</p>",
        answer_text="x",
        coverage_caveat={"inputs": []},
    )
    app, _ = _build_app(answer)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/ask",
            json={"question": "Wie geht's?"},
            headers={**_auth(app), "Accept-Language": "de-DE,de;q=0.9"},
        )
    reason = resp.json()["degraded"]["reason_text"]
    assert "vorhandenen Daten" in reason  # the German localization, not the English constant


def test_region_tag_localizes_catalog_by_primary_subtag_and_directive_keeps_full_tag() -> None:
    """A region/full BCP-47 header drives BOTH roles from one resolved tag (API-R37 / LANG-R1).

    The model-free degraded catalog resolves by the PRIMARY language subtag (``de-DE`` -> ``de``
    German copy, never the English floor), while the directive the engine receives keeps the FULL
    tag (``de-de``) so the coach answers in the exact variant. Both halves pinned; mutation-prove
    by removing :func:`catalog_locale`'s primary-subtag reduction (a bare ``.get`` on the full tag)
    and the German half falls back to English here.
    """
    answer = AgentAnswer(
        status=RunStatus.DEGRADED,
        thread_id="01T",
        answer_html="<p>x</p>",
        answer_text="x",
        coverage_caveat={"inputs": []},
    )
    app, engine = _build_app(answer)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/ask",
            json={"question": "Wie geht's?"},
            headers={**_auth(app), "Accept-Language": "de-DE"},
        )
    # model-free catalog half: German copy via the primary subtag, not the English floor
    assert "vorhandenen Daten" in resp.json()["degraded"]["reason_text"]
    # directive half: the engine received the FULL region tag (header-normalized to lowercase),
    # NOT a primary-subtag-truncated "de" — the region subtag survives to the directive
    assert engine.seen_locale == "de-de"


def test_body_language_overrides_accept_language() -> None:
    """The body ``language`` field takes precedence over Accept-Language (API-R37)."""
    answer = AgentAnswer(
        status=RunStatus.DEGRADED,
        thread_id="01T",
        answer_html="<p>x</p>",
        answer_text="x",
        coverage_caveat={"inputs": []},
    )
    app, _ = _build_app(answer)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/ask",
            json={"question": "?", "language": "ru"},
            headers={**_auth(app), "Accept-Language": "de"},
        )
    assert "источник" in resp.json()["degraded"]["reason_text"]  # ru wins over the de header


def test_streamed_error_includes_field_errors() -> None:
    """The streamed RFC 9457 ``error`` body carries ``errors[]`` like the sync path (API-R16)."""
    exc = ProblemError(
        "validation-error",
        errors=[FieldError(code="question_required", message="", pointer="/question")],
    )
    body = problem_event(exc, _fake_request())
    assert body["type"].endswith("/validation-error")
    assert body["errors"][0]["code"] == "question_required"
    assert body["errors"][0]["pointer"] == "/question"


def test_identity_is_server_derived_not_client_supplied() -> None:
    """A forged caller-identity body field is rejected before the engine (AUTH-R3 / SCHEMA-R4).

    ``AgentAskRequest`` is ``additionalProperties:false``, so a client-supplied
    ``athlete_id`` is a ``422`` validation error — it never reaches the engine and can
    never widen the acting subject (the engine is keyed only on the server-derived id).
    """
    app, engine = _build_app(_grounded_answer())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/ask",
            json={"question": "Ready?", "athlete_id": "attacker"},
            headers=_auth(app),
        )
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/validation-error")
    assert engine.seen_athlete_id is None  # the forged field never reached the engine


def test_server_derived_identity_passed_to_engine() -> None:
    """A clean request passes the server-derived athlete id to the engine (AUTH-R3)."""
    app, engine = _build_app(_grounded_answer())
    with TestClient(app) as client:
        client.post("/v1/agent/ask", json={"question": "Ready?"}, headers=_auth(app))
    assert engine.seen_athlete_id == "owner"


# --- GET /v1/agent/readiness (API-R41) -------------------------------------------

#: The forbidden numeric-readiness KPI/score fields — none may appear on the readiness
#: response (API-R41 / COACH-R7: readiness is a typed verdict, never a number).
FORBIDDEN_READINESS_FIELDS = ("readiness", "readiness_score", "score")


def test_readiness_returns_typed_verdict_state_first_no_numeric_kpi() -> None:
    """The readiness endpoint returns a typed verdict + state-first summary, no number (API-R41)."""
    app, _ = _build_app(
        _grounded_answer(),
        readiness=_readiness(
            citations=(
                Citation(
                    record_id="form@2026-06-08", metric="form", value=-21.4, as_of="2026-06-08"
                ),
            ),
        ),
    )
    with TestClient(app) as client:
        resp = client.get("/v1/agent/readiness", headers=_auth(app))
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "rest"  # a typed verdict, not a number
    assert body["summary_text"]
    assert not any(ch.isdigit() for ch in _first_sentence(body["summary_text"]))  # state-first
    # the form number is on-demand backing only: a grounded citation, never a hero KPI
    assert body["citations"][0]["metric"] == "form"
    flat = json.dumps(body)
    for field in FORBIDDEN_READINESS_FIELDS:
        assert f'"{field}"' not in flat, f"numeric readiness field {field!r} leaked (API-R41)"
    for field in FORBIDDEN_FIELDS:
        assert f'"{field}"' not in flat  # no billing/model machinery either (API-R11c)


def test_readiness_html_is_sanitized_inert() -> None:
    """The readiness summary_html is server-side sanitized before return (API-R13 / SCHEMA-R7)."""
    payload = "You're set.<script>alert(1)</script>"
    app, _ = _build_app(
        _grounded_answer(),
        readiness=_readiness(verdict=ReadinessVerdict.GO, summary_text=payload),
    )
    with TestClient(app) as client:
        resp = client.get("/v1/agent/readiness", headers=_auth(app))
    assert resp.status_code == 200
    assert "<script" not in resp.json()["summary_html"].lower()


def test_readiness_graceful_when_unconfigured_returns_null_verdict() -> None:
    """An unconfigured agent yields a graceful readiness: null verdict + sentence (RUN-R4.1)."""
    app, _ = _build_app(
        _grounded_answer(),
        readiness=_readiness(
            verdict=None,
            summary_text="Coaching isn't switched on for this account yet.",
            coverage={"reason": "agent_unconfigured"},
        ),
    )
    with TestClient(app) as client:
        resp = client.get("/v1/agent/readiness", headers=_auth(app))
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] is None  # typed graceful response, not an error
    assert body["summary_text"]


def test_readiness_passes_server_derived_identity_to_engine() -> None:
    """The readiness route passes the server-derived athlete id to the engine (AUTH-R3)."""
    app, engine = _build_app(_grounded_answer(), readiness=_readiness())
    with TestClient(app) as client:
        client.get("/v1/agent/readiness", headers=_auth(app))
    assert engine.seen_athlete_id == "owner"


def _first_sentence(text: str) -> str:
    """The leading sentence of a plain-text summary (for the state-first digit check)."""
    for end in (". ", "! ", "? "):
        idx = text.find(end)
        if idx != -1:
            return text[: idx + 1]
    return text


# --- POST /v1/agent/threads/{thread_id}/decision (API-R12a / CKPT-R9) ------------


class _FakeDecisionEngine:
    """A controllable stand-in for the HITL decision engine (ARCH-R21 seam, API-R12a).

    Models the real atomic single-consume: the live interrupt is consumed exactly ONCE; a
    second decision (or an unknown/foreign interrupt) raises :class:`DecisionRefused`, mirroring
    ``consume_interrupt`` returning ``False``. ``interrupt_status`` is the read-only probe the
    router consults to split 404 (unknown) from 409 (consumed). The seen ``decision``/
    ``edited_plan``/``athlete_id`` are recorded so the test asserts the router passes the
    server-derived id (AUTH-R3) and the edit body through (GROUND-R3).
    """

    def __init__(self, *, known: set[str] | None = None) -> None:
        self._live: set[str] = set(known) if known is not None else {"01INT"}
        self._consumed: set[str] = set()
        self.seen_athlete_id: str | None = None
        self.seen_decision: str | None = None
        self.seen_edited_plan: str | None = None
        self.seen_thread_id: str | None = None

    async def decision(
        self,
        *,
        athlete_id: str,
        thread_id: str,
        interrupt_id: str,
        decision: str,
        edited_plan: str | None,
        entitlement: Any = None,  # MED-2: the route threads the per-request resolved plan
    ) -> Plan:
        self.seen_athlete_id = athlete_id
        self.seen_decision = decision
        self.seen_edited_plan = edited_plan
        self.seen_thread_id = thread_id
        if interrupt_id not in self._live:  # atomic consume matched no live row -> refused
            raise DecisionRefused(f"no live interrupt {interrupt_id!r}")
        self._live.discard(interrupt_id)
        self._consumed.add(interrupt_id)
        body = f"<p>Day 1: {edited_plan or 'Endurance ride'}.</p>"
        return Plan(
            status=RunStatus.COMPLETED,
            thread_id=thread_id,
            plan_html=body,
            plan_text=f"Day 1: {edited_plan or 'Endurance ride'}.",
            observations=(Observation(observation_id="01OBS", text="Build aerobic base."),),
            citations=(Citation(record_id="01CIT", metric="ctl", value=42.0, as_of="2026-06-07"),),
            suggested_followups=("Make it harder",),
        )

    async def interrupt_status(self, *, athlete_id: str, thread_id: str, interrupt_id: str) -> str:
        if interrupt_id in self._consumed:
            return "consumed"
        if interrupt_id in self._live:
            return "live"
        return "unknown"


def _build_decision_app(engine: _FakeDecisionEngine) -> FastAPI:
    """Assemble the app with the decision engine + overridden seams (mirrors _build_app)."""
    settings = load_settings(
        app__environment=Environment.DEVELOPMENT,
        database_dsn="sqlite+aiosqlite:///:memory:",
        token_signing_key="test-signing-key-0123456789abcdef",
    )
    app = create_app(settings)
    app.include_router(agent_routes.router)
    app.dependency_overrides[agent_routes.require_agent_scope] = lambda: None
    app.dependency_overrides[agent_routes.current_athlete_id] = lambda: "owner"
    app.dependency_overrides[agent_routes.agent_engine] = lambda: engine
    bucket = RateLimiter()
    app.dependency_overrides[agent_routes.rate_limiter] = lambda: bucket
    return app


def test_decision_approve_finalizes_plan_and_passes_server_identity() -> None:
    """F-APPROVE: approve resumes the durable plan; identity is server-derived (API-R12a)."""
    engine = _FakeDecisionEngine()
    app = _build_decision_app(engine)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/threads/owner:01CONV/decision",
            json={"interrupt_id": "01INT", "decision": "approve"},
            headers=_auth(app),
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["decision"] == "approve"
    assert body["grounding"]["grounded"] is True
    assert body["plan_text"].startswith("Day 1")
    assert engine.seen_athlete_id == "owner"  # AUTH-R3: never a client value
    assert engine.seen_thread_id == "owner:01CONV"  # the path param is the durable scope
    flat = json.dumps(body)
    for field in FORBIDDEN_FIELDS:
        assert f'"{field}"' not in flat  # no billing/model machinery (API-R11c)


def test_decision_edit_passes_edited_plan_through_for_regrounding() -> None:
    """F-EDIT: an edit forwards edited_plan to the engine (which re-grounds it, GROUND-R3)."""
    engine = _FakeDecisionEngine()
    app = _build_decision_app(engine)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/threads/owner:01CONV/decision",
            json={"interrupt_id": "01INT", "decision": "edit", "edited_plan": "Recovery spin 45m"},
            headers=_auth(app),
        )
    assert resp.status_code == 200
    assert resp.json()["decision"] == "edit"
    assert engine.seen_edited_plan == "Recovery spin 45m"
    assert "Recovery spin 45m" in resp.json()["plan_text"]


def test_decision_reject_resumes_without_approval() -> None:
    """F-REJECT: reject still resumes the durable thread to a terminal outcome (API-R12a)."""
    engine = _FakeDecisionEngine()
    app = _build_decision_app(engine)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/threads/owner:01CONV/decision",
            json={"interrupt_id": "01INT", "decision": "reject"},
            headers=_auth(app),
        )
    assert resp.status_code == 200
    assert resp.json()["decision"] == "reject"
    assert engine.seen_decision == "reject"


def test_decision_edit_without_body_is_422_validation_error() -> None:
    """An edit with no edited_plan is a 422 cross-field validation error (API-R12a / SCHEMA-R4)."""
    engine = _FakeDecisionEngine()
    app = _build_decision_app(engine)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/threads/owner:01CONV/decision",
            json={"interrupt_id": "01INT", "decision": "edit"},
            headers=_auth(app),
        )
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/validation-error")
    assert engine.seen_decision is None  # never reached the engine


def test_decision_edited_plan_on_non_edit_is_422() -> None:
    """A non-edit decision carrying an edited_plan is a 422 (API-R12a cross-field rule)."""
    app = _build_decision_app(_FakeDecisionEngine())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/threads/owner:01CONV/decision",
            json={"interrupt_id": "01INT", "decision": "approve", "edited_plan": "sneaky"},
            headers=_auth(app),
        )
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/validation-error")


def test_decision_forged_athlete_id_field_is_422() -> None:
    """A forged body identity field is rejected before the engine (SCHEMA-R4 / AUTH-R3)."""
    engine = _FakeDecisionEngine()
    app = _build_decision_app(engine)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/threads/owner:01CONV/decision",
            json={"interrupt_id": "01INT", "decision": "approve", "athlete_id": "attacker"},
            headers=_auth(app),
        )
    assert resp.status_code == 422
    assert engine.seen_athlete_id is None  # the forged field never reached the engine


def test_decision_double_decision_is_409_decision_conflict() -> None:
    """F-409: a second decision on a consumed interrupt is 409 decision-conflict (CKPT-R9)."""
    engine = _FakeDecisionEngine()
    app = _build_decision_app(engine)
    with TestClient(app) as client:
        first = client.post(
            "/v1/agent/threads/owner:01CONV/decision",
            json={"interrupt_id": "01INT", "decision": "approve"},
            headers=_auth(app),
        )
        assert first.status_code == 200
        second = client.post(
            "/v1/agent/threads/owner:01CONV/decision",
            json={"interrupt_id": "01INT", "decision": "approve"},
            headers=_auth(app),
        )
    assert second.status_code == 409
    assert second.json()["type"].endswith("/decision-conflict")


def test_decision_unknown_interrupt_is_404_not_found() -> None:
    """F-404: a decision against an interrupt that was never recorded is 404 not-found (CKPT-R9)."""
    engine = _FakeDecisionEngine()
    app = _build_decision_app(engine)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/threads/owner:01CONV/decision",
            json={"interrupt_id": "01NOPE", "decision": "approve"},
            headers=_auth(app),
        )
    assert resp.status_code == 404
    assert resp.json()["type"].endswith("/not-found")


def test_decision_html_is_sanitized_inert() -> None:
    """The resumed plan_html is server-side sanitized before return (API-R13 / SCHEMA-R7)."""
    engine = _FakeDecisionEngine()
    app = _build_decision_app(engine)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/threads/owner:01CONV/decision",
            json={
                "interrupt_id": "01INT",
                "decision": "edit",
                "edited_plan": "ride<script>alert(1)</script>",
            },
            headers=_auth(app),
        )
    assert resp.status_code == 200
    html = resp.json()["plan_html"].lower()
    assert "<script" not in html
    assert "alert" not in html or "<" not in resp.json()["plan_html"]


def test_awaiting_approval_render_surfaces_interrupt_id_and_plan() -> None:
    """render_plan_awaiting yields awaiting_approval + interrupt_id + sanitized body (API-R12a)."""
    paused = Plan(
        status=RunStatus.AWAITING_APPROVAL,
        thread_id="owner:01CONV",
        plan_html="<p>Week 1 base.</p><script>x()</script>",
        plan_text="Week 1 base.",
        interrupt_id="01INT",
        citations=(Citation(record_id="01CIT", metric="ctl", value=42.0, as_of="2026-06-07"),),
        suggested_followups=("Make it harder",),
    )
    rendered = agent_schemas.render_plan_awaiting(paused, trace_id="tid-1")
    assert rendered.status == "awaiting_approval"
    assert rendered.interrupt_id == "01INT"
    assert rendered.grounding.grounded is True
    assert rendered.plan_text == "Week 1 base."
    assert rendered.plan_html is not None
    assert "<script" not in rendered.plan_html.lower()  # sanitized HERE (API-R13)
    assert rendered.answer_html == rendered.plan_html  # body mirrored for status-agnostic clients


# --- sanitize.py (SCHEMA-R7) -----------------------------------------------------


def test_sanitize_strips_script_and_handlers_to_inert() -> None:
    """Script/handlers/iframe/js-uri are stripped; the result is inert (SCHEMA-R7)."""
    payload = (
        "<p>ok</p><script>steal()</script>"
        '<a href="javascript:evil()">x</a>'
        "<img src=x onerror=alert(1)>"
        '<iframe src="e"></iframe>'
        '<p style="background:url(javascript:x)">styled</p>'
    )
    cleaned = sanitize_html(payload)
    assert is_inert(cleaned)
    assert "<p>ok</p>" in cleaned
    assert "<script" not in cleaned
    assert "javascript:" not in cleaned
    assert "onerror" not in cleaned
    assert "<iframe" not in cleaned


def test_sanitize_keeps_allowed_formatting() -> None:
    """Allowed coach-narrative formatting survives sanitization (SCHEMA-R7)."""
    cleaned = sanitize_html("<p>Form is <strong>good</strong> — <em>rest up</em>.</p>")
    assert "<strong>good</strong>" in cleaned
    assert "<em>rest up</em>" in cleaned


# --- redaction.py (API-R19 / ERR-R5) ---------------------------------------------


def test_redaction_masks_pii_and_secrets() -> None:
    """Emails / tokens / keys are masked out of free text (API-R19 / ERR-R5)."""
    raw = "Reach me at rider@example.com using sk-abcdef0123456789ABCDEF or Bearer tok123abc456"
    redacted = redact_text(raw)
    assert "rider@example.com" not in redacted
    assert "sk-abcdef0123456789ABCDEF" not in redacted
    assert not contains_pii(redacted)


def test_redaction_masks_secret_keyed_payload_fields() -> None:
    """A secret-named key is masked wholesale in a structured trace payload (API-R19)."""
    payload = {"detail": "rider@example.com asked", "token": "should-be-masked", "count": 3}
    out = redact_payload(payload)
    assert out["token"] == "[redacted]"
    assert "rider@example.com" not in out["detail"]
    assert out["count"] == 3  # non-text scalars pass through unchanged


# --- ratelimit.py (LIMIT-R2) -----------------------------------------------------


def test_ratelimiter_isolates_per_athlete() -> None:
    """One athlete's exhausted bucket does not affect another's (LIMIT-R1, per-athlete)."""
    limiter = RateLimiter()
    for _ in range(20):
        limiter.check("a", LimitClass.AGENT)
    # 'a' is now exhausted; 'b' is untouched and still served.
    headers = limiter.check("b", LimitClass.AGENT)
    assert headers.remaining == 19
    assert headers.limit == 20


def _sse_events(sse_body: str) -> list[str]:
    """Return the ordered ``event:`` names from an SSE body (API-R22 discriminators)."""
    prefix = "event:"
    return [
        line[len(prefix) :].strip() for line in sse_body.splitlines() if line.startswith(prefix)
    ]


def _last_data_block(sse_body: str) -> dict[str, Any]:
    """Return the JSON payload of the last ``data:`` line in an SSE body."""
    prefix = "data:"
    data_lines = [
        line[len(prefix) :].strip() for line in sse_body.splitlines() if line.startswith(prefix)
    ]
    assert data_lines, "SSE body carried no data lines"
    parsed: dict[str, Any] = json.loads(data_lines[-1])
    return parsed


# --- API-R37: the persisted language default on the agent path -------------------


def test_ask_applies_persisted_language_default() -> None:
    """With no body language and no Accept-Language, the STORED subtag is applied (API-R37)."""
    app, engine = _build_app(_grounded_answer())
    app.dependency_overrides[agent_routes.persisted_locale] = lambda: "de"
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/ask",
            json={"question": "How am I doing?"},
            headers={"Authorization": f"Bearer {_token(app)}"},
        )
    assert resp.status_code == 200
    assert engine.seen_locale == "de"  # the persisted default reached the engine


def test_ask_per_request_language_overrides_persisted_default() -> None:
    """A per-request body language wins over the stored default without mutating it (API-R37)."""
    app, engine = _build_app(_grounded_answer())
    app.dependency_overrides[agent_routes.persisted_locale] = lambda: "de"
    with TestClient(app) as client:
        resp = client.post(
            "/v1/agent/ask",
            json={"question": "How am I doing?", "language": "ru"},
            headers={"Authorization": f"Bearer {_token(app)}"},
        )
    assert resp.status_code == 200
    assert engine.seen_locale == "ru"  # the override applies for this one call only
