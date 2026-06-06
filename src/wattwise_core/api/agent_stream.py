"""SSE framing + keepalive for the agent stream (API-R22 / API-R22a / PERF-R10(b)).

The agent endpoint's ``stream:true`` branch emits a ``text/event-stream`` of typed
events (``status``/``restart``/``error``/``done``) with a terminal ``done`` or ``error``
ALWAYS last (API-R22). This module owns the low-level framing — the closed ``sse_event``
discriminator on each ``event:`` line, a resumable ``id:``, periodic ``:``-comment
heartbeats so idle connections survive proxies (API-R22a), and the streamed RFC 9457
``error`` body (the SAME shape as the synchronous problem, INCLUDING ``errors[]``,
API-R16) — so the router file stays within the size ceiling (QUAL-R9).

Requirement IDs: API-R22 (typed SSE events + terminal frame), API-R22a (heartbeat +
resumable id), API-R16 (streamed error == synchronous problem body), ERR-R5 (no leak in
the streamed ``detail``/``errors[].message``).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, Final

from fastapi import Request

from wattwise_core.api.errors import ProblemError, resolve_trace_id
from wattwise_core.api.redaction import redact_text

#: The terminal SSE event members (API-R22): one is ALWAYS emitted last.
SSE_TERMINAL_DONE: Final = "done"
SSE_TERMINAL_ERROR: Final = "error"

#: The SSE keepalive heartbeat interval (~15s) so idle connections are not dropped by
#: intermediaries while a long run produces no token events (API-R22a).
HEARTBEAT_SECONDS: Final = 15.0

#: Anti-buffering + keep-alive headers so SSE works through proxies/CDNs (API-R22a).
SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def sse_event(event: str, data: dict[str, Any], *, event_id: str | None = None) -> str:
    """Encode one SSE frame: the typed ``event:`` name + a JSON ``data:`` line (API-R22).

    The ``event`` is the closed ``sse_event`` discriminator on the wire ``event:`` line
    (the client dispatches on it); an optional ``id:`` makes the event resumable
    (API-R22a). The frame ends with the blank line SSE requires.
    """
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, separators=(',', ':'))}")
    return "\n".join(lines) + "\n\n"


async def heartbeat_until(run: asyncio.Future[Any], request: Request) -> AsyncIterator[str]:
    """Emit ``:``-comment heartbeats until the run completes or the client disconnects.

    Awaits the run with a bounded timeout so a long, token-silent run still flushes a
    keepalive every ~15s (API-R22a). On a detected client disconnect the run is cancelled
    (PERF-R10(b)) and the generator returns without a terminal frame (the connection is
    already gone). A run failure surfaces via ``run.result()`` in the caller.
    """
    while not run.done():
        done, _ = await asyncio.wait({run}, timeout=HEARTBEAT_SECONDS)
        if run in done:
            break
        if await request.is_disconnected():
            run.cancel()
            return
        yield ": keep-alive\n\n"


def problem_event(exc: ProblemError, request: Request) -> dict[str, Any]:
    """Render a :class:`ProblemError` as the streamed ``error`` body (API-R16).

    The streamed ``error`` carries the SAME RFC 9457 problem body as the non-streamed
    contract (§6 / API-R16) — INCLUDING the ``errors[]`` field-level findings when present
    — so a streamed validation/grounding error is as complete as the synchronous one. Free
    text is redacted (ERR-R5) the same way the synchronous problem document is.
    """
    body: dict[str, Any] = {
        "type": exc.problem_type.uri,
        "title": exc.problem_type.title,
        "status": exc.problem_type.status,
        "detail": redact_text(exc.detail if exc.detail is not None else exc.problem_type.title),
        "instance": request.url.path,
        "trace_id": resolve_trace_id(request),
    }
    if exc.errors:
        body["errors"] = [
            {**err.to_dict(), "message": redact_text(err.message)} for err in exc.errors
        ]
    return body


__all__ = [
    "HEARTBEAT_SECONDS",
    "SSE_HEADERS",
    "SSE_TERMINAL_DONE",
    "SSE_TERMINAL_ERROR",
    "heartbeat_until",
    "problem_event",
    "sse_event",
]
