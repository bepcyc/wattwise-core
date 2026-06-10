"""ASGI middleware: server-side JSON body-size cap (LIMIT-R5/R6) + per-endpoint metrics (OBS-R5).

A JSON request body is capped at a 256 KiB default; an oversized body yields the
catalog ``413 payload-too-large`` problem. The cap is enforced from the STREAMED byte
count as the body arrives (never by trusting a client-sent ``Content-Length``, LIMIT-R6),
so a lying length header cannot bypass it. Multipart uploads are exempt here — their
larger cap is enforced mid-stream by the imports router — so only ``application/json``
(and empty) bodies are bounded by this middleware.

:class:`EndpointMetricsMiddleware` records the OBS-R5 operational metrics — per-endpoint
request rate, latency, and error rate — onto the process metrics registry the ``/metrics``
scrape surface serves, so request rate/latency/error rate per endpoint are monitorable in
production (OBS-R5), not just ghost metric names.

Requirement IDs: LIMIT-R5 (256 KiB JSON body cap -> ``413``), LIMIT-R6 (server-side,
from the streamed bytes, not a client header), ERR-R1 (uniform problem document),
OBS-R5 (per-endpoint request rate/latency/error rate).
"""

from __future__ import annotations

import time
from typing import Final

from fastapi import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from wattwise_core.api.errors import PROBLEM_MEDIA_TYPE, render_problem_bytes
from wattwise_core.observability import metrics as _metrics

#: LIMIT-R5 default JSON body cap (256 KiB). Multipart uploads use the upload cap.
DEFAULT_JSON_MAX_BYTES: Final = 256 * 1024


class JSONBodySizeLimitMiddleware:
    """Reject an oversized JSON request body with ``413`` from the streamed bytes (LIMIT-R5)."""

    def __init__(self, app: ASGIApp, *, max_bytes: int = DEFAULT_JSON_MAX_BYTES) -> None:
        """Wrap ``app``, capping JSON bodies at ``max_bytes`` (default 256 KiB)."""
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Buffer the JSON body under the cap; emit ``413`` once the streamed bytes exceed it."""
        if scope["type"] != "http" or not _is_json(scope):
            await self._app(scope, receive, send)
            return

        buffered: list[Message] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                buffered.append(message)
                break
            total += len(message.get("body", b""))
            if total > self._max_bytes:
                await self._reject(scope, send)
                return
            buffered.append(message)
            if not message.get("more_body", False):
                break

        iterator = iter(buffered)

        async def _replay() -> Message:
            try:
                return next(iterator)
            except StopIteration:
                return await receive()

        await self._app(scope, _replay, send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        """Emit the catalog ``413 payload-too-large`` problem document (ERR-R1)."""
        body = render_problem_bytes("payload-too-large", Request(scope))
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", PROBLEM_MEDIA_TYPE.encode()),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _is_json(scope: Scope) -> bool:
    """True iff the request declares a JSON content type (the only capped class)."""
    for name, value in scope.get("headers", ()):
        if name == b"content-type":
            media = bytes(value).split(b";", 1)[0].strip().lower()
            return media == b"application/json"
    return False


class EndpointMetricsMiddleware:
    """Record per-endpoint request rate, latency, and error rate on every request (OBS-R5).

    OBS-R5 mandates operational metrics including request rate/latency/error rate PER ENDPOINT.
    This pure-ASGI middleware (no ``BaseHTTPMiddleware`` dependency, matching
    :class:`JSONBodySizeLimitMiddleware`) increments
    :data:`~wattwise_core.observability.metrics.ENDPOINT_REQUESTS` (labelled by the matched route
    template + an ``ok``/``error`` outcome) and observes
    :data:`~wattwise_core.observability.metrics.ENDPOINT_LATENCY_SECONDS` (labelled by route) for
    every HTTP request, so those metric names are LIVE on the ``/metrics`` surface rather than
    permanently-zero ghosts.

    The endpoint label is the matched route's PATH TEMPLATE (e.g. ``/v1/activities/{id}``), not the
    raw path, so per-resource ids do not explode label cardinality; an unmatched path (404 before
    routing) labels as ``unmatched``. The outcome is ``error`` when the response status is ``>=500``
    or the inner app raised, else ``ok`` — so the error RATE per endpoint is derivable from the
    labelled counter. Latency is wall-clock from request entry to the response-start event.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap ``app``, recording per-endpoint request/latency/error metrics (OBS-R5)."""
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Time the request and record the OBS-R5 per-endpoint counters/summary on completion."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        started = time.monotonic()
        status_holder = {"status": 500}

        async def _send(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = int(message["status"])
            await send(message)

        try:
            await self._app(scope, receive, _send)
        except BaseException:
            # The inner app raised (no response started): record the request as an error before
            # re-raising so a crashing endpoint's error rate is still observable (OBS-R5).
            self._record(scope, started, outcome="error")
            raise
        outcome = "error" if status_holder["status"] >= 500 else "ok"
        self._record(scope, started, outcome=outcome)

    def _record(self, scope: Scope, started: float, *, outcome: str) -> None:
        """Increment the per-endpoint request counter + observe its latency (OBS-R5)."""
        endpoint = _route_template(scope)
        registry = _metrics.get_registry()
        registry.increment(
            _metrics.ENDPOINT_REQUESTS, labels={"endpoint": endpoint, "outcome": outcome}
        )
        registry.observe(
            _metrics.ENDPOINT_LATENCY_SECONDS,
            time.monotonic() - started,
            labels={"endpoint": endpoint},
        )


def _route_template(scope: Scope) -> str:
    """The matched route's path template (low cardinality), else ``unmatched`` (OBS-R5).

    Starlette binds the matched :class:`~starlette.routing.Route` to ``scope['route']`` once routing
    resolves; its ``path`` is the template (``/v1/activities/{id}``) so per-resource ids do not
    explode the label set. A request that never matched a route (a 404) has no bound route, so it is
    bucketed under ``unmatched`` rather than spraying the raw path across the label space.
    """
    route = scope.get("route")
    template = getattr(route, "path", None)
    if isinstance(template, str) and template:
        return template
    return "unmatched"


__all__ = [
    "DEFAULT_JSON_MAX_BYTES",
    "EndpointMetricsMiddleware",
    "JSONBodySizeLimitMiddleware",
]
