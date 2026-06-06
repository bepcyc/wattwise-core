"""ASGI middleware: server-side JSON body-size cap (LIMIT-R5/R6).

A JSON request body is capped at a 256 KiB default; an oversized body yields the
catalog ``413 payload-too-large`` problem. The cap is enforced from the STREAMED byte
count as the body arrives (never by trusting a client-sent ``Content-Length``, LIMIT-R6),
so a lying length header cannot bypass it. Multipart uploads are exempt here — their
larger cap is enforced mid-stream by the imports router — so only ``application/json``
(and empty) bodies are bounded by this middleware.

Requirement IDs: LIMIT-R5 (256 KiB JSON body cap -> ``413``), LIMIT-R6 (server-side,
from the streamed bytes, not a client header), ERR-R1 (uniform problem document).
"""

from __future__ import annotations

from typing import Final

from fastapi import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from wattwise_core.api.errors import PROBLEM_MEDIA_TYPE, render_problem_bytes

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


__all__ = ["DEFAULT_JSON_MAX_BYTES", "JSONBodySizeLimitMiddleware"]
