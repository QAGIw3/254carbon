"""Reusable ASGI middleware components."""

from __future__ import annotations

import time
from typing import Callable, Coroutine

from starlette.requests import Request
from starlette.responses import Response


class RequestTimingMiddleware:
    """Collect request latency metrics via a callback."""

    def __init__(self, app: Callable, callback: Callable[[float, Request, Response], None]) -> None:
        self.app = app
        self.callback = callback

    async def __call__(self, scope, receive, send) -> None:  # pragma: no cover - ASGI runtime
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        start = time.monotonic()
        response_container = {}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                response_container["status_code"] = message.get("status")
            await send(message)

        await self.app(scope, receive, send_wrapper)
        duration = time.monotonic() - start
        response = Response(status_code=response_container.get("status_code", 500))
        self.callback(duration, request, response)


__all__ = ["RequestTimingMiddleware"]

