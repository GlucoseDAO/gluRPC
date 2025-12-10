import asyncio
import time
import logging
from typing import Optional, Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from glurpc.schemas import RequestMetrics

logger = logging.getLogger("glurpc.middleware")


class RequestCounterMiddleware(BaseHTTPMiddleware):
    """Count all HTTP requests, errors, and track request times (app.state-backed)."""

    def __init__(self, app, *, clock: Callable[[], float] = time.perf_counter):
        super().__init__(app)
        self._clock = clock
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):
        metrics: Optional[RequestMetrics] = getattr(
            request.app.state, "request_metrics", None
        )
        if metrics is None:
            metrics = RequestMetrics()
            request.app.state.request_metrics = metrics

        metrics.total_http_requests += 1

        start_time = self._clock()
        response = await call_next(request)
        end_time = self._clock()

        duration_ms = (end_time - start_time) * 1000.0
        async with self._lock:
            metrics.request_times.append(duration_ms)

        if response.status_code >= 400:
            metrics.total_http_errors += 1

        return response


class DisconnectMiddleware(BaseHTTPMiddleware):
    """
    Attach a per-request disconnect event to Request.state.
    Uses asyncio.wait_for around request.is_disconnected to avoid hangs.
    """

    def __init__(
        self,
        app,
        *,
        poll_interval: float = 0.05,
        check_timeout: float = 0.1,
    ):
        super().__init__(app)
        self._poll_interval = poll_interval
        self._check_timeout = check_timeout

    async def dispatch(self, request: Request, call_next):
        disconnect_event = asyncio.Event()

        async def watch_disconnect() -> None:
            while True:
                try:
                    await asyncio.sleep(self._poll_interval)
                    if await asyncio.wait_for(
                        request.is_disconnected(), timeout=self._check_timeout
                    ):
                        disconnect_event.set()
                        return
                except asyncio.TimeoutError:
                    # Expected; keep polling
                    continue
                except asyncio.CancelledError:
                    # Exit quietly on cancellation
                    return
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("disconnect watch error: %s", exc, exc_info=True)
                    await asyncio.sleep(self._poll_interval)

        watcher: Optional[asyncio.Task] = asyncio.create_task(
            watch_disconnect(), name="disconnect-watch"
        )

        request.state.disconnect_event = disconnect_event
        request.state.disconnect_watcher = watcher

        try:
            response = await call_next(request)
        finally:
            if watcher:
                watcher.cancel()
        return response
