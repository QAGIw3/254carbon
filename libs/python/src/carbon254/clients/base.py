"""Base HTTP client with resilience primitives.

This module provides a reusable `BaseHttpClient` built on httpx with:
- Timeouts per request
- Exponential backoff with jitter
- Circuit breaker semantics (half-open after cool-down)
- Idempotency enforcement for retry-safe methods

Services can subclass this client to tailor behaviour while keeping
consistent resilience characteristics.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


logger = logging.getLogger(__name__)


class CircuitOpenError(RuntimeError):
    """Raised when the circuit breaker is open."""


@dataclass
class CircuitBreakerState:
    failure_threshold: int = 5
    recovery_time_seconds: int = 30
    failure_count: int = 0
    opened_at: Optional[float] = None

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.opened_at = time.time()

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened_at = None

    def allowed(self) -> bool:
        if self.opened_at is None:
            return True
        if (time.time() - self.opened_at) >= self.recovery_time_seconds:
            # Half-open state
            return True
        return False


class BaseHttpClient:
    """Shared HTTP client wrapper enforcing resilience best practices."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        idempotent_methods: Optional[set[str]] = None,
    ) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.idempotent_methods = idempotent_methods or {"GET", "HEAD", "OPTIONS"}
        self.circuit_state = CircuitBreakerState()

    def request(
        self,
        method: str,
        url: str,
        **kwargs: Dict[str, Any],
    ) -> httpx.Response:
        if not self.circuit_state.allowed():
            raise CircuitOpenError("Circuit breaker open for remote service")

        attempt = 0
        method_upper = method.upper()
        while True:
            attempt += 1
            try:
                response = self._client.request(method_upper, url, **kwargs)
                response.raise_for_status()
                self.circuit_state.record_success()
                return response
            except httpx.HTTPError as exc:  # pragma: no cover - http errors
                if method_upper not in self.idempotent_methods:
                    raise

                self.circuit_state.record_failure()
                if attempt > self.max_retries:
                    logger.error("Max retries reached for %s %s", method_upper, url)
                    raise

                sleep_time = self._compute_backoff(attempt)
                logger.warning(
                    "Retrying %s %s after error %s (attempt %s, sleep %.2fs)",
                    method_upper,
                    url,
                    exc,
                    attempt,
                    sleep_time,
                )
                time.sleep(sleep_time)

    def close(self) -> None:
        self._client.close()

    def _compute_backoff(self, attempt: int) -> float:
        base = self.backoff_factor * (2 ** (attempt - 1))
        jitter = random.uniform(0, base)
        return base + jitter


__all__ = ["BaseHttpClient", "CircuitOpenError"]

