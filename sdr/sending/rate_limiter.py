"""Token bucket rate limiter for outbound message sending.

Enforces per-channel rate limits to avoid API throttling and
maintain natural sending patterns.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Simple token bucket rate limiter."""

    capacity: float
    refill_rate: float  # tokens per second
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        self._tokens = self.capacity
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    def acquire(self, timeout: float = 30.0) -> bool:
        """Try to acquire a token, blocking up to timeout seconds.

        Returns True if acquired, False if timed out.
        """
        deadline = time.monotonic() + timeout
        with self._lock:
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                # Sleep until enough time for one token, or timeout
                wait = min(1.0 / self.refill_rate, remaining)
                self._lock.release()
                time.sleep(wait)
                self._lock.acquire()

    def try_acquire(self) -> bool:
        """Non-blocking attempt to acquire a token."""
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


class RateLimiter:
    """Per-channel rate limiter for outbound sends."""

    def __init__(self, gmail_per_hour: int = 20, linkedin_per_hour: int = 10):
        self._buckets = {
            "gmail": TokenBucket(
                capacity=float(gmail_per_hour),
                refill_rate=gmail_per_hour / 3600.0,
            ),
            "linkedin": TokenBucket(
                capacity=float(linkedin_per_hour),
                refill_rate=linkedin_per_hour / 3600.0,
            ),
        }

    def acquire(self, channel: str, timeout: float = 30.0) -> bool:
        """Acquire a send token for the given channel."""
        bucket = self._buckets.get(channel.lower())
        if not bucket:
            return True  # Unknown channel — don't block
        return bucket.acquire(timeout)

    def try_acquire(self, channel: str) -> bool:
        """Non-blocking check if we can send on the given channel."""
        bucket = self._buckets.get(channel.lower())
        if not bucket:
            return True
        return bucket.try_acquire()
