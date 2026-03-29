from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeSettings:
    api_key: str | None
    rate_limit_per_minute: int


def load_runtime_settings() -> RuntimeSettings:
    api_key = os.getenv("SMART_PARKING_API_KEY") or None
    raw_limit = os.getenv("SMART_PARKING_RATE_LIMIT_PER_MINUTE", "180")
    try:
        rate_limit_per_minute = max(int(raw_limit), 1)
    except ValueError:
        rate_limit_per_minute = 180
    return RuntimeSettings(api_key=api_key, rate_limit_per_minute=rate_limit_per_minute)


class InMemoryRateLimiter:
    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.lock = threading.Lock()
        self.events: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> dict[str, int | bool]:
        now = time.monotonic()
        with self.lock:
            queue = self.events[key]
            while queue and (now - queue[0]) >= self.window_seconds:
                queue.popleft()

            if len(queue) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - queue[0])))
                return {
                    "allowed": False,
                    "remaining": 0,
                    "retry_after": retry_after,
                }

            queue.append(now)
            remaining = max(self.limit - len(queue), 0)
            return {
                "allowed": True,
                "remaining": remaining,
                "retry_after": 0,
            }
