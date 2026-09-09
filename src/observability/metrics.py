"""In-memory metrics (FR-27).

Deliberately simple: a single process, in-memory counters, reset on
restart. No Prometheus client, no external metrics backend — nothing in
the approved architecture calls for one at portfolio scale. Exposed via
GET /metrics.
"""

import threading
from dataclasses import dataclass, field


@dataclass
class _MetricsState:
    request_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_request(self) -> None:
        with self._lock:
            self.request_count += 1

    def record_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "request_count": self.request_count,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
            }


metrics_state = _MetricsState()
