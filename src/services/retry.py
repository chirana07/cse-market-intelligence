from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    attempts: int = 2
    backoff_sec: float = 0.5
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,)


@dataclass(frozen=True)
class RetryResult(Generic[T]):
    value: T | None
    error: BaseException | None
    attempts: int
    elapsed_sec: float

    @property
    def ok(self) -> bool:
        return self.error is None


def run_with_retry(func: Callable[[], T], config: RetryConfig | None = None) -> RetryResult[T]:
    config = config or RetryConfig()
    attempts = max(1, config.attempts)
    started = time.time()
    last_error: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            value = func()
            return RetryResult(
                value=value,
                error=None,
                attempts=attempt,
                elapsed_sec=round(time.time() - started, 3),
            )
        except config.retry_exceptions as exc:
            last_error = exc
            if attempt >= attempts:
                break
            time.sleep(max(0.0, config.backoff_sec) * attempt)

    return RetryResult(
        value=None,
        error=last_error,
        attempts=attempts,
        elapsed_sec=round(time.time() - started, 3),
    )
