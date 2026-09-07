from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import time


class RunBudgetExceeded(RuntimeError):
    """Raised when the global solve budget cannot permit more work."""


Clock = Callable[[], float]


@dataclass
class RunBudget:
    """
    Global budget for one complete solve invocation.

    This sits above the Phase 3 RepairBudget.

    Responsibilities:
    - enforce the overall wall-clock deadline
    - retain a safety margin for audit/publish
    - bound candidate attempts
    - bound model calls
    - optionally bound total model tokens
    - provide safe subprocess/model timeouts
    """

    total_wall_seconds: float

    safety_margin_seconds: float = 5.0

    max_candidate_attempts: int = 6
    max_model_calls: int = 6
    max_total_tokens: int | None = None

    candidate_attempts_used: int = 0
    model_calls_used: int = 0
    tokens_used: int = 0

    _clock: Clock = field(
        default=time.monotonic,
        repr=False,
        compare=False,
    )

    _started_at: float = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.total_wall_seconds <= 0:
            raise ValueError(
                "total_wall_seconds must be positive"
            )

        if self.safety_margin_seconds < 0:
            raise ValueError(
                "safety_margin_seconds "
                "must be non-negative"
            )

        if (
            self.safety_margin_seconds
            >= self.total_wall_seconds
        ):
            raise ValueError(
                "safety_margin_seconds must be "
                "smaller than total_wall_seconds"
            )

        if self.max_candidate_attempts < 0:
            raise ValueError(
                "max_candidate_attempts "
                "must be non-negative"
            )

        if self.max_model_calls < 0:
            raise ValueError(
                "max_model_calls must be non-negative"
            )

        if (
            self.max_total_tokens is not None
            and self.max_total_tokens < 0
        ):
            raise ValueError(
                "max_total_tokens must be "
                "non-negative"
            )

        if (
            self.candidate_attempts_used < 0
            or self.model_calls_used < 0
            or self.tokens_used < 0
        ):
            raise ValueError(
                "run budget usage must be "
                "non-negative"
            )

        self._started_at = self._clock()

    @property
    def elapsed_seconds(self) -> float:
        return max(
            0.0,
            self._clock() - self._started_at,
        )

    @property
    def hard_remaining_seconds(self) -> float:
        return max(
            0.0,
            self.total_wall_seconds
            - self.elapsed_seconds,
        )

    @property
    def usable_remaining_seconds(self) -> float:
        """
        Time available for ordinary pipeline work.

        The safety margin is deliberately excluded so
        clean-room audit and publish still have time.
        """

        return max(
            0.0,
            self.hard_remaining_seconds
            - self.safety_margin_seconds,
        )

    @property
    def exhausted(self) -> bool:
        return self.usable_remaining_seconds <= 0

    @property
    def remaining_candidate_attempts(self) -> int:
        return max(
            0,
            self.max_candidate_attempts
            - self.candidate_attempts_used,
        )

    @property
    def remaining_model_calls(self) -> int:
        return max(
            0,
            self.max_model_calls
            - self.model_calls_used,
        )

    @property
    def remaining_tokens(self) -> int | None:
        if self.max_total_tokens is None:
            return None

        return max(
            0,
            self.max_total_tokens
            - self.tokens_used,
        )

    def require_time(
        self,
        *,
        minimum_seconds: float = 0.0,
    ) -> None:
        if minimum_seconds < 0:
            raise ValueError(
                "minimum_seconds must be "
                "non-negative"
            )

        if (
            self.usable_remaining_seconds
            < minimum_seconds
        ):
            raise RunBudgetExceeded(
                "Global solve deadline does not "
                "have enough usable time remaining."
            )

    def bounded_timeout(
        self,
        requested_seconds: float,
        *,
        minimum_seconds: float = 0.1,
    ) -> float:
        """
        Return a timeout that cannot consume the
        reserved final safety margin.
        """

        if requested_seconds <= 0:
            raise ValueError(
                "requested_seconds must be positive"
            )

        if minimum_seconds <= 0:
            raise ValueError(
                "minimum_seconds must be positive"
            )

        available = self.usable_remaining_seconds

        timeout = min(
            float(requested_seconds),
            available,
        )

        if timeout < minimum_seconds:
            raise RunBudgetExceeded(
                "Insufficient global time remaining "
                "for another bounded operation."
            )

        return timeout

    def reserve_candidate_attempt(self) -> int:
        self.require_time(
            minimum_seconds=0.1
        )

        if (
            self.candidate_attempts_used
            >= self.max_candidate_attempts
        ):
            raise RunBudgetExceeded(
                "Global candidate-attempt budget "
                "is exhausted."
            )

        self.candidate_attempts_used += 1

        return self.candidate_attempts_used

    def reserve_model_call(
        self,
        *,
        estimated_tokens: int = 0,
    ) -> int:
        if estimated_tokens < 0:
            raise ValueError(
                "estimated_tokens must be "
                "non-negative"
            )

        self.require_time(
            minimum_seconds=0.1
        )

        if (
            self.model_calls_used
            >= self.max_model_calls
        ):
            raise RunBudgetExceeded(
                "Global model-call budget "
                "is exhausted."
            )

        if self.max_total_tokens is not None:
            if (
                self.tokens_used
                + estimated_tokens
                > self.max_total_tokens
            ):
                raise RunBudgetExceeded(
                    "Global token budget "
                    "would be exceeded."
                )

        self.model_calls_used += 1

        return self.model_calls_used

    def record_tokens(
        self,
        tokens: int,
    ) -> None:
        if tokens < 0:
            raise ValueError(
                "token usage must be non-negative"
            )

        self.tokens_used += int(tokens)

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_wall_seconds": (
                self.total_wall_seconds
            ),
            "safety_margin_seconds": (
                self.safety_margin_seconds
            ),
            "elapsed_seconds": (
                self.elapsed_seconds
            ),
            "hard_remaining_seconds": (
                self.hard_remaining_seconds
            ),
            "usable_remaining_seconds": (
                self.usable_remaining_seconds
            ),
            "exhausted": self.exhausted,
            "max_candidate_attempts": (
                self.max_candidate_attempts
            ),
            "candidate_attempts_used": (
                self.candidate_attempts_used
            ),
            "remaining_candidate_attempts": (
                self.remaining_candidate_attempts
            ),
            "max_model_calls": (
                self.max_model_calls
            ),
            "model_calls_used": (
                self.model_calls_used
            ),
            "remaining_model_calls": (
                self.remaining_model_calls
            ),
            "max_total_tokens": (
                self.max_total_tokens
            ),
            "tokens_used": self.tokens_used,
            "remaining_tokens": (
                self.remaining_tokens
            ),
        }