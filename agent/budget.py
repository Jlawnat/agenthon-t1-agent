from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class RepairBudgetExceeded(RuntimeError):
    """Raised when another repair would exceed its budget."""


@dataclass
class RepairBudget:
    """
    Bounded budget for targeted candidate repair.

    Repairs are limited independently by:
    - number of repair attempts
    - model tokens
    - repair wall time
    """

    max_attempts: int = 2
    max_total_tokens: int = 6000
    max_wall_seconds: float = 120.0

    attempts_used: int = 0
    tokens_used: int = 0
    wall_seconds_used: float = 0.0

    def __post_init__(self) -> None:
        if self.max_attempts < 0:
            raise ValueError(
                "max_attempts must be non-negative"
            )

        if self.max_total_tokens < 0:
            raise ValueError(
                "max_total_tokens must be non-negative"
            )

        if self.max_wall_seconds < 0:
            raise ValueError(
                "max_wall_seconds must be non-negative"
            )

        if (
            self.attempts_used < 0
            or self.tokens_used < 0
            or self.wall_seconds_used < 0
        ):
            raise ValueError(
                "repair budget usage must be non-negative"
            )

    @property
    def remaining_attempts(self) -> int:
        return max(
            0,
            self.max_attempts - self.attempts_used,
        )

    @property
    def remaining_tokens(self) -> int:
        return max(
            0,
            self.max_total_tokens - self.tokens_used,
        )

    @property
    def remaining_wall_seconds(self) -> float:
        return max(
            0.0,
            self.max_wall_seconds
            - self.wall_seconds_used,
        )

    def can_reserve(
        self,
        *,
        estimated_tokens: int = 0,
        estimated_wall_seconds: float = 0.0,
    ) -> bool:
        if (
            estimated_tokens < 0
            or estimated_wall_seconds < 0
        ):
            raise ValueError(
                "estimated repair usage must be "
                "non-negative"
            )

        return (
            self.attempts_used < self.max_attempts
            and self.tokens_used
            < self.max_total_tokens
            and self.wall_seconds_used
            < self.max_wall_seconds
            and (
                self.tokens_used
                + estimated_tokens
                <= self.max_total_tokens
            )
            and (
                self.wall_seconds_used
                + estimated_wall_seconds
                <= self.max_wall_seconds
            )
        )

    def reserve_attempt(
        self,
        *,
        estimated_tokens: int = 0,
        estimated_wall_seconds: float = 0.0,
    ) -> int:
        """
        Reserve one repair attempt.

        Returns the 1-based repair attempt number.
        """

        if not self.can_reserve(
            estimated_tokens=estimated_tokens,
            estimated_wall_seconds=(
                estimated_wall_seconds
            ),
        ):
            raise RepairBudgetExceeded(
                "Targeted repair budget is exhausted; "
                "no further repair attempt is allowed."
            )

        self.attempts_used += 1

        return self.attempts_used

    def record_usage(
        self,
        *,
        tokens: int = 0,
        wall_seconds: float = 0.0,
    ) -> None:
        if (
            tokens < 0
            or wall_seconds < 0
        ):
            raise ValueError(
                "repair usage must be non-negative"
            )

        self.tokens_used += int(tokens)

        self.wall_seconds_used += float(
            wall_seconds
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "remaining_attempts": (
                self.remaining_attempts
            ),
            "remaining_tokens": (
                self.remaining_tokens
            ),
            "remaining_wall_seconds": (
                self.remaining_wall_seconds
            ),
        }