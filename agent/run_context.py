from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import os

from agent.run_budget import RunBudget
from agent.specification import TaskSpecification


DEFAULT_RUNTIME_SECONDS = 1800.0


class RuntimeConfigurationError(RuntimeError):
    """Raised when required runtime configuration is invalid."""


@dataclass
class RunContext:
    """
    Runtime configuration for one solve invocation.

    Responsibilities:
    - derive the global deadline from the task card
    - capture QFBENCH_SEED once
    - own the shared RunBudget
    - provide deterministic subprocess environment
    """

    task_id: str | None
    card_timeout_seconds: float
    qfbench_seed: int
    budget: RunBudget

    @classmethod
    def from_task_specification(
        cls,
        specification: TaskSpecification,
        *,
        environ: Mapping[str, str] | None = None,
        default_runtime_seconds: float = (
            DEFAULT_RUNTIME_SECONDS
        ),
        safety_margin_seconds: float = 5.0,
        max_candidate_attempts: int = 6,
        max_model_calls: int = 12,
        max_total_tokens: int | None = None,
    ) -> "RunContext":

        environment = (
            os.environ
            if environ is None
            else environ
        )

        timeout = specification.runtime_seconds

        if timeout is None:
            timeout = default_runtime_seconds

        if (
            isinstance(timeout, bool)
            or not isinstance(
                timeout,
                (int, float),
            )
            or timeout <= 0
        ):
            raise RuntimeConfigurationError(
                "Task runtime must be a positive "
                "number of seconds."
            )

        timeout = float(timeout)

        raw_seed = environment.get(
            "QFBENCH_SEED"
        )

        if raw_seed is None:
            raise RuntimeConfigurationError(
                "QFBENCH_SEED is required for "
                "deterministic execution."
            )

        try:
            seed = int(raw_seed)

        except (TypeError, ValueError) as exc:
            raise RuntimeConfigurationError(
                "QFBENCH_SEED must be an integer."
            ) from exc

        if seed < 0:
            raise RuntimeConfigurationError(
                "QFBENCH_SEED must be "
                "non-negative."
            )

        budget = RunBudget(
            total_wall_seconds=timeout,
            safety_margin_seconds=(
                safety_margin_seconds
            ),
            max_candidate_attempts=(
                max_candidate_attempts
            ),
            max_model_calls=max_model_calls,
            max_total_tokens=(
                max_total_tokens
            ),
        )

        return cls(
            task_id=specification.task_id,
            card_timeout_seconds=timeout,
            qfbench_seed=seed,
            budget=budget,
        )

    def execution_environment(
        self,
    ) -> dict[str, str]:
        """
        Environment propagated to candidate
        subprocesses.

        PYTHONHASHSEED is included because it must
        exist before the child interpreter starts.
        """

        seed = str(
            self.qfbench_seed
        )

        return {
            "QFBENCH_SEED": seed,
            "PYTHONHASHSEED": seed,
        }

    def bounded_timeout(
        self,
        requested_seconds: float | None = None,
    ) -> float:
        """
        Calculate an operation timeout without
        crossing the card/global deadline.
        """

        requested = (
            self.card_timeout_seconds
            if requested_seconds is None
            else float(requested_seconds)
        )

        return self.budget.bounded_timeout(
            requested
        )

    def snapshot(
        self,
    ) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "card_timeout_seconds": (
                self.card_timeout_seconds
            ),
            "qfbench_seed": (
                self.qfbench_seed
            ),
            "budget": self.budget.snapshot(),
        }