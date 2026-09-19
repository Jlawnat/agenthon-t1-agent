from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent.executor import run_python_candidate
from agent.run_context import (
    RunContext,
    RuntimeConfigurationError,
)
from agent.specification import (
    TaskSpecification,
)


def _spec(
    *,
    runtime_seconds: int | None = 1800,
) -> TaskSpecification:
    return TaskSpecification(
        task_id="task-1",
        category="risk-management",
        instruction_text="test",
        required_output_paths=[],
        candidate_schema_fields=[],
        required_output_columns=[],
        runtime_seconds=runtime_seconds,
        network_mode=None,
        raw_card={},
    )


class RunContextTests(unittest.TestCase):

    def test_card_timeout_is_authoritative(
        self,
    ) -> None:
        context = (
            RunContext.from_task_specification(
                _spec(runtime_seconds=600),
                environ={
                    "QFBENCH_SEED": "42",
                },
            )
        )

        self.assertEqual(
            context.card_timeout_seconds,
            600,
        )

        self.assertEqual(
            context.budget.total_wall_seconds,
            600,
        )

    def test_missing_card_timeout_uses_fallback(
        self,
    ) -> None:
        context = (
            RunContext.from_task_specification(
                _spec(runtime_seconds=None),
                environ={
                    "QFBENCH_SEED": "7",
                },
                default_runtime_seconds=900,
            )
        )

        self.assertEqual(
            context.card_timeout_seconds,
            900,
        )

    def test_seed_is_captured(
        self,
    ) -> None:
        context = (
            RunContext.from_task_specification(
                _spec(),
                environ={
                    "QFBENCH_SEED": "1234",
                },
            )
        )

        self.assertEqual(
            context.qfbench_seed,
            1234,
        )

    def test_missing_seed_fails_closed(
        self,
    ) -> None:
        with self.assertRaises(
            RuntimeConfigurationError
        ):
            RunContext.from_task_specification(
                _spec(),
                environ={},
            )

    def test_invalid_seed_fails_closed(
        self,
    ) -> None:
        with self.assertRaises(
            RuntimeConfigurationError
        ):
            RunContext.from_task_specification(
                _spec(),
                environ={
                    "QFBENCH_SEED": "abc",
                },
            )

    def test_execution_environment_contains_seed(
        self,
    ) -> None:
        context = (
            RunContext.from_task_specification(
                _spec(),
                environ={
                    "QFBENCH_SEED": "99",
                },
            )
        )

        environment = (
            context.execution_environment()
        )

        self.assertEqual(
            environment["QFBENCH_SEED"],
            "99",
        )

        self.assertEqual(
            environment["PYTHONHASHSEED"],
            "99",
        )

    def test_executor_propagates_seed_to_child(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            script = root / "solver.py"

            script.write_text(
                """
import os

print(os.environ["QFBENCH_SEED"])
print(os.environ["PYTHONHASHSEED"])
""".strip()
                + "\n",
                encoding="utf-8",
            )

            result = run_python_candidate(
                script_path=script,
                cwd=root,
                timeout_seconds=5,
                env_overrides={
                    "QFBENCH_SEED": "31415",
                    "PYTHONHASHSEED": "31415",
                },
            )

            self.assertEqual(
                result.return_code,
                0,
            )

            lines = (
                result.stdout
                .strip()
                .splitlines()
            )

            self.assertEqual(
                lines,
                [
                    "31415",
                    "31415",
                ],
            )

    def test_bounded_timeout_uses_global_budget(
        self,
    ) -> None:
        context = (
            RunContext.from_task_specification(
                _spec(runtime_seconds=100),
                environ={
                    "QFBENCH_SEED": "5",
                },
                safety_margin_seconds=10,
            )
        )

        timeout = (
            context.bounded_timeout(
                requested_seconds=500
            )
        )

        self.assertLessEqual(
            timeout,
            90,
        )
    def test_default_budget_leaves_capacity_for_repairs(
        self,
    ) -> None:
        specification = TaskSpecification(
            task_id="repair-capacity-test",
            category="risk-management",
            instruction_text="test",
            required_output_paths=[],
            candidate_schema_fields=[],
            required_output_columns=[],
            runtime_seconds=600,
            network_mode=None,
            raw_card={},
        )

        context = (
            RunContext.from_task_specification(
                specification,
                environ={
                    "QFBENCH_SEED": "42",
                },
            )
        )

        self.assertEqual(
            context.budget.max_candidate_attempts,
            6,
        )

        self.assertEqual(
            context.budget.max_model_calls,
            12,
        )


if __name__ == "__main__":
    unittest.main()