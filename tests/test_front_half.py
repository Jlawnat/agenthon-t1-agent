from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent.front_half import (
    FrontHalfDependencies,
    run_deterministic_front_half,
)
from agent.orchestrator_state import (
    OrchestratorState,
    PipelineStage,
    RunStatus,
)


def _write_task(
    root: Path,
) -> Path:
    task_dir = root / "task"

    data_dir = (
        task_dir
        / "environment"
        / "data"
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        task_dir
        / "instruction.md"
    ).write_text(
        """
Produce the required result.

### /output/results.csv

| Column | Type |
|---|---|
| `id` | `int64` |
| `value` | `float64` |
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (
        task_dir
        / "card.toml"
    ).write_text(
        """
[metadata]
task_id = "front-half-test"
category = "risk-management"

[environment]
timeout_seconds = 600
""".strip()
        + "\n",
        encoding="utf-8",
    )

    return task_dir


class FrontHalfTests(unittest.TestCase):

    def test_exact_front_half_order(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _write_task(
                root
            )

            calls: list[str] = []

            def snapshot_adapter(
                received_task_dir: Path,
            ):
                calls.append(
                    "snapshot"
                )

                self.assertEqual(
                    received_task_dir,
                    task_dir.resolve(),
                )

                return {
                    "snapshot": "ok",
                }

            def plan_adapter(
                snapshot,
                specification,
                compiled,
            ):
                calls.append(
                    "plan"
                )

                self.assertEqual(
                    snapshot["snapshot"],
                    "ok",
                )

                self.assertEqual(
                    specification.task_id,
                    "front-half-test",
                )

                self.assertEqual(
                    compiled.task_id,
                    "front-half-test",
                )

                return {
                    "plan": "ok",
                }

            state = OrchestratorState(
                run_id="run-1"
            )

            result = (
                run_deterministic_front_half(
                    task_dir=task_dir,
                    state=state,
                    dependencies=(
                        FrontHalfDependencies(
                            snapshot=(
                                snapshot_adapter
                            ),
                            plan=(
                                plan_adapter
                            ),
                        )
                    ),
                    environ={
                        "QFBENCH_SEED": "42",
                    },
                )
            )

            self.assertEqual(
                calls,
                [
                    "snapshot",
                    "plan",
                ],
            )

            self.assertEqual(
                state.completed_stages,
                [
                    PipelineStage.SNAPSHOT,
                    PipelineStage.SPECIFICATION,
                    PipelineStage.COMPILE,
                    PipelineStage.PLAN,
                ],
            )

            self.assertEqual(
                state.expected_stage(),
                PipelineStage.GENERATE,
            )

            self.assertEqual(
                result.run_context.qfbench_seed,
                42,
            )

            self.assertEqual(
                (
                    result
                    .run_context
                    .card_timeout_seconds
                ),
                600,
            )

    def test_task_id_is_propagated_to_state(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _write_task(
                root
            )

            state = OrchestratorState(
                run_id="run-1"
            )

            run_deterministic_front_half(
                task_dir=task_dir,
                state=state,
                dependencies=(
                    FrontHalfDependencies(
                        snapshot=lambda path: {},
                        plan=(
                            lambda snapshot,
                            specification,
                            compiled: {}
                        ),
                    )
                ),
                environ={
                    "QFBENCH_SEED": "7",
                },
            )

            self.assertEqual(
                state.task_id,
                "front-half-test",
            )

    def test_snapshot_failure_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _write_task(
                root
            )

            state = OrchestratorState(
                run_id="run-1"
            )

            def broken_snapshot(
                task_dir: Path,
            ):
                raise RuntimeError(
                    "snapshot failed"
                )

            with self.assertRaises(
                RuntimeError
            ):
                run_deterministic_front_half(
                    task_dir=task_dir,
                    state=state,
                    dependencies=(
                        FrontHalfDependencies(
                            snapshot=(
                                broken_snapshot
                            ),
                            plan=(
                                lambda a, b, c: {}
                            ),
                        )
                    ),
                    environ={
                        "QFBENCH_SEED": "1",
                    },
                )

            self.assertEqual(
                state.status,
                RunStatus.FAILED,
            )

            self.assertEqual(
                state.history[-1].stage,
                PipelineStage.SNAPSHOT,
            )

            self.assertEqual(
                state.history[-1].status,
                "failed",
            )

    def test_specification_failure_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = (
                root
                / "missing-instruction"
            )

            task_dir.mkdir()

            state = OrchestratorState(
                run_id="run-1"
            )

            with self.assertRaises(
                FileNotFoundError
            ):
                run_deterministic_front_half(
                    task_dir=task_dir,
                    state=state,
                    dependencies=(
                        FrontHalfDependencies(
                            snapshot=lambda path: {},
                            plan=(
                                lambda a, b, c: {}
                            ),
                        )
                    ),
                    environ={
                        "QFBENCH_SEED": "1",
                    },
                )

            self.assertEqual(
                state.status,
                RunStatus.FAILED,
            )

            self.assertEqual(
                state.history[-1].stage,
                PipelineStage.SPECIFICATION,
            )

    def test_planner_failure_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _write_task(
                root
            )

            state = OrchestratorState(
                run_id="run-1"
            )

            def broken_plan(
                snapshot,
                specification,
                compiled,
            ):
                raise ValueError(
                    "planning failed"
                )

            with self.assertRaises(
                ValueError
            ):
                run_deterministic_front_half(
                    task_dir=task_dir,
                    state=state,
                    dependencies=(
                        FrontHalfDependencies(
                            snapshot=lambda path: {},
                            plan=broken_plan,
                        )
                    ),
                    environ={
                        "QFBENCH_SEED": "11",
                    },
                )

            self.assertEqual(
                state.status,
                RunStatus.FAILED,
            )

            self.assertEqual(
                state.history[-1].stage,
                PipelineStage.PLAN,
            )

            self.assertEqual(
                state.completed_stages,
                [
                    PipelineStage.SNAPSHOT,
                    PipelineStage.SPECIFICATION,
                    PipelineStage.COMPILE,
                ],
            )

    def test_front_half_never_advances_to_generate(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _write_task(
                root
            )

            state = OrchestratorState(
                run_id="run-1"
            )

            run_deterministic_front_half(
                task_dir=task_dir,
                state=state,
                dependencies=(
                    FrontHalfDependencies(
                        snapshot=lambda path: {},
                        plan=(
                            lambda a, b, c: {}
                        ),
                    )
                ),
                environ={
                    "QFBENCH_SEED": "99",
                },
            )

            self.assertNotIn(
                PipelineStage.GENERATE,
                state.completed_stages,
            )

            self.assertEqual(
                state.expected_stage(),
                PipelineStage.GENERATE,
            )


if __name__ == "__main__":
    unittest.main()