from __future__ import annotations

import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import pandas as pd

from agent.budget import RepairBudget
from agent.candidate_production import (
    GeneratedCandidate,
    GeneratorAdapter,
    run_candidate_production,
)
from agent.orchestrator_state import (
    OrchestratorState,
    PipelineStage,
)
from agent.quality_pipeline import (
    run_initial_quality_evaluation,
)
from agent.repair_pipeline import (
    RepairedCandidate,
    RepairAdapter,
    run_targeted_repair_stage,
)
from agent.run_context import RunContext
from agent.spec_compiler import compile_specification
from agent.specification import load_task_specification


def _write_task(root: Path) -> Path:
    task_dir = root / "task"
    data_dir = task_dir / "environment" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "id": [1, 2],
            "input_value": [10.0, 20.0],
        }
    ).to_csv(
        data_dir / "input.csv",
        index=False,
    )

    (task_dir / "instruction.md").write_text(
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

    (task_dir / "card.toml").write_text(
        """
[metadata]
task_id = "repair-pipeline-test"
category = "risk-management"

[environment]
timeout_seconds = 600
""".strip()
        + "\n",
        encoding="utf-8",
    )

    return task_dir


def _state_at_generate() -> OrchestratorState:
    state = OrchestratorState(
        run_id="repair-run"
    )

    for stage in (
        PipelineStage.SNAPSHOT,
        PipelineStage.SPECIFICATION,
        PipelineStage.COMPILE,
        PipelineStage.PLAN,
    ):
        state.start_stage(stage)
        state.complete_stage(stage)

    return state


def _load_task(task_dir: Path):
    specification = load_task_specification(
        task_dir
    )

    compiled = compile_specification(
        specification
    )

    context = RunContext.from_task_specification(
        specification,
        environ={
            "QFBENCH_SEED": "42",
        },
    )

    return (
        specification,
        compiled,
        context,
    )


def _good_code() -> str:
    return """
import csv
from pathlib import Path

input_path = Path(
    "input/data/input.csv"
)

output_dir = Path("output")
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

rows = []

with input_path.open(
    "r",
    encoding="utf-8",
    newline="",
) as handle:
    reader = csv.DictReader(handle)

    for row in reader:
        rows.append(
            {
                "id": int(row["id"]),
                "value": float(
                    row["input_value"]
                ),
            }
        )

with (
    output_dir
    / "results.csv"
).open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "id",
            "value",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)
""".strip()


def _seed_safe_code() -> str:
    return """
import csv
import os
from pathlib import Path

if "QFBENCH_SEED" not in os.environ:
    raise RuntimeError(
        "seed missing"
    )

for forbidden in (
    "MODEL_ENDPOINT",
    "MODEL_NAME",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
):
    if forbidden in os.environ:
        raise RuntimeError(
            "sensitive environment leaked"
        )

seed = float(
    os.environ["QFBENCH_SEED"]
)

input_path = Path(
    "input/data/input.csv"
)

output_dir = Path("output")
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

rows = []

with input_path.open(
    "r",
    encoding="utf-8",
    newline="",
) as handle:
    reader = csv.DictReader(handle)

    for row in reader:
        rows.append(
            {
                "id": int(row["id"]),
                "value": seed,
            }
        )

with (
    output_dir
    / "results.csv"
).open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "id",
            "value",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)
""".strip()


def _runtime_failure_code() -> str:
    return """
raise RuntimeError(
    "public candidate failure"
)
""".strip()


def _produce_and_evaluate(
    *,
    root: Path,
    generate,
    candidate_count: int,
):
    task_dir = _write_task(
        root
    )

    (
        specification,
        compiled,
        context,
    ) = _load_task(
        task_dir
    )

    state = _state_at_generate()

    production = run_candidate_production(
        task_dir=task_dir,
        workspace_base_dir=(
            root / "initial_candidates"
        ),
        state=state,
        run_context=context,
        plan={},
        specification=specification,
        compiled_specification=compiled,
        generator=GeneratorAdapter(
            generate=generate,
            uses_model_budget=False,
            name="local-test-generator",
        ),
        candidate_count=candidate_count,
    )

    quality = run_initial_quality_evaluation(
        task_dir=task_dir,
        state=state,
        production=production,
        specification=specification,
        compiled_specification=compiled,
    )

    return (
        task_dir,
        specification,
        compiled,
        context,
        state,
        production,
        quality,
    )


class RepairPipelineTests(
    unittest.TestCase
):

    def test_valid_candidate_is_never_repaired(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            (
                task_dir,
                specification,
                compiled,
                context,
                state,
                production,
                quality,
            ) = _produce_and_evaluate(
                root=root,
                generate=(
                    lambda request:
                    GeneratedCandidate(
                        code=_good_code()
                    )
                ),
                candidate_count=1,
            )

            calls: list[int] = []

            def repair(request):
                calls.append(
                    request.candidate_id
                )

                raise AssertionError(
                    "Valid candidate must "
                    "not be repaired."
                )

            result = run_targeted_repair_stage(
                task_dir=task_dir,
                repair_workspace_base_dir=(
                    root / "repairs"
                ),
                state=state,
                run_context=context,
                quality=quality,
                specification=specification,
                compiled_specification=compiled,
                repairer=RepairAdapter(
                    repair=repair,
                    uses_model_budget=False,
                ),
            )

            self.assertEqual(
                calls,
                [],
            )

            self.assertEqual(
                result.attempted_candidate_ids,
                (),
            )

            self.assertEqual(
                result.repaired_candidate_ids,
                (),
            )

            self.assertEqual(
                result.skipped_candidate_ids,
                (1,),
            )

            self.assertEqual(
                len(
                    production
                    .items[0]
                    .candidate
                    .attempts
                ),
                1,
            )

            self.assertEqual(
                state.expected_stage(),
                PipelineStage.SELECT,
            )

    def test_runtime_failure_repairs_to_attempt_two(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            (
                task_dir,
                specification,
                compiled,
                context,
                state,
                production,
                quality,
            ) = _produce_and_evaluate(
                root=root,
                generate=(
                    lambda request:
                    GeneratedCandidate(
                        code=(
                            _runtime_failure_code()
                        )
                    )
                ),
                candidate_count=1,
            )

            candidate = (
                production
                .items[0]
                .candidate
            )

            self.assertEqual(
                candidate.current_status,
                "failed",
            )

            self.assertEqual(
                len(candidate.attempts),
                1,
            )

            result = run_targeted_repair_stage(
                task_dir=task_dir,
                repair_workspace_base_dir=(
                    root / "repairs"
                ),
                state=state,
                run_context=context,
                quality=quality,
                specification=specification,
                compiled_specification=compiled,
                repairer=RepairAdapter(
                    repair=(
                        lambda request:
                        RepairedCandidate(
                            code=_good_code()
                        )
                    ),
                    uses_model_budget=False,
                    name="local-repair",
                ),
            )

            self.assertEqual(
                result.attempted_candidate_ids,
                (1,),
            )

            self.assertEqual(
                result.repaired_candidate_ids,
                (1,),
            )

            self.assertEqual(
                candidate.current_status,
                "validated",
            )

            self.assertEqual(
                len(candidate.attempts),
                2,
            )

            self.assertEqual(
                candidate.attempts[1]
                .attempt_number,
                2,
            )

            self.assertIsNotNone(
                candidate.attempts[1]
                .repair_reason
            )

            self.assertEqual(
                state.expected_stage(),
                PipelineStage.SELECT,
            )

    def test_syntax_invalid_candidate_can_recover(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            def generate(request):
                if request.candidate_id == 1:
                    return GeneratedCandidate(
                        code=_good_code()
                    )

                return GeneratedCandidate(
                    code="def broken(:"
                )

            (
                task_dir,
                specification,
                compiled,
                context,
                state,
                production,
                quality,
            ) = _produce_and_evaluate(
                root=root,
                generate=generate,
                candidate_count=2,
            )

            second = (
                production
                .items[1]
                .candidate
            )

            self.assertEqual(
                second.current_status,
                "code_invalid",
            )

            labels: list[str] = []

            def repair(request):
                labels.append(
                    request.brief.failure_label
                )

                return RepairedCandidate(
                    code=_good_code()
                )

            result = run_targeted_repair_stage(
                task_dir=task_dir,
                repair_workspace_base_dir=(
                    root / "repairs"
                ),
                state=state,
                run_context=context,
                quality=quality,
                specification=specification,
                compiled_specification=compiled,
                repairer=RepairAdapter(
                    repair=repair,
                    uses_model_budget=False,
                ),
            )

            self.assertEqual(
                labels,
                ["syntax"],
            )

            self.assertEqual(
                result.attempted_candidate_ids,
                (2,),
            )

            self.assertEqual(
                result.repaired_candidate_ids,
                (2,),
            )

            self.assertEqual(
                second.current_status,
                "validated",
            )

            self.assertEqual(
                len(second.attempts),
                2,
            )

            self.assertEqual(
                second.attempts[0]
                .attempt_number,
                1,
            )

            self.assertEqual(
                second.attempts[1]
                .attempt_number,
                2,
            )

    def test_repair_budget_caps_candidates_deterministically(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            (
                task_dir,
                specification,
                compiled,
                context,
                state,
                production,
                quality,
            ) = _produce_and_evaluate(
                root=root,
                generate=(
                    lambda request:
                    GeneratedCandidate(
                        code=(
                            _runtime_failure_code()
                        )
                    )
                ),
                candidate_count=2,
            )

            repair_budget = RepairBudget(
                max_attempts=1,
                max_total_tokens=1000,
                max_wall_seconds=30.0,
            )

            result = run_targeted_repair_stage(
                task_dir=task_dir,
                repair_workspace_base_dir=(
                    root / "repairs"
                ),
                state=state,
                run_context=context,
                quality=quality,
                specification=specification,
                compiled_specification=compiled,
                repairer=RepairAdapter(
                    repair=(
                        lambda request:
                        RepairedCandidate(
                            code=_good_code()
                        )
                    ),
                    uses_model_budget=False,
                ),
                repair_budget=repair_budget,
            )

            self.assertEqual(
                result.attempted_candidate_ids,
                (1,),
            )

            self.assertEqual(
                result.repaired_candidate_ids,
                (1,),
            )

            self.assertIn(
                2,
                result.skipped_candidate_ids,
            )

            self.assertIn(
                "budget",
                result.failure_messages[2]
                .lower(),
            )

            self.assertEqual(
                repair_budget.attempts_used,
                1,
            )

    def test_repair_workspace_preserves_initial_workspace(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            (
                task_dir,
                specification,
                compiled,
                context,
                state,
                production,
                quality,
            ) = _produce_and_evaluate(
                root=root,
                generate=(
                    lambda request:
                    GeneratedCandidate(
                        code=(
                            _runtime_failure_code()
                        )
                    )
                ),
                candidate_count=1,
            )

            item = production.items[0]

            self.assertIsNotNone(
                item.workspace
            )

            initial_workspace = (
                item.workspace.root_dir
            )

            initial_solver = (
                item.solver_path
            )

            self.assertIsNotNone(
                initial_solver
            )

            initial_code = (
                initial_solver.read_text(
                    encoding="utf-8"
                )
            )

            run_targeted_repair_stage(
                task_dir=task_dir,
                repair_workspace_base_dir=(
                    root / "repairs"
                ),
                state=state,
                run_context=context,
                quality=quality,
                specification=specification,
                compiled_specification=compiled,
                repairer=RepairAdapter(
                    repair=(
                        lambda request:
                        RepairedCandidate(
                            code=_good_code()
                        )
                    ),
                    uses_model_budget=False,
                ),
            )

            self.assertTrue(
                initial_workspace.exists()
            )

            self.assertTrue(
                initial_solver.exists()
            )

            self.assertEqual(
                initial_solver.read_text(
                    encoding="utf-8"
                ),
                initial_code,
            )

            self.assertIsNotNone(
                item.workspace
            )

            self.assertNotEqual(
                item.workspace.root_dir,
                initial_workspace,
            )

    def test_repaired_subprocess_gets_seed_but_not_sensitive_environment(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            (
                task_dir,
                specification,
                compiled,
                context,
                state,
                production,
                quality,
            ) = _produce_and_evaluate(
                root=root,
                generate=(
                    lambda request:
                    GeneratedCandidate(
                        code=(
                            _runtime_failure_code()
                        )
                    )
                ),
                candidate_count=1,
            )

            item = production.items[0]

            sensitive_parent = {
                "MODEL_ENDPOINT": (
                    "https://example.invalid"
                ),
                "MODEL_NAME": (
                    "private-model"
                ),
                "OPENAI_API_KEY": (
                    "secret"
                ),
                "ANTHROPIC_API_KEY": (
                    "secret"
                ),
            }

            with patch.dict(
                os.environ,
                sensitive_parent,
                clear=False,
            ):
                result = (
                    run_targeted_repair_stage(
                        task_dir=task_dir,
                        repair_workspace_base_dir=(
                            root / "repairs"
                        ),
                        state=state,
                        run_context=context,
                        quality=quality,
                        specification=(
                            specification
                        ),
                        compiled_specification=(
                            compiled
                        ),
                        repairer=RepairAdapter(
                            repair=(
                                lambda request:
                                RepairedCandidate(
                                    code=(
                                        _seed_safe_code()
                                    )
                                )
                            ),
                            uses_model_budget=False,
                        ),
                    )
                )

            self.assertEqual(
                result.repaired_candidate_ids,
                (1,),
            )

            self.assertEqual(
                item.candidate.current_status,
                "validated",
            )

            self.assertIsNotNone(
                item.workspace
            )

            output_path = (
                item.workspace.output_dir
                / "results.csv"
            )

            frame = pd.read_csv(
                output_path
            )

            expected_seed = float(
                item.candidate_seed
            )

            self.assertTrue(
                (
                    frame["value"]
                    == expected_seed
                ).all()
            )


if __name__ == "__main__":
    unittest.main()