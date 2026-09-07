from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

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
from agent.run_context import (
    RunContext,
)
from agent.spec_compiler import (
    compile_specification,
)
from agent.specification import (
    load_task_specification,
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

    pd.DataFrame(
        {
            "id": [1, 2],
            "input_value": [
                10.0,
                20.0,
            ],
        }
    ).to_csv(
        data_dir / "input.csv",
        index=False,
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
task_id = "quality-pipeline-test"
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
        run_id="quality-run"
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


class QualityPipelineTests(
    unittest.TestCase
):

    def test_batch_evaluation_advances_to_repair_stage(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _write_task(
                root
            )

            specification = (
                load_task_specification(
                    task_dir
                )
            )

            compiled = (
                compile_specification(
                    specification
                )
            )

            context = (
                RunContext
                .from_task_specification(
                    specification,
                    environ={
                        "QFBENCH_SEED": (
                            "42"
                        ),
                    },
                )
            )

            def generate(request):
                if request.candidate_id == 2:
                    return GeneratedCandidate(
                        code=(
                            "raise RuntimeError("
                            "'candidate failure'"
                            ")"
                        )
                    )

                return GeneratedCandidate(
                    code=_good_code()
                )

            state = _state_at_generate()

            production = (
                run_candidate_production(
                    task_dir=task_dir,
                    workspace_base_dir=(
                        root / "candidates"
                    ),
                    state=state,
                    run_context=context,
                    plan={},
                    specification=(
                        specification
                    ),
                    compiled_specification=(
                        compiled
                    ),
                    generator=(
                        GeneratorAdapter(
                            generate=generate,
                            uses_model_budget=False,
                            name="local-test",
                        )
                    ),
                    candidate_count=2,
                )
            )

            result = (
                run_initial_quality_evaluation(
                    task_dir=task_dir,
                    state=state,
                    production=production,
                    specification=(
                        specification
                    ),
                    compiled_specification=(
                        compiled
                    ),
                )
            )

            self.assertEqual(
                result.evaluated_candidate_ids,
                (1, 2),
            )

            self.assertEqual(
                state.expected_stage(),
                PipelineStage.TARGETED_REPAIR,
            )

    def test_execution_evidence_is_not_duplicated(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _write_task(
                root
            )

            specification = (
                load_task_specification(
                    task_dir
                )
            )

            compiled = (
                compile_specification(
                    specification
                )
            )

            context = (
                RunContext
                .from_task_specification(
                    specification,
                    environ={
                        "QFBENCH_SEED": "42",
                    },
                )
            )

            state = _state_at_generate()

            production = (
                run_candidate_production(
                    task_dir=task_dir,
                    workspace_base_dir=(
                        root / "candidates"
                    ),
                    state=state,
                    run_context=context,
                    plan={},
                    specification=(
                        specification
                    ),
                    compiled_specification=(
                        compiled
                    ),
                    generator=(
                        GeneratorAdapter(
                            generate=(
                                lambda request:
                                GeneratedCandidate(
                                    code=_good_code()
                                )
                            ),
                            uses_model_budget=False,
                        )
                    ),
                    candidate_count=1,
                )
            )

            result = (
                run_initial_quality_evaluation(
                    task_dir=task_dir,
                    state=state,
                    production=production,
                    specification=(
                        specification
                    ),
                    compiled_specification=(
                        compiled
                    ),
                )
            )

            attempt = (
                result
                .attempts_by_candidate_id[1]
            )

            execution_items = [
                item
                for item
                in attempt.structural_evidence
                if item.name == "execution"
            ]

            self.assertEqual(
                len(execution_items),
                1,
            )

    def test_good_candidate_becomes_validated(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _write_task(
                root
            )

            specification = (
                load_task_specification(
                    task_dir
                )
            )

            compiled = (
                compile_specification(
                    specification
                )
            )

            state = _state_at_generate()

            production = (
                run_candidate_production(
                    task_dir=task_dir,
                    workspace_base_dir=(
                        root / "candidates"
                    ),
                    state=state,
                    run_context=(
                        RunContext
                        .from_task_specification(
                            specification,
                            environ={
                                "QFBENCH_SEED": (
                                    "42"
                                ),
                            },
                        )
                    ),
                    plan={},
                    specification=(
                        specification
                    ),
                    compiled_specification=(
                        compiled
                    ),
                    generator=(
                        GeneratorAdapter(
                            generate=(
                                lambda request:
                                GeneratedCandidate(
                                    code=_good_code()
                                )
                            ),
                            uses_model_budget=False,
                        )
                    ),
                    candidate_count=1,
                )
            )

            run_initial_quality_evaluation(
                task_dir=task_dir,
                state=state,
                production=production,
                specification=specification,
                compiled_specification=(
                    compiled
                ),
            )

            candidate = (
                production.items[0]
                .candidate
            )

            self.assertEqual(
                candidate.current_status,
                "validated",
            )

            self.assertEqual(
                candidate.hard_failures,
                [],
            )

    def test_runtime_failure_is_preserved_for_repair(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _write_task(
                root
            )

            specification = (
                load_task_specification(
                    task_dir
                )
            )

            compiled = (
                compile_specification(
                    specification
                )
            )

            state = _state_at_generate()

            production = (
                run_candidate_production(
                    task_dir=task_dir,
                    workspace_base_dir=(
                        root / "candidates"
                    ),
                    state=state,
                    run_context=(
                        RunContext
                        .from_task_specification(
                            specification,
                            environ={
                                "QFBENCH_SEED": (
                                    "42"
                                ),
                            },
                        )
                    ),
                    plan={},
                    specification=(
                        specification
                    ),
                    compiled_specification=(
                        compiled
                    ),
                    generator=(
                        GeneratorAdapter(
                            generate=(
                                lambda request:
                                GeneratedCandidate(
                                    code=(
                                        "raise "
                                        "RuntimeError("
                                        "'boom')"
                                    )
                                )
                            ),
                            uses_model_budget=False,
                        )
                    ),
                    candidate_count=1,
                )
            )

            result = (
                run_initial_quality_evaluation(
                    task_dir=task_dir,
                    state=state,
                    production=production,
                    specification=(
                        specification
                    ),
                    compiled_specification=(
                        compiled
                    ),
                )
            )

            candidate = (
                production.items[0]
                .candidate
            )

            attempt = (
                result
                .attempts_by_candidate_id[1]
            )

            self.assertEqual(
                candidate.current_status,
                "failed",
            )

            self.assertNotEqual(
                attempt.return_code,
                0,
            )

            self.assertTrue(
                candidate.hard_failures
            )


if __name__ == "__main__":
    unittest.main()