from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from agent.candidate_production import (
    GeneratedCandidate,
    GeneratorAdapter,
)
from agent.front_half import (
    FrontHalfDependencies,
)
from agent.orchestrator import (
    solve_task,
)
from agent.orchestrator_state import (
    RunStatus,
)
from agent.repair_pipeline import (
    RepairedCandidate,
    RepairAdapter,
)
def _write_task(
    *,
    root: Path,
    output_format: str,
) -> Path:
    task_dir = (
        root
        / "task"
    )

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
            "id": [
                1,
                2,
            ],
            "input_value": [
                10.0,
                20.0,
            ],
        }
    ).to_csv(
        data_dir
        / "input.csv",
        index=False,
    )

    output_name = (
        f"results.{output_format}"
    )

    (
        task_dir
        / "instruction.md"
    ).write_text(
        f"""
Read the supplied input data and produce the required result.

### /output/{output_name}

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
        f"""
[metadata]
task_id = "orchestrator-e2e-{output_format}"
category = "risk-management"

[environment]
timeout_seconds = 600
""".strip()
        + "\n",
        encoding="utf-8",
    )

    return task_dir

def _snapshot_adapter(
    *args,
    **kwargs,
):
    return {
        "source": "e2e-test",
        "deterministic": True,
    }


def _plan_adapter(
    *args,
    **kwargs,
):
    return {
        "steps": [
            "read input",
            "compute result",
            "write deliverable",
        ],
    }


def _dependencies(
) -> FrontHalfDependencies:
    return FrontHalfDependencies(
        snapshot=(
            _snapshot_adapter
        ),
        plan=(
            _plan_adapter
        ),
    )

def _good_code(
    output_format: str,
) -> str:
    if output_format == "csv":
        writer = """
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
"""

    elif output_format == "json":
        writer = """
with (
    output_dir
    / "results.json"
).open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        rows,
        handle,
        ensure_ascii=False,
    )
"""

    elif output_format == "parquet":
        writer = """
frame = pd.DataFrame(
    rows
)

frame = frame.astype(
    {
        "id": "int64",
        "value": "float64",
    }
)

frame.to_parquet(
    output_dir
    / "results.parquet",
    index=False,
)
"""

    else:
        raise ValueError(
            f"Unsupported format: {output_format}"
        )

    imports = """
import csv
import json
import os
from pathlib import Path
"""

    if output_format == "parquet":
        imports += (
            "\nimport pandas as pd\n"
        )

    return (
        imports
        + """
input_path = Path(
    "input/data/input.csv"
)

output_dir = Path(
    "output"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

if "QFBENCH_SEED" not in os.environ:
    raise RuntimeError(
        "QFBENCH_SEED missing"
    )

seed = float(
    os.environ[
        "QFBENCH_SEED"
    ]
)

rows = []

with input_path.open(
    "r",
    encoding="utf-8",
    newline="",
) as handle:
    reader = csv.DictReader(
        handle
    )

    for row in reader:
        rows.append(
            {
                "id": int(
                    row["id"]
                ),
                "value": seed,
            }
        )

"""
        + writer
    ).strip()


def _runtime_failure_code(
) -> str:
    return """
raise RuntimeError(
    "intentional candidate failure"
)
""".strip()


def _good_generator(
    output_format: str,
) -> GeneratorAdapter:

    def generate(
        request,
    ):
        return GeneratedCandidate(
            code=_good_code(
                output_format
            ),
            approach_name=(
                "deterministic-e2e"
            ),
            strategy={
                "format": (
                    output_format
                ),
            },
        )

    return GeneratorAdapter(
        generate=generate,
        uses_model_budget=False,
        name="local-e2e-generator",
    )


def _partial_failure_generator(
    output_format: str,
) -> GeneratorAdapter:

    def generate(
        request,
    ):
        if (
            request.candidate_id
            == 1
        ):
            return GeneratedCandidate(
                code=(
                    _runtime_failure_code()
                ),
                approach_name=(
                    "intentional-failure"
                ),
            )

        return GeneratedCandidate(
            code=_good_code(
                output_format
            ),
            approach_name=(
                "valid-fallback"
            ),
        )

    return GeneratorAdapter(
        generate=generate,
        uses_model_budget=False,
        name=(
            "partial-failure-generator"
        ),
    )


def _failing_repairer(
) -> RepairAdapter:

    def repair(
        request,
    ):
        raise RuntimeError(
            "intentional repair failure"
        )

    return RepairAdapter(
        repair=repair,
        uses_model_budget=False,
        name="intentional-failing-repairer",
    )


def _successful_repairer(
    output_format: str,
) -> RepairAdapter:

    def repair(
        request,
    ):
        return RepairedCandidate(
            code=_good_code(
                output_format
            ),
            tokens_used=0,
        )

    return RepairAdapter(
        repair=repair,
        uses_model_budget=False,
        name="local-e2e-repairer",
    )


def _read_output(
    *,
    output_dir: Path,
    output_format: str,
):
    path = (
        output_dir
        / f"results.{output_format}"
    )

    if output_format == "csv":
        return pd.read_csv(
            path
        )

    if output_format == "json":
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            return pd.DataFrame(
                json.load(
                    handle
                )
            )

    if output_format == "parquet":
        return pd.read_parquet(
            path
        )

    raise ValueError(
        output_format
    )


def _solve(
    *,
    root: Path,
    output_format: str,
    generator: GeneratorAdapter,
    repairer: RepairAdapter,
    candidate_count: int,
    seed: int = 12345,
):
    task_dir = _write_task(
        root=root,
        output_format=(
            output_format
        ),
    )

    final_output_dir = (
        root
        / "final_output"
    )

    work_root = (
        root
        / "work"
    )

    result = solve_task(
        task_dir=task_dir,
        final_output_dir=(
            final_output_dir
        ),
        work_root=work_root,
        front_half_dependencies=(
            _dependencies()
        ),
        generator=generator,
        repairer=repairer,
        candidate_count=(
            candidate_count
        ),
        execution_timeout_seconds=30.0,
        audit_timeout_seconds=30.0,
        environ={
            "QFBENCH_SEED": (
                str(seed)
            ),
        },
    )

    return (
        task_dir,
        final_output_dir,
        result,
    )

class OrchestratorEndToEndTests(
    unittest.TestCase
):

    def _assert_format_run(
        self,
        output_format: str,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            (
                task_dir,
                final_output_dir,
                result,
            ) = _solve(
                root=root,
                output_format=(
                    output_format
                ),
                generator=(
                    _good_generator(
                        output_format
                    )
                ),
                repairer=(
                    _failing_repairer()
                ),
                candidate_count=1,
            )

            self.assertTrue(
                (
                    final_output_dir
                    / (
                        f"results."
                        f"{output_format}"
                    )
                ).exists()
            )

            frame = _read_output(
                output_dir=(
                    final_output_dir
                ),
                output_format=(
                    output_format
                ),
            )

            self.assertEqual(
                list(
                    frame.columns
                ),
                [
                    "id",
                    "value",
                ],
            )

            self.assertEqual(
                frame["id"].tolist(),
                [
                    1,
                    2,
                ],
            )

            self.assertTrue(
                frame[
                    "value"
                ].notna().all()
            )

            self.assertEqual(
                result.state.status,
                RunStatus.SUCCEEDED,
            )

            self.assertTrue(
                result.audit.audit.passed
            )

            self.assertEqual(
                result.publish.published_files,
                (
                    f"results."
                    f"{output_format}",
                ),
            )

            # Clean-room audit output is a
            # separate workspace from the selected
            # candidate workspace.
            self.assertNotEqual(
                result.audit
                .workspace
                .root_dir,
                result.selection
                .selected_item
                .workspace
                .root_dir,
            )

            audited_path = (
                result.audit
                .audited_output_dir
                / (
                    f"results."
                    f"{output_format}"
                )
            )

            published_path = (
                final_output_dir
                / (
                    f"results."
                    f"{output_format}"
                )
            )

            self.assertTrue(
                audited_path.exists()
            )

            self.assertTrue(
                published_path.exists()
            )

            # Structured run records must exist.
            self.assertTrue(
                (
                    result.run_dir
                    / "events.jsonl"
                ).exists()
            )

            self.assertTrue(
                (
                    result.run_dir
                    / "checkpoint.json"
                ).exists()
            )

    def test_csv_full_one_command_solve(
        self,
    ) -> None:
        self._assert_format_run(
            "csv"
        )

    def test_json_full_one_command_solve(
        self,
    ) -> None:
        self._assert_format_run(
            "json"
        )

    def test_parquet_full_one_command_solve(
        self,
    ) -> None:
        self._assert_format_run(
            "parquet"
        )

    def test_partial_candidate_failure_preserves_valid_candidate(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            (
                task_dir,
                final_output_dir,
                result,
            ) = _solve(
                root=root,
                output_format="csv",
                generator=(
                    _partial_failure_generator(
                        "csv"
                    )
                ),
                repairer=(
                    _failing_repairer()
                ),
                candidate_count=3,
            )

            self.assertEqual(
                result.state.status,
                RunStatus.SUCCEEDED,
            )

            # Candidate 1 intentionally failed,
            # therefore a later valid candidate
            # must win.
            self.assertNotEqual(
                result.selection
                .selected_candidate_id,
                1,
            )

            self.assertFalse(
                result.selection
                .selection
                .first_candidate_valid
            )

            self.assertTrue(
                result.selection
                .selection
                .any_of_three_valid
            )

            self.assertTrue(
                (
                    final_output_dir
                    / "results.csv"
                ).exists()
            )

            frame = pd.read_csv(
                final_output_dir
                / "results.csv"
            )

            self.assertEqual(
                len(frame),
                2,
            )

    def test_targeted_repair_can_rescue_only_candidate(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            def generate(
                request,
            ):
                return GeneratedCandidate(
                    code=(
                        _runtime_failure_code()
                    ),
                    approach_name=(
                        "repair-required"
                    ),
                )

            generator = (
                GeneratorAdapter(
                    generate=generate,
                    uses_model_budget=False,
                    name=(
                        "repair-required-generator"
                    ),
                )
            )

            (
                task_dir,
                final_output_dir,
                result,
            ) = _solve(
                root=root,
                output_format="csv",
                generator=generator,
                repairer=(
                    _successful_repairer(
                        "csv"
                    )
                ),
                candidate_count=1,
            )

            self.assertEqual(
                result.state.status,
                RunStatus.SUCCEEDED,
            )

            self.assertEqual(
                result.repair
                .attempted_candidate_ids,
                (1,),
            )

            self.assertEqual(
                result.repair
                .repaired_candidate_ids,
                (1,),
            )

            candidate = (
                result.production
                .items[0]
                .candidate
            )

            self.assertGreaterEqual(
                len(
                    candidate.attempts
                ),
                2,
            )

            self.assertEqual(
                candidate
                .current_status,
                "validated",
            )

            self.assertTrue(
                (
                    final_output_dir
                    / "results.csv"
                ).exists()
            )

    def test_same_seed_replays_deterministically(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base = Path(
                directory
            )

            shared_task_root = (
                base
                / "shared"
            )

            task_dir = _write_task(
                root=shared_task_root,
                output_format="csv",
            )

            first_output = (
                base
                / "output_first"
            )

            second_output = (
                base
                / "output_second"
            )

            seed = 987654

            first = solve_task(
                task_dir=task_dir,
                final_output_dir=(
                    first_output
                ),
                work_root=(
                    base
                    / "work_first"
                ),
                front_half_dependencies=(
                    _dependencies()
                ),
                generator=(
                    _good_generator(
                        "csv"
                    )
                ),
                repairer=(
                    _failing_repairer()
                ),
                candidate_count=3,
                execution_timeout_seconds=30.0,
                audit_timeout_seconds=30.0,
                environ={
                    "QFBENCH_SEED": (
                        str(seed)
                    ),
                },
            )

            second = solve_task(
                task_dir=task_dir,
                final_output_dir=(
                    second_output
                ),
                work_root=(
                    base
                    / "work_second"
                ),
                front_half_dependencies=(
                    _dependencies()
                ),
                generator=(
                    _good_generator(
                        "csv"
                    )
                ),
                repairer=(
                    _failing_repairer()
                ),
                candidate_count=3,
                execution_timeout_seconds=30.0,
                audit_timeout_seconds=30.0,
                environ={
                    "QFBENCH_SEED": (
                        str(seed)
                    ),
                },
            )

            self.assertEqual(
                first.run_id,
                second.run_id,
            )

            first_seeds = [
                item.candidate_seed
                for item
                in first.production.items
            ]

            second_seeds = [
                item.candidate_seed
                for item
                in second.production.items
            ]

            self.assertEqual(
                first_seeds,
                second_seeds,
            )

            self.assertEqual(
                first.selection
                .selected_candidate_id,
                second.selection
                .selected_candidate_id,
            )

            first_frame = (
                pd.read_csv(
                    first_output
                    / "results.csv"
                )
            )

            second_frame = (
                pd.read_csv(
                    second_output
                    / "results.csv"
                )
            )

            pd.testing.assert_frame_equal(
                first_frame,
                second_frame,
            )

    def test_run_logs_preserve_pass_telemetry(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            (
                task_dir,
                final_output_dir,
                result,
            ) = _solve(
                root=root,
                output_format="csv",
                generator=(
                    _partial_failure_generator(
                        "csv"
                    )
                ),
                repairer=(
                    _failing_repairer()
                ),
                candidate_count=3,
            )

            events_path = (
                result.run_dir
                / "events.jsonl"
            )

            events = []

            for line in (
                events_path
                .read_text(
                    encoding="utf-8"
                )
                .splitlines()
            ):
                events.append(
                    json.loads(
                        line
                    )
                )

            event_types = [
                event[
                    "event_type"
                ]
                for event in events
            ]

            self.assertIn(
                "run_started",
                event_types,
            )

            self.assertIn(
                "selection_completed",
                event_types,
            )

            self.assertIn(
                "clean_room_audit_completed",
                event_types,
            )

            self.assertIn(
                "publish_completed",
                event_types,
            )

            self.assertIn(
                "run_completed",
                event_types,
            )

            completed = [
                event
                for event in events
                if (
                    event[
                        "event_type"
                    ]
                    == "run_completed"
                )
            ][-1]

            self.assertFalse(
                completed[
                    "details"
                ][
                    "pass_at_1"
                ]
            )

            self.assertTrue(
                completed[
                    "details"
                ][
                    "any_of_three"
                ]
            )

            checkpoint = json.loads(
                (
                    result.run_dir
                    / "checkpoint.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                checkpoint[
                    "extra"
                ][
                    "phase"
                ],
                "completed",
            )

            self.assertEqual(
                checkpoint[
                    "state"
                ][
                    "status"
                ],
                "succeeded",
            )

            self.assertEqual(
                checkpoint[
                    "selection"
                ][
                    "first_candidate_valid"
                ],
                False,
            )

            self.assertEqual(
                checkpoint[
                    "selection"
                ][
                    "any_of_three_valid"
                ],
                True,
            )


if __name__ == "__main__":
    unittest.main()