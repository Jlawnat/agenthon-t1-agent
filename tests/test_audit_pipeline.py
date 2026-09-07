from __future__ import annotations

import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from agent.audit_pipeline import (
    CleanRoomAuditError,
    run_clean_room_audit_stage,
)
from agent.candidate_contract import (
    CandidateAttempt,
    CandidateRecord,
)
from agent.candidate_production import (
    CandidateProductionItem,
)
from agent.candidate_selector import (
    SelectionResult,
)
from agent.candidate_workspace import (
    CandidateWorkspace,
)
from agent.compiled_specification import (
    CompiledSpecification,
)
from agent.executor import (
    ExecutionResult,
)
from agent.orchestrator_state import (
    OrchestratorState,
    PipelineStage,
    RunStatus,
)
from agent.run_budget import (
    RunBudget,
)
from agent.run_context import (
    RunContext,
)
from agent.schema_expectations import (
    SchemaExpectations,
)
from agent.selection_pipeline import (
    SelectionPipelineResult,
)


def _write_task(
    root: Path,
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

    input_path = (
        data_dir
        / "input.csv"
    )

    with input_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "input_value",
            ],
        )

        writer.writeheader()

        writer.writerows(
            [
                {
                    "id": 1,
                    "input_value": 10.0,
                },
                {
                    "id": 2,
                    "input_value": 20.0,
                },
            ]
        )

    return task_dir


def _compiled_specification(
) -> CompiledSpecification:
    return CompiledSpecification(
        task_id="audit-pipeline-test",
        category="risk-management",
        difficulty="easy",
        deliverables=[
            {
                "path": (
                    "/output/results.csv"
                ),
                "format": "csv",
            }
        ],
        required_columns=[
            "id",
            "value",
        ],
        required_dtypes={
            "id": "int64",
            "value": "float64",
        },
        required_row_rules=[],
        ordering_rules=[],
        units={},
        conventions=[],
        edge_cases=[],
        invariants=[],
        review_packs=[
            "risk-management",
        ],
        numerical_risks=[],
        assumptions=[],
        unresolved_questions=[],
    )


def _schema_expectations(
) -> SchemaExpectations:
    return SchemaExpectations(
        expected_rows=2,
        id_column="id",
        expected_ids=[
            1,
            2,
        ],
        require_id_order=True,
        expected_dtypes={
            "id": "int64",
            "value": "float64",
        },
    )


def _run_context(
) -> RunContext:
    return RunContext(
        task_id="audit-pipeline-test",
        card_timeout_seconds=600.0,
        qfbench_seed=42,
        budget=RunBudget(
            total_wall_seconds=600.0,
            safety_margin_seconds=5.0,
        ),
    )


def _state_at_clean_room_audit(
) -> OrchestratorState:
    state = OrchestratorState(
        run_id="audit-run"
    )

    for stage in (
        PipelineStage.SNAPSHOT,
        PipelineStage.SPECIFICATION,
        PipelineStage.COMPILE,
        PipelineStage.PLAN,
        PipelineStage.GENERATE,
        PipelineStage.VALIDATE_CODE,
        PipelineStage.ISOLATED_EXECUTE,
        PipelineStage.COLLECT_EVIDENCE,
        PipelineStage.EVALUATE,
        PipelineStage.TARGETED_REPAIR,
        PipelineStage.SELECT,
    ):
        state.start_stage(
            stage
        )

        state.complete_stage(
            stage
        )

    return state


def _execution(
) -> ExecutionResult:
    return ExecutionResult(
        command=[
            "python",
            "solver.py",
        ],
        return_code=0,
        stdout="",
        stderr="",
        runtime_seconds=0.01,
        timed_out=False,
    )


def _selection(
    *,
    root: Path,
    task_dir: Path,
    solver_code: str,
    candidate_seed: int = 1001,
) -> SelectionPipelineResult:
    workspace = (
        CandidateWorkspace.create(
            base_dir=(
                root
                / "selected_workspace"
            ),
            candidate_id=1,
            task_dir=task_dir,
        )
    )

    solver_path = (
        workspace.source_dir
        / "solver.py"
    )

    solver_path.write_text(
        solver_code,
        encoding="utf-8",
    )

    candidate = CandidateRecord(
        candidate_id=1,
        approach_name=(
            "selected-candidate"
        ),
        strategy={},
    )

    candidate.add_attempt(
        CandidateAttempt(
            attempt_number=1,
            return_code=0,
            runtime_seconds=0.01,
        )
    )

    candidate.current_status = (
        "validated"
    )

    candidate.selected = True

    item = CandidateProductionItem(
        candidate=candidate,
        candidate_seed=(
            candidate_seed
        ),
        raw_code=solver_code,
        validated_code=(
            solver_code
        ),
        workspace=workspace,
        solver_path=solver_path,
        execution=_execution(),
    )

    selection = SelectionResult(
        selected_candidate_id=1,
        ranked_candidate_ids=(1,),
        first_candidate_valid=True,
        any_of_three_valid=True,
        scorecards=(),
    )

    return SelectionPipelineResult(
        selection=selection,
        selected_item=item,
    )


def _good_code() -> str:
    return """
import csv
from pathlib import Path

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
                "value": float(
                    row[
                        "input_value"
                    ]
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
    writer.writerows(
        rows
    )
""".strip()


def _seed_safe_code() -> str:
    return """
import csv
import os
from pathlib import Path

if "QFBENCH_SEED" not in os.environ:
    raise RuntimeError(
        "QFBENCH_SEED missing"
    )

for name in (
    "MODEL_ENDPOINT",
    "MODEL_NAME",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
):
    if name in os.environ:
        raise RuntimeError(
            "Sensitive environment leaked"
        )

seed = float(
    os.environ[
        "QFBENCH_SEED"
    ]
)

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
    writer.writerows(
        rows
    )
""".strip()


def _unexpected_file_code() -> str:
    return (
        _good_code()
        + """

from pathlib import Path

Path(
    "output/extra.txt"
).write_text(
    "unexpected",
    encoding="utf-8",
)
"""
    )


def _reward_file_code() -> str:
    return (
        _good_code()
        + """

from pathlib import Path

Path(
    "output/reward.json"
).write_text(
    "{}",
    encoding="utf-8",
)
"""
    )


def _runtime_failure_code(
) -> str:
    return """
raise RuntimeError(
    "clean room failure"
)
""".strip()


class AuditPipelineTests(
    unittest.TestCase
):

    def test_successful_clean_room_audit_advances_to_publish(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            task_dir = (
                _write_task(
                    root
                )
            )

            selection = (
                _selection(
                    root=root,
                    task_dir=task_dir,
                    solver_code=(
                        _good_code()
                    ),
                )
            )

            state = (
                _state_at_clean_room_audit()
            )

            result = (
                run_clean_room_audit_stage(
                    task_dir=task_dir,
                    audit_workspace_base_dir=(
                        root
                        / "audit_workspace"
                    ),
                    state=state,
                    run_context=(
                        _run_context()
                    ),
                    selection=selection,
                    compiled_specification=(
                        _compiled_specification()
                    ),
                    schema_expectations=(
                        _schema_expectations()
                    ),
                )
            )

            self.assertTrue(
                result.audit.passed
            )

            self.assertEqual(
                result.selected_candidate_id,
                1,
            )

            self.assertTrue(
                (
                    result
                    .audited_output_dir
                    / "results.csv"
                ).exists()
            )

            self.assertEqual(
                state.expected_stage(),
                PipelineStage.PUBLISH,
            )

    def test_clean_room_preserves_original_selected_workspace(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            task_dir = (
                _write_task(
                    root
                )
            )

            selection = (
                _selection(
                    root=root,
                    task_dir=task_dir,
                    solver_code=(
                        _good_code()
                    ),
                )
            )

            original_workspace = (
                selection
                .selected_item
                .workspace
            )

            self.assertIsNotNone(
                original_workspace
            )

            original_solver = (
                selection
                .selected_item
                .solver_path
            )

            self.assertIsNotNone(
                original_solver
            )

            original_solver_text = (
                original_solver.read_text(
                    encoding="utf-8"
                )
            )

            original_output = (
                original_workspace
                .output_dir
                / "results.csv"
            )

            original_output.write_text(
                "ORIGINAL WORKSPACE\n",
                encoding="utf-8",
            )

            result = (
                run_clean_room_audit_stage(
                    task_dir=task_dir,
                    audit_workspace_base_dir=(
                        root
                        / "audit_workspace"
                    ),
                    state=(
                        _state_at_clean_room_audit()
                    ),
                    run_context=(
                        _run_context()
                    ),
                    selection=selection,
                    compiled_specification=(
                        _compiled_specification()
                    ),
                    schema_expectations=(
                        _schema_expectations()
                    ),
                )
            )

            self.assertTrue(
                original_workspace
                .root_dir
                .exists()
            )

            self.assertEqual(
                original_solver.read_text(
                    encoding="utf-8"
                ),
                original_solver_text,
            )

            self.assertEqual(
                original_output.read_text(
                    encoding="utf-8"
                ),
                "ORIGINAL WORKSPACE\n",
            )

            self.assertNotEqual(
                result.workspace.root_dir,
                original_workspace.root_dir,
            )

    def test_clean_room_receives_seed_but_not_sensitive_environment(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            task_dir = (
                _write_task(
                    root
                )
            )

            candidate_seed = (
                987654
            )

            selection = (
                _selection(
                    root=root,
                    task_dir=task_dir,
                    solver_code=(
                        _seed_safe_code()
                    ),
                    candidate_seed=(
                        candidate_seed
                    ),
                )
            )

            sensitive_parent = {
                "MODEL_ENDPOINT": (
                    "https://example.invalid"
                ),
                "MODEL_NAME": (
                    "secret-model"
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
                    run_clean_room_audit_stage(
                        task_dir=task_dir,
                        audit_workspace_base_dir=(
                            root
                            / "audit_workspace"
                        ),
                        state=(
                            _state_at_clean_room_audit()
                        ),
                        run_context=(
                            _run_context()
                        ),
                        selection=selection,
                        compiled_specification=(
                            _compiled_specification()
                        ),
                        schema_expectations=(
                            _schema_expectations()
                        ),
                    )
                )

            self.assertTrue(
                result.audit.passed
            )

            output_path = (
                result
                .audited_output_dir
                / "results.csv"
            )

            with output_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(
                    csv.DictReader(
                        handle
                    )
                )

            self.assertEqual(
                len(rows),
                2,
            )

            for row in rows:
                self.assertEqual(
                    float(
                        row["value"]
                    ),
                    float(
                        candidate_seed
                    ),
                )

    def test_unexpected_output_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            task_dir = (
                _write_task(
                    root
                )
            )

            selection = (
                _selection(
                    root=root,
                    task_dir=task_dir,
                    solver_code=(
                        _unexpected_file_code()
                    ),
                )
            )

            state = (
                _state_at_clean_room_audit()
            )

            with self.assertRaises(
                CleanRoomAuditError
            ):
                run_clean_room_audit_stage(
                    task_dir=task_dir,
                    audit_workspace_base_dir=(
                        root
                        / "audit_workspace"
                    ),
                    state=state,
                    run_context=(
                        _run_context()
                    ),
                    selection=selection,
                    compiled_specification=(
                        _compiled_specification()
                    ),
                    schema_expectations=(
                        _schema_expectations()
                    ),
                )

            self.assertEqual(
                state.status,
                RunStatus.FAILED,
            )

            self.assertIn(
                "stale or unrequested",
                (
                    state.failure_reason
                    or ""
                ).lower(),
            )

    def test_reward_artifact_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            task_dir = (
                _write_task(
                    root
                )
            )

            selection = (
                _selection(
                    root=root,
                    task_dir=task_dir,
                    solver_code=(
                        _reward_file_code()
                    ),
                )
            )

            state = (
                _state_at_clean_room_audit()
            )

            with self.assertRaises(
                CleanRoomAuditError
            ):
                run_clean_room_audit_stage(
                    task_dir=task_dir,
                    audit_workspace_base_dir=(
                        root
                        / "audit_workspace"
                    ),
                    state=state,
                    run_context=(
                        _run_context()
                    ),
                    selection=selection,
                    compiled_specification=(
                        _compiled_specification()
                    ),
                    schema_expectations=(
                        _schema_expectations()
                    ),
                )

            self.assertEqual(
                state.status,
                RunStatus.FAILED,
            )

            self.assertIn(
                "forbidden",
                (
                    state.failure_reason
                    or ""
                ).lower(),
            )

    def test_clean_room_execution_failure_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            task_dir = (
                _write_task(
                    root
                )
            )

            selection = (
                _selection(
                    root=root,
                    task_dir=task_dir,
                    solver_code=(
                        _runtime_failure_code()
                    ),
                )
            )

            state = (
                _state_at_clean_room_audit()
            )

            with self.assertRaises(
                CleanRoomAuditError
            ):
                run_clean_room_audit_stage(
                    task_dir=task_dir,
                    audit_workspace_base_dir=(
                        root
                        / "audit_workspace"
                    ),
                    state=state,
                    run_context=(
                        _run_context()
                    ),
                    selection=selection,
                    compiled_specification=(
                        _compiled_specification()
                    ),
                    schema_expectations=(
                        _schema_expectations()
                    ),
                )

            self.assertEqual(
                state.status,
                RunStatus.FAILED,
            )

            self.assertIn(
                "return code",
                (
                    state.failure_reason
                    or ""
                ).lower(),
            )


if __name__ == "__main__":
    unittest.main()