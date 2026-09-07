from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import pandas as pd

from agent.budget import RepairBudget
from agent.candidate_contract import CandidateRecord
from agent.candidate_evaluator import evaluate_candidate_attempt
from agent.schema_expectations import infer_schema_expectations
from agent.spec_compiler import compile_specification
from agent.specification import load_task_specification
from agent.targeted_repair import (
    UnsafeRepairEvidenceError,
    build_repair_brief,
)


def _write_task(
    root: Path,
    *,
    category: str,
    output_columns: list[tuple[str, str]],
    input_frame: pd.DataFrame,
) -> tuple[Path, Path]:
    task_dir = root / "task"
    data_dir = task_dir / "environment" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    schema_rows = "\n".join(
        f"| `{column}` | `{dtype}` |"
        for column, dtype in output_columns
    )

    instruction = (
        "Produce the required result.\n\n"
        "### /output/results.csv\n\n"
        "| Column | Type |\n"
        "|---|---|\n"
        f"{schema_rows}\n"
    )

    (task_dir / "instruction.md").write_text(
        instruction,
        encoding="utf-8",
    )

    (task_dir / "card.toml").write_text(
        f'[metadata]\ncategory = "{category}"\n',
        encoding="utf-8",
    )

    input_frame.to_csv(
        data_dir / "input.csv",
        index=False,
    )

    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    return task_dir, output_dir


def _execution_ok():
    return SimpleNamespace(
        return_code=0,
        timed_out=False,
        runtime_seconds=0.01,
        stdout="",
        stderr="",
    )


def _candidate() -> CandidateRecord:
    return CandidateRecord(
        candidate_id=1,
        approach_name="integration-test",
        strategy={},
    )


def _evaluate(
    *,
    task_dir: Path,
    output_dir: Path,
    candidate: CandidateRecord,
):
    task_spec = load_task_specification(task_dir)
    compiled = compile_specification(task_spec)

    schema = infer_schema_expectations(
        task_dir=task_dir,
        candidate_schema_fields=(
            task_spec.candidate_schema_fields
        ),
        required_dtypes=compiled.required_dtypes,
    )

    attempt = evaluate_candidate_attempt(
        candidate=candidate,
        execution=_execution_ok(),
        output_dir=output_dir,
        solver_path=output_dir / "solver.py",
        required_output_paths=(
            task_spec.required_output_paths
        ),
        required_columns=(
            compiled.required_columns
        ),
        schema_expectations=schema,
        attempt_number=1,
        compiled_specification=compiled,
    )

    return task_spec, compiled, schema, attempt


class Phase311IntegrationAcceptanceTests(
    unittest.TestCase
):
    def test_explicit_dtype_failure_reaches_dtype_repair(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir, output_dir = _write_task(
                root,
                category="derivatives-pricing",
                output_columns=[
                    ("option_id", "int64"),
                    ("price", "float64"),
                ],
                input_frame=pd.DataFrame(
                    {
                        "option_id": [1],
                    }
                ),
            )

            pd.DataFrame(
                {
                    "option_id": [1],
                    "price": ["not-a-number"],
                }
            ).to_csv(
                output_dir / "results.csv",
                index=False,
            )

            candidate = _candidate()

            _, compiled, schema, attempt = _evaluate(
                task_dir=task_dir,
                output_dir=output_dir,
                candidate=candidate,
            )

            self.assertEqual(
                compiled.required_dtypes["price"],
                "float64",
            )
            self.assertEqual(
                schema.expected_dtypes["price"],
                "float64",
            )

            failures = {
                item.name: item
                for item in attempt.structural_evidence
                if item.status == "fail"
            }

            self.assertIn(
                "dtype_mismatch",
                failures,
            )
            self.assertEqual(
                candidate.current_status,
                "failed",
            )
            self.assertIs(
                candidate.latest_attempt(),
                attempt,
            )

            budget = RepairBudget(max_attempts=1)

            brief = build_repair_brief(
                attempt=attempt,
                budget=budget,
            )

            self.assertIsNotNone(brief)
            self.assertEqual(
                brief.failure_label,
                "dtype",
            )
            self.assertEqual(
                brief.strategy,
                "repair_output_dtype",
            )
            self.assertEqual(
                budget.attempts_used,
                1,
            )

    def test_quant_failure_reaches_numerical_repair(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir, output_dir = _write_task(
                root,
                category="credit",
                output_columns=[
                    ("option_id", "int64"),
                    (
                        "default_probability",
                        "float64",
                    ),
                ],
                input_frame=pd.DataFrame(
                    {
                        "option_id": [1, 2],
                    }
                ),
            )

            pd.DataFrame(
                {
                    "option_id": [1, 2],
                    "default_probability": [
                        0.20,
                        1.20,
                    ],
                }
            ).to_csv(
                output_dir / "results.csv",
                index=False,
            )

            candidate = _candidate()

            _, _, _, attempt = _evaluate(
                task_dir=task_dir,
                output_dir=output_dir,
                candidate=candidate,
            )

            quant_failures = [
                item
                for item in attempt.quant_evidence
                if (
                    item.status == "fail"
                    and item.severity == "hard_fail"
                )
            ]

            self.assertTrue(quant_failures)
            self.assertEqual(
                candidate.current_status,
                "failed",
            )

            budget = RepairBudget(max_attempts=1)

            brief = build_repair_brief(
                attempt=attempt,
                budget=budget,
            )

            self.assertIsNotNone(brief)
            self.assertEqual(
                brief.failure_label,
                "numerical_reconciliation",
            )
            self.assertEqual(
                brief.strategy,
                "repair_numerical_reconciliation",
            )
            self.assertTrue(
                brief.relevant_invariants
            )

    def test_private_runtime_evidence_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir, output_dir = _write_task(
                root,
                category="credit",
                output_columns=[
                    ("option_id", "int64"),
                ],
                input_frame=pd.DataFrame(
                    {
                        "option_id": [1],
                    }
                ),
            )

            task_spec = load_task_specification(
                task_dir
            )
            compiled = compile_specification(
                task_spec
            )

            schema = infer_schema_expectations(
                task_dir=task_dir,
                candidate_schema_fields=(
                    task_spec.candidate_schema_fields
                ),
                required_dtypes=(
                    compiled.required_dtypes
                ),
            )

            execution = SimpleNamespace(
                return_code=1,
                timed_out=False,
                runtime_seconds=0.01,
                stdout="",
                stderr=(
                    "RuntimeError: oracle reference "
                    "output was accessed"
                ),
            )

            candidate = _candidate()

            attempt = evaluate_candidate_attempt(
                candidate=candidate,
                execution=execution,
                output_dir=output_dir,
                solver_path=(
                    output_dir / "solver.py"
                ),
                required_output_paths=(
                    task_spec.required_output_paths
                ),
                required_columns=(
                    compiled.required_columns
                ),
                schema_expectations=schema,
                attempt_number=1,
                compiled_specification=compiled,
            )

            budget = RepairBudget(max_attempts=1)

            with self.assertRaises(
                UnsafeRepairEvidenceError
            ):
                build_repair_brief(
                    attempt=attempt,
                    budget=budget,
                )

            self.assertEqual(
                budget.attempts_used,
                0,
            )


if __name__ == "__main__":
    unittest.main()
