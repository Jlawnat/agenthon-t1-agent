from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from agent.candidate_contract import (
    CandidateRecord,
)
from agent.candidate_evaluator import (
    evaluate_candidate_attempt,
    evaluate_collected_attempt,
)
from agent.evidence_collector import (
    collect_execution_evidence,
)
from agent.executor import (
    ExecutionResult,
)
from agent.schema_expectations import (
    SchemaExpectations,
)


def _execution() -> ExecutionResult:
    return ExecutionResult(
        command=["python", "solver.py"],
        return_code=0,
        stdout="",
        stderr="",
        runtime_seconds=0.01,
        timed_out=False,
    )


def _candidate(
    candidate_id: int,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        approach_name="test",
        strategy={},
    )


def _schema() -> SchemaExpectations:
    return SchemaExpectations(
        expected_rows=1,
        id_column="id",
        expected_ids=[1],
        require_id_order=True,
        expected_dtypes={
            "id": "int64",
            "value": "float64",
        },
    )


class CandidateEvaluatorSplitTests(
    unittest.TestCase
):

    def test_split_path_does_not_duplicate_execution_evidence(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            output_dir = (
                root / "output"
            )

            output_dir.mkdir()

            pd.DataFrame(
                {
                    "id": [1],
                    "value": [1.5],
                }
            ).to_csv(
                output_dir
                / "results.csv",
                index=False,
            )

            attempt = (
                collect_execution_evidence(
                    execution=_execution(),
                    output_dir=output_dir,
                    required_output_paths=[
                        "/output/results.csv",
                    ],
                    attempt_number=1,
                    solver_path=(
                        root / "solver.py"
                    ),
                )
            )

            execution_evidence_before = [
                item
                for item
                in attempt.structural_evidence
                if item.name == "execution"
            ]

            self.assertEqual(
                len(
                    execution_evidence_before
                ),
                1,
            )

            candidate = _candidate(1)

            result = (
                evaluate_collected_attempt(
                    candidate=candidate,
                    attempt=attempt,
                    output_dir=output_dir,
                    required_output_paths=[
                        "/output/results.csv",
                    ],
                    required_columns=[
                        "id",
                        "value",
                    ],
                    schema_expectations=(
                        _schema()
                    ),
                )
            )

            self.assertIs(
                result,
                attempt,
            )

            execution_evidence_after = [
                item
                for item
                in attempt.structural_evidence
                if item.name == "execution"
            ]

            self.assertEqual(
                len(
                    execution_evidence_after
                ),
                1,
            )

            self.assertIs(
                candidate.latest_attempt(),
                attempt,
            )

            self.assertEqual(
                candidate.current_status,
                "structurally_valid",
            )

    def test_legacy_wrapper_still_behaves_identically(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            output_dir = (
                root / "output"
            )

            output_dir.mkdir()

            pd.DataFrame(
                {
                    "id": [1],
                    "value": [1.5],
                }
            ).to_csv(
                output_dir
                / "results.csv",
                index=False,
            )

            candidate = _candidate(2)

            attempt = (
                evaluate_candidate_attempt(
                    candidate=candidate,
                    execution=_execution(),
                    output_dir=output_dir,
                    solver_path=(
                        root / "solver.py"
                    ),
                    required_output_paths=[
                        "/output/results.csv",
                    ],
                    required_columns=[
                        "id",
                        "value",
                    ],
                    schema_expectations=(
                        _schema()
                    ),
                    attempt_number=1,
                )
            )

            execution_evidence = [
                item
                for item
                in attempt.structural_evidence
                if item.name == "execution"
            ]

            self.assertEqual(
                len(execution_evidence),
                1,
            )

            self.assertEqual(
                candidate.current_status,
                "structurally_valid",
            )

            self.assertIs(
                candidate.latest_attempt(),
                attempt,
            )


if __name__ == "__main__":
    unittest.main()