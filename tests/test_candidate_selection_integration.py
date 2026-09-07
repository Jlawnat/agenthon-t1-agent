from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import pandas as pd

from agent.candidate_contract import CandidateRecord
from agent.candidate_evaluator import (
    evaluate_candidate_attempt,
)
from agent.candidate_selector import (
    select_best_candidate,
)
from agent.schema_expectations import (
    SchemaExpectations,
)


def _execution(
    *,
    return_code: int = 0,
    timed_out: bool = False,
    runtime_seconds: float = 1.0,
    stderr: str = "",
):
    return SimpleNamespace(
        return_code=return_code,
        timed_out=timed_out,
        runtime_seconds=runtime_seconds,
        stdout="",
        stderr=stderr,
    )


def _candidate(
    candidate_id: int,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        approach_name=(
            f"integration-{candidate_id}"
        ),
        strategy={},
    )


def _compiled_credit_spec():
    return SimpleNamespace(
        category="credit",
        review_packs=(),
        invariants=(),
        units={},
        conventions=(),
    )


def _schema() -> SchemaExpectations:
    return SchemaExpectations(
        expected_rows=2,
        id_column="option_id",
        expected_ids=[1, 2],
        require_id_order=True,
        expected_dtypes={
            "option_id": "int64",
            "default_probability": "float64",
        },
    )


def _evaluate(
    *,
    candidate: CandidateRecord,
    output_dir: Path,
    execution,
):
    return evaluate_candidate_attempt(
        candidate=candidate,
        execution=execution,
        output_dir=output_dir,
        solver_path=(
            output_dir / "solver.py"
        ),
        required_output_paths=[
            "/output/results.csv"
        ],
        required_columns=[
            "option_id",
            "default_probability",
        ],
        schema_expectations=_schema(),
        attempt_number=1,
        compiled_specification=(
            _compiled_credit_spec()
        ),
    )


class CandidateSelectionIntegrationTests(
    unittest.TestCase
):
    def test_real_evaluator_evidence_selects_clean_candidate(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            crash_dir = root / "crash"
            bad_quant_dir = root / "bad_quant"
            clean_dir = root / "clean"

            crash_dir.mkdir()
            bad_quant_dir.mkdir()
            clean_dir.mkdir()

            candidate_1 = _candidate(1)
            candidate_2 = _candidate(2)
            candidate_3 = _candidate(3)

            _evaluate(
                candidate=candidate_1,
                output_dir=crash_dir,
                execution=_execution(
                    return_code=1,
                    runtime_seconds=0.1,
                    stderr=(
                        "RuntimeError: candidate failed"
                    ),
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
                bad_quant_dir / "results.csv",
                index=False,
            )

            _evaluate(
                candidate=candidate_2,
                output_dir=bad_quant_dir,
                execution=_execution(
                    return_code=0,
                    runtime_seconds=0.5,
                ),
            )

            pd.DataFrame(
                {
                    "option_id": [1, 2],
                    "default_probability": [
                        0.20,
                        0.40,
                    ],
                }
            ).to_csv(
                clean_dir / "results.csv",
                index=False,
            )

            _evaluate(
                candidate=candidate_3,
                output_dir=clean_dir,
                execution=_execution(
                    return_code=0,
                    runtime_seconds=2.0,
                ),
            )

            result = select_best_candidate(
                [
                    candidate_1,
                    candidate_2,
                    candidate_3,
                ]
            )

            self.assertEqual(
                result.selected_candidate_id,
                3,
            )

            self.assertEqual(
                result.ranked_candidate_ids,
                (3, 2, 1),
            )

            self.assertFalse(
                result.first_candidate_valid
            )

            self.assertTrue(
                result.any_of_three_valid
            )

            self.assertFalse(
                candidate_1.selected
            )

            self.assertFalse(
                candidate_2.selected
            )

            self.assertTrue(
                candidate_3.selected
            )

            self.assertEqual(
                candidate_1.current_status,
                "failed",
            )

            self.assertEqual(
                candidate_2.current_status,
                "failed",
            )

            self.assertEqual(
                candidate_3.current_status,
                "validated",
            )


if __name__ == "__main__":
    unittest.main()