from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import pandas as pd

from agent.final_audit import (
    audit_final_output,
)
from agent.schema_expectations import (
    SchemaExpectations,
)


def _compiled_spec(
    *,
    path: str = "/output/results.csv",
    format_name: str = "csv",
):
    return SimpleNamespace(
        task_id="audit-test",
        category="credit",
        difficulty="easy",
        deliverables=[
            {
                "path": path,
                "format": format_name,
            }
        ],
        required_columns=[
            "option_id",
            "default_probability",
        ],
        required_row_rules=[],
        ordering_rules=[],
        units={},
        conventions=[],
        edge_cases=[],
        invariants=[],
        review_packs=[],
        numerical_risks=[],
        assumptions=[],
        unresolved_questions=[],
        required_dtypes={
            "option_id": "int64",
            "default_probability": "float64",
        },
    )


def _schema():
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


def _finding(
    result,
    name: str,
):
    return [
        item
        for item in result.findings
        if item.name == name
    ]


class FinalAuditAcceptanceTests(
    unittest.TestCase
):
    def test_clean_csv_output_passes(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)

            pd.DataFrame(
                {
                    "option_id": [1, 2],
                    "default_probability": [
                        0.20,
                        0.40,
                    ],
                }
            ).to_csv(
                output_dir / "results.csv",
                index=False,
            )

            result = audit_final_output(
                output_dir=output_dir,
                compiled_specification=(
                    _compiled_spec()
                ),
                schema_expectations=_schema(),
            )

            self.assertTrue(
                result.passed
            )

            self.assertEqual(
                result.required_files,
                ("results.csv",),
            )

            self.assertEqual(
                result.produced_files,
                ("results.csv",),
            )

    def test_stale_unrequested_file_fails(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)

            pd.DataFrame(
                {
                    "option_id": [1, 2],
                    "default_probability": [
                        0.20,
                        0.40,
                    ],
                }
            ).to_csv(
                output_dir / "results.csv",
                index=False,
            )

            (
                output_dir
                / "old_debug.txt"
            ).write_text(
                "stale",
                encoding="utf-8",
            )

            result = audit_final_output(
                output_dir=output_dir,
                compiled_specification=(
                    _compiled_spec()
                ),
                schema_expectations=_schema(),
            )

            self.assertFalse(
                result.passed
            )

            findings = _finding(
                result,
                "unexpected_files",
            )

            self.assertEqual(
                findings[0].status,
                "fail",
            )

            self.assertIn(
                "old_debug.txt",
                findings[0].details[
                    "unexpected_files"
                ],
            )

    def test_reward_json_is_forbidden(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)

            pd.DataFrame(
                {
                    "option_id": [1, 2],
                    "default_probability": [
                        0.20,
                        0.40,
                    ],
                }
            ).to_csv(
                output_dir / "results.csv",
                index=False,
            )

            (
                output_dir
                / "reward.json"
            ).write_text(
                "{}",
                encoding="utf-8",
            )

            result = audit_final_output(
                output_dir=output_dir,
                compiled_specification=(
                    _compiled_spec()
                ),
                schema_expectations=_schema(),
            )

            self.assertFalse(
                result.passed
            )

            findings = _finding(
                result,
                "forbidden_artifacts",
            )

            self.assertEqual(
                findings[0].status,
                "fail",
            )

            self.assertIn(
                "reward.json",
                findings[0].details[
                    "forbidden_files"
                ],
            )

    def test_wrong_dtype_fails(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)

            pd.DataFrame(
                {
                    "option_id": [1, 2],
                    "default_probability": [
                        "bad",
                        "also-bad",
                    ],
                }
            ).to_csv(
                output_dir / "results.csv",
                index=False,
            )

            result = audit_final_output(
                output_dir=output_dir,
                compiled_specification=(
                    _compiled_spec()
                ),
                schema_expectations=_schema(),
            )

            self.assertFalse(
                result.passed
            )

            findings = _finding(
                result,
                "structural:dtype_mismatch",
            )

            self.assertTrue(findings)

            self.assertEqual(
                findings[0].status,
                "fail",
            )

    def test_wrong_identifier_order_fails(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)

            pd.DataFrame(
                {
                    "option_id": [2, 1],
                    "default_probability": [
                        0.20,
                        0.40,
                    ],
                }
            ).to_csv(
                output_dir / "results.csv",
                index=False,
            )

            result = audit_final_output(
                output_dir=output_dir,
                compiled_specification=(
                    _compiled_spec()
                ),
                schema_expectations=_schema(),
            )

            self.assertFalse(
                result.passed
            )

            findings = _finding(
                result,
                "structural:identifier_order",
            )

            self.assertTrue(findings)

            self.assertEqual(
                findings[0].status,
                "fail",
            )

    def test_json_output_passes(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)

            payload = [
                {
                    "option_id": 1,
                    "default_probability": 0.20,
                },
                {
                    "option_id": 2,
                    "default_probability": 0.40,
                },
            ]

            (
                output_dir
                / "results.json"
            ).write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            result = audit_final_output(
                output_dir=output_dir,
                compiled_specification=(
                    _compiled_spec(
                        path=(
                            "/output/results.json"
                        ),
                        format_name="json",
                    )
                ),
                schema_expectations=_schema(),
            )

            self.assertTrue(
                result.passed
            )

    def test_jsonl_output_passes(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)

            rows = [
                {
                    "option_id": 1,
                    "default_probability": 0.20,
                },
                {
                    "option_id": 2,
                    "default_probability": 0.40,
                },
            ]

            output_path = (
                output_dir
                / "results.jsonl"
            )

            with output_path.open(
                "w",
                encoding="utf-8",
            ) as handle:
                for row in rows:
                    handle.write(
                        json.dumps(row)
                        + "\n"
                    )

            result = audit_final_output(
                output_dir=output_dir,
                compiled_specification=(
                    _compiled_spec(
                        path=(
                            "/output/results.jsonl"
                        ),
                        format_name="jsonl",
                    )
                ),
                schema_expectations=_schema(),
            )

            self.assertTrue(
                result.passed
            )

    def test_declared_format_mismatch_fails(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)

            pd.DataFrame(
                {
                    "option_id": [1, 2],
                    "default_probability": [
                        0.20,
                        0.40,
                    ],
                }
            ).to_csv(
                output_dir / "results.csv",
                index=False,
            )

            result = audit_final_output(
                output_dir=output_dir,
                compiled_specification=(
                    _compiled_spec(
                        path=(
                            "/output/results.csv"
                        ),
                        format_name="json",
                    )
                ),
                schema_expectations=_schema(),
            )

            self.assertFalse(
                result.passed
            )

            findings = _finding(
                result,
                "output_format",
            )

            self.assertEqual(
                findings[0].status,
                "fail",
            )

    def test_missing_required_output_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)

            result = audit_final_output(
                output_dir=output_dir,
                compiled_specification=(
                    _compiled_spec()
                ),
                schema_expectations=_schema(),
            )

            self.assertFalse(
                result.passed
            )

            findings = _finding(
                result,
                "required_deliverables",
            )

            self.assertEqual(
                findings[0].status,
                "fail",
            )

            self.assertIn(
                "results.csv",
                findings[0].details[
                    "missing_files"
                ],
            )


if __name__ == "__main__":
    unittest.main()