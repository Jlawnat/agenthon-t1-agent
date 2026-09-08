from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from agent.quant_invariants import (
    QuantInvariantConfig,
    evaluate_quant_invariants,
)
from agent.spec_compiler import compile_specification
from agent.specification import TaskSpecification


def _by_name(evidence):
    return {
        item.name: item
        for item in evidence
    }


class WeightReconciliationTests(unittest.TestCase):
    def test_compiler_captures_explicit_weight_sum_convention(
        self,
    ) -> None:
        spec = TaskSpecification(
            task_id="fixed-income-weight-test",
            category="fixed-income",
            instruction_text=(
                "Construct the hedge portfolio. "
                "Hedge weights must sum to one."
            ),
            required_output_paths=[
                "/output/hedge.json",
            ],
            candidate_schema_fields=[],
            required_output_columns=[],
            required_output_dtypes={},
            runtime_seconds=300,
            network_mode="restricted",
            raw_card={},
        )

        compiled = compile_specification(spec)

        self.assertIn(
            "weights sum to one",
            compiled.conventions,
        )

    def test_nested_weight_map_violation_is_hard_failure(
        self,
    ) -> None:
        payload = {
            "valuation_date": "2023-06-30",
            "weights": {
                "BUL2": 0.30,
                "BUL4": 0.30,
                "BUL7": 0.30,
            },
        }

        with TemporaryDirectory() as directory:
            output_path = (
                Path(directory)
                / "hedge_t0.json"
            )

            output_path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            evidence = evaluate_quant_invariants(
                output_path=output_path,
                config=QuantInvariantConfig(
                    category="fixed-income",
                    conventions=(
                        "weights sum to one",
                    ),
                ),
            )

        result = _by_name(
            evidence
        )["weight_reconciliation"]

        self.assertEqual(
            result.status,
            "fail",
        )
        self.assertEqual(
            result.severity,
            "hard_fail",
        )
        self.assertAlmostEqual(
            result.details["actual_sum"],
            0.90,
        )
        self.assertEqual(
            result.details["target"],
            1.0,
        )

    def test_nested_weight_map_passes_at_one(
        self,
    ) -> None:
        payload = {
            "weights": {
                "BUL2": 0.40,
                "BUL4": 0.35,
                "BUL7": 0.25,
            },
        }

        with TemporaryDirectory() as directory:
            output_path = (
                Path(directory)
                / "hedge_t0.json"
            )

            output_path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            evidence = evaluate_quant_invariants(
                output_path=output_path,
                config=QuantInvariantConfig(
                    category="fixed-income",
                    conventions=(
                        "weights sum to one",
                    ),
                ),
            )

        result = _by_name(
            evidence
        )["weight_reconciliation"]

        self.assertEqual(
            result.status,
            "pass",
        )


if __name__ == "__main__":
    unittest.main()
