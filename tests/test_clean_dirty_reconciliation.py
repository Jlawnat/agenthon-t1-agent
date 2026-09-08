from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from agent.quant_invariants import (
    QuantInvariantConfig,
    evaluate_quant_invariants,
)


def _by_name(evidence):
    return {
        item.name: item
        for item in evidence
    }


class CleanDirtyReconciliationTests(
    unittest.TestCase
):
    def _evaluate(
        self,
        payload,
    ):
        with TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "instrument_analytics_t0.json"
            )

            path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            return evaluate_quant_invariants(
                output_path=path,
                config=QuantInvariantConfig(
                    category="fixed-income",
                    review_packs=(
                        "fixed-income",
                    ),
                ),
            )

    def test_clean_dirty_identity_passes(
        self,
    ) -> None:
        evidence = self._evaluate(
            [
                {
                    "id": "BUL2",
                    "clean_price": 99.25,
                    "accrued_interest": 0.75,
                    "dirty_price": 100.00,
                },
                {
                    "id": "BUL4",
                    "clean_price": 101.10,
                    "accrued_interest": 0.40,
                    "dirty_price": 101.50,
                },
            ]
        )

        result = _by_name(
            evidence
        )["clean_dirty_reconciliation"]

        self.assertEqual(
            result.status,
            "pass",
        )

    def test_clean_dirty_identity_violation_is_hard_failure(
        self,
    ) -> None:
        evidence = self._evaluate(
            [
                {
                    "id": "BUL2",
                    "clean_price": 99.25,
                    "accrued_interest": 0.75,
                    "dirty_price": 99.50,
                }
            ]
        )

        result = _by_name(
            evidence
        )["clean_dirty_reconciliation"]

        self.assertEqual(
            result.status,
            "fail",
        )
        self.assertEqual(
            result.severity,
            "hard_fail",
        )
        self.assertEqual(
            result.details[
                "violation_count"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
