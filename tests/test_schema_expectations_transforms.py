from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from agent.schema_expectations import (
    infer_schema_expectations,
)


class SchemaExpectationTransformTests(unittest.TestCase):
    def test_single_input_without_shared_id_does_not_infer_row_count(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task"
            data = task / "environment" / "data"
            data.mkdir(parents=True)

            pd.DataFrame(
                {
                    "identifier": [1, 2, 3, 4],
                    "date": [
                        "2026-01-31",
                        "2026-01-31",
                        "2026-02-28",
                        "2026-02-28",
                    ],
                    "signal": [0.1, 0.2, 0.3, 0.4],
                }
            ).to_csv(
                data / "panel.csv",
                index=False,
            )

            result = infer_schema_expectations(
                task_dir=task,
                candidate_schema_fields=[
                    "date",
                    "net_return",
                ],
            )

            self.assertIsNone(
                result.expected_rows
            )
            self.assertIsNone(
                result.id_column
            )
            self.assertIsNone(
                result.expected_ids
            )
            self.assertFalse(
                result.require_id_order
            )

    def test_shared_explicit_id_keeps_row_and_order_inference(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task"
            data = task / "environment" / "data"
            data.mkdir(parents=True)

            pd.DataFrame(
                {
                    "option_id": [10, 20, 30],
                    "spot": [100.0, 101.0, 102.0],
                }
            ).to_csv(
                data / "options.csv",
                index=False,
            )

            result = infer_schema_expectations(
                task_dir=task,
                candidate_schema_fields=[
                    "option_id",
                    "price",
                ],
            )

            self.assertEqual(
                result.expected_rows,
                3,
            )
            self.assertEqual(
                result.id_column,
                "option_id",
            )
            self.assertEqual(
                result.expected_ids,
                [10, 20, 30],
            )
            self.assertTrue(
                result.require_id_order
            )


if __name__ == "__main__":
    unittest.main()
