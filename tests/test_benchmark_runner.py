from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.benchmark_runner import (
    classify_status,
    load_inventory,
    parse_pytest_report,
    parse_reward,
    select_units,
    sha256_file,
)


class BenchmarkRunnerContractTests(unittest.TestCase):
    def test_load_inventory_requires_a_units_list(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.json"
            path.write_text(json.dumps({"units": "not-a-list"}), encoding="utf-8")

            with self.assertRaises(SystemExit):
                load_inventory(path)

    def test_load_inventory_rejects_non_object_units(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.json"
            path.write_text(json.dumps({"units": [{"unit_dir": "ok"}, 3]}), encoding="utf-8")

            with self.assertRaises(SystemExit):
                load_inventory(path)

    def test_selection_preserves_requested_order_and_limit(self) -> None:
        inventory = [
            {"unit_dir": "first"},
            {"unit_dir": "second"},
            {"unit_dir": "third"},
        ]

        selected = select_units(
            inventory,
            requested=["third", "first"],
            limit=None,
        )

        self.assertEqual(
            [item["unit_dir"] for item in selected],
            ["third", "first"],
        )

    def test_selection_rejects_non_positive_limit(self) -> None:
        with self.assertRaises(SystemExit):
            select_units(
                [{"unit_dir": "unit"}],
                requested=[],
                limit=0,
            )

    def test_selection_rejects_duplicate_units(self) -> None:
        with self.assertRaises(SystemExit):
            select_units(
                [{"unit_dir": "unit"}],
                requested=["unit", "unit"],
                limit=None,
            )

    def test_reward_and_pytest_artifacts_are_parsed(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "reward.json").write_text(
                json.dumps({"reward": 1.0, "details": "pass"}),
                encoding="utf-8",
            )
            (output / "pytest_report.json").write_text(
                json.dumps(
                    {
                        "tests": [
                            {"outcome": "passed", "nodeid": "test_ok"},
                            {"outcome": "failed", "nodeid": "test_bad"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            reward = parse_reward(output)
            report = parse_pytest_report(output)

            self.assertEqual(reward["reward"], 1.0)
            self.assertEqual(report["pytest_total"], 2)
            self.assertEqual(report["pytest_passed"], 1)
            self.assertEqual(report["pytest_failed"], 1)
            self.assertEqual(report["failed_tests"], ["test_bad"])

    def test_non_object_pytest_report_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "pytest_report.json").write_text(
                json.dumps(["not", "an", "object"]),
                encoding="utf-8",
            )

            report = parse_pytest_report(output)

            self.assertIsNone(report["pytest_total"])
            self.assertEqual(report["failed_tests"], [])

    def test_status_classification_is_fail_closed(self) -> None:
        self.assertEqual(
            classify_status(return_code=0, timed_out=False, reward=1.0),
            "pass",
        )
        self.assertEqual(
            classify_status(return_code=0, timed_out=False, reward=0.0),
            "fail",
        )
        self.assertEqual(
            classify_status(return_code=1, timed_out=False, reward=1.0),
            "harness_error",
        )
        self.assertEqual(
            classify_status(
                return_code=0,
                timed_out=False,
                reward=1.0,
                pytest_failed=1,
            ),
            "fail",
        )
        self.assertEqual(
            classify_status(return_code=1, timed_out=False, reward=None),
            "harness_error",
        )
        self.assertEqual(
            classify_status(return_code=0, timed_out=False, reward=None),
            "no_reward",
        )
        self.assertEqual(
            classify_status(return_code=None, timed_out=True, reward=1.0),
            "runner_timeout",
        )

    def test_inventory_hash_is_stable(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.json"
            path.write_text("{}\n", encoding="utf-8")
            first = sha256_file(path)
            second = sha256_file(path)

            self.assertEqual(first, second)
            self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
