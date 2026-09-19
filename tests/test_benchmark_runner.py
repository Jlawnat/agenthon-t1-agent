from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.benchmark_runner import (
    benchmark_exit_code,
    build_agent_command,
    build_checker_command,
    build_verifier_command,
    classify_status,
    load_inventory,
    parse_pytest_report,
    parse_reward,
    run_smoke,
    select_units,
    sha256_file,
    SmokeExecution,
)


class BenchmarkRunnerContractTests(unittest.TestCase):
    def test_unexpected_verifier_exit_propagates(self) -> None:
        execution = SmokeExecution(
            agent_return_code=0,
            agent_timed_out=False,
            checker_return_code=0,
            checker_timed_out=False,
            verifier_return_code=7,
            verifier_timed_out=False,
            elapsed_seconds=1.0,
        )

        self.assertEqual(execution.return_code, 7)
        self.assertEqual(
            classify_status(
                return_code=execution.return_code,
                timed_out=execution.timed_out,
                reward=1.0,
            ),
            "harness_error",
        )
        self.assertEqual(
            benchmark_exit_code([{"status": "harness_error"}]),
            1,
        )

        verifier_failure = SmokeExecution(
            agent_return_code=0,
            agent_timed_out=False,
            checker_return_code=0,
            checker_timed_out=False,
            verifier_return_code=1,
            verifier_timed_out=False,
            elapsed_seconds=1.0,
        )
        self.assertEqual(verifier_failure.return_code, 1)

    def test_subprocess_launch_failure_is_captured_per_unit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "tools.benchmark_runner.subprocess.Popen",
                side_effect=OSError("simulated launch failure"),
            ):
                execution = run_smoke(
                    repo=root,
                    unit_dir=root / "unit",
                    output_dir=root / "output",
                    image="agenthon-t1:dev",
                    network="qfb2-eval",
                    agent_timeout_seconds=10.0,
                    verifier_timeout_seconds=10.0,
                    log_path=root / "smoke.log",
                )

        self.assertEqual(execution.agent_return_code, None)
        self.assertFalse(execution.agent_timed_out)
        self.assertEqual(execution.launch_error, "agent")
        self.assertEqual(execution.return_code, 127)

    def test_agent_command_owns_container_execution(self) -> None:
        command = build_agent_command(
            unit_dir=Path("/units/example"),
            output_dir=Path("/runs/output"),
            image="agenthon-t1:dev",
            network="qfb2-eval",
            environment={
                "MODEL_ENDPOINT": "http://agenthon-mock-model:8000",
                "MODEL_TOKEN": "test-token",
                "QFBENCH_SEED": "42",
            },
        )

        self.assertEqual(command[:5], [
            "docker",
            "run",
            "--rm",
            "--network",
            "qfb2-eval",
        ])
        self.assertIn("/units/example:/input:ro", command)
        self.assertIn("/runs/output:/app/output", command)
        self.assertEqual(command[-5:], [
            "solve",
            "--task-dir",
            "/input",
            "--out",
            "/app/output",
        ])
        self.assertNotIn("--agent-image", command)

    def test_verifier_command_matches_installed_qfbench2_contract(self) -> None:
        command = build_verifier_command(
            unit_dir=Path("/units/example"),
            output_dir=Path("/runs/output"),
        )

        self.assertEqual(
            command,
            [
                "qfbench2",
                "smoke",
                "/units/example",
                "/runs/output",
                "--track",
                "coding",
            ],
        )
        self.assertNotIn("--agent-image", command)

    def test_checker_command_owns_reward_artifact_generation(self) -> None:
        command = build_checker_command(
            unit_dir=Path("/units/example"),
            output_dir=Path("/runs/output"),
        )

        self.assertIn("--network", command)
        self.assertIn("none", command)
        self.assertIn("/units/example:/input:ro", command)
        self.assertIn("/runs/output:/app/output", command)
        self.assertIn("/runs/output:/output", command)
        self.assertEqual(command[-2:], [
            "bash",
            "/input/checks/test.sh",
        ])

    def test_harness_errors_produce_nonzero_runner_exit(self) -> None:
        self.assertEqual(
            benchmark_exit_code(
                [{"status": "pass"}, {"status": "fail"}]
            ),
            0,
        )
        self.assertEqual(
            benchmark_exit_code(
                [{"status": "harness_error"}]
            ),
            1,
        )
        self.assertEqual(
            benchmark_exit_code(
                [{"status": "runner_timeout"}]
            ),
            1,
        )

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
