from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent.specification import (
    load_task_specification,
)


class Phase5MorningContractTests(
    unittest.TestCase
):
    def _write_task(
        self,
        root: Path,
        *,
        instruction: str,
        card: str,
    ) -> Path:
        task_dir = (
            root / "task"
        )
        task_dir.mkdir(
            parents=True
        )

        (
            task_dir
            / "instruction.md"
        ).write_text(
            instruction,
            encoding="utf-8",
        )

        (
            task_dir
            / "card.toml"
        ).write_text(
            card,
            encoding="utf-8",
        )

        return task_dir

    def test_reads_agent_timeout_not_environment_timeout(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = self._write_task(
                root,
                instruction=(
                    "### /app/output/results.csv\n\n"
                    "| Column | Type |\n"
                    "|---|---|\n"
                    "| `id` | `int64` |\n"
                ),
                card=(
                    'schema_version = "2.0"\n\n'
                    '[task]\n'
                    'id = "official-task-id"\n\n'
                    '[metadata]\n'
                    'category = "risk-management"\n\n'
                    '[agent]\n'
                    'timeout_sec = 5400.0\n\n'
                    '[environment]\n'
                    'timeout_seconds = 12\n'
                    'network = "restricted"\n'
                ),
            )

            spec = (
                load_task_specification(
                    task_dir
                )
            )

            self.assertEqual(
                spec.task_id,
                "official-task-id",
            )
            self.assertEqual(
                spec.runtime_seconds,
                5400.0,
            )
            self.assertEqual(
                spec.network_mode,
                "restricted",
            )

    def test_extracts_app_output_deliverable_and_columns(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = self._write_task(
                root,
                instruction=(
                    "Produce the file.\n\n"
                    "### /app/output/nested/results.json\n\n"
                    "| Column | Type |\n"
                    "|---|---|\n"
                    "| `id` | `int64` |\n"
                    "| `value` | `float64` |\n"
                ),
                card=(
                    '[task]\n'
                    'id = "nested-output"\n'
                ),
            )

            spec = (
                load_task_specification(
                    task_dir
                )
            )

            self.assertEqual(
                spec.required_output_paths,
                [
                    (
                        "/app/output/"
                        "nested/results.json"
                    )
                ],
            )
            self.assertEqual(
                spec.required_output_columns,
                [
                    "id",
                    "value",
                ],
            )

    def test_extracts_multiple_dual_mount_deliverables(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = self._write_task(
                root,
                instruction=(
                    "Write `/app/output/a.csv` and "
                    "`/output/nested/b.json`.\n"
                ),
                card=(
                    '[task]\n'
                    'id = "multi-file"\n'
                ),
            )

            spec = (
                load_task_specification(
                    task_dir
                )
            )

            self.assertEqual(
                spec.required_output_paths,
                [
                    "/app/output/a.csv",
                    "/output/nested/b.json",
                ],
            )

    def test_verifier_reward_artifacts_are_not_deliverables(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = self._write_task(
                root,
                instruction=(
                    "Write `/app/output/results.csv`.\n"
                    "The checker later writes "
                    "`/output/reward.json` and "
                    "`/app/output/pytest_report.json`.\n"
                ),
                card=(
                    '[task]\n'
                    'id = "reward-filter"\n'
                ),
            )

            spec = (
                load_task_specification(
                    task_dir
                )
            )

            self.assertEqual(
                spec.required_output_paths,
                [
                    "/app/output/results.csv",
                ],
            )


if __name__ == "__main__":
    unittest.main()