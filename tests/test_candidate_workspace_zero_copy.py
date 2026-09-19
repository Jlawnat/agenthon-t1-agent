from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent.candidate_workspace import (
    CandidateWorkspace,
    CandidateWorkspaceError,
)


class CandidateWorkspaceZeroCopyTests(unittest.TestCase):
    def _task(
        self,
        root: Path,
        *,
        dockerfile_text: str | None = None,
    ) -> tuple[Path, Path]:
        task = root / "task"
        env = task / "environment"
        data = env / "data"
        data.mkdir(parents=True)

        (data / "sample.csv").write_text(
            "x\n1\n2\n",
            encoding="utf-8",
        )

        if dockerfile_text is not None:
            (env / "Dockerfile").write_text(
                dockerfile_text,
                encoding="utf-8",
            )

        return task, data

    def test_workspace_uses_zero_copy_file_links(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task, data = self._task(root)

            with patch(
                "agent.candidate_workspace."
                "_task_data_is_effectively_read_only",
                return_value=True,
            ):
                workspace = CandidateWorkspace.create(
                    base_dir=root / "work",
                    candidate_id=1,
                    task_dir=task,
                )

            candidate_data = (
                workspace.input_dir
                / "data"
            )
            staged_file = (
                candidate_data
                / "sample.csv"
            )

            self.assertTrue(
                candidate_data.is_dir()
            )
            self.assertFalse(
                candidate_data.is_symlink()
            )
            self.assertTrue(
                staged_file.is_symlink()
            )
            self.assertEqual(
                staged_file.resolve(),
                (data / "sample.csv").resolve(),
            )

            shutil.rmtree(
                workspace.root_dir
            )

            self.assertTrue(
                (data / "sample.csv").exists()
            )

    def test_dockerfile_rename_creates_legacy_alias(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task, data = self._task(
                root,
                dockerfile_text=(
                    "FROM finance-bench-sandbox:latest\n"
                    "WORKDIR /app\n"
                    "COPY data/sample.csv "
                    "/app/data/legacy_name.csv\n"
                ),
            )

            with patch(
                "agent.candidate_workspace."
                "_task_data_is_effectively_read_only",
                return_value=True,
            ):
                workspace = CandidateWorkspace.create(
                    base_dir=root / "work",
                    candidate_id=1,
                    task_dir=task,
                )

            raw_name = (
                workspace.input_dir
                / "data"
                / "sample.csv"
            )
            legacy_name = (
                workspace.input_dir
                / "data"
                / "legacy_name.csv"
            )

            self.assertTrue(
                raw_name.is_symlink()
            )
            self.assertTrue(
                legacy_name.is_symlink()
            )
            self.assertEqual(
                raw_name.resolve(),
                (data / "sample.csv").resolve(),
            )
            self.assertEqual(
                legacy_name.resolve(),
                (data / "sample.csv").resolve(),
            )

    def test_source_data_symlinks_are_still_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task, data = self._task(root)

            outside = root / "outside.txt"
            outside.write_text(
                "secret",
                encoding="utf-8",
            )

            (data / "bad-link").symlink_to(
                outside
            )

            with self.assertRaises(
                CandidateWorkspaceError
            ):
                CandidateWorkspace.create(
                    base_dir=root / "work",
                    candidate_id=1,
                    task_dir=task,
                )


if __name__ == "__main__":
    unittest.main()
