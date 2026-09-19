from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent.candidate_workspace import CandidateWorkspace


class CandidateWorkspaceNoDataTests(unittest.TestCase):
    def test_environment_exists_but_data_directory_is_absent(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task"

            (task / "environment").mkdir(parents=True)
            (task / "instruction.md").write_text(
                "# Parameter-only task\n",
                encoding="utf-8",
            )

            workspace = CandidateWorkspace.create(
                base_dir=root / "work",
                candidate_id=1,
                task_dir=task,
            )

            data = workspace.input_dir / "data"

            self.assertTrue(data.exists())
            self.assertTrue(data.is_dir())
            self.assertFalse(data.is_symlink())
            self.assertEqual(list(data.iterdir()), [])

    def test_environment_directory_can_be_absent(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task"
            task.mkdir(parents=True)

            (task / "instruction.md").write_text(
                "# Parameter-only task\n",
                encoding="utf-8",
            )

            workspace = CandidateWorkspace.create(
                base_dir=root / "work",
                candidate_id=2,
                task_dir=task,
            )

            data = workspace.input_dir / "data"

            self.assertTrue(data.exists())
            self.assertTrue(data.is_dir())
            self.assertEqual(list(data.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
