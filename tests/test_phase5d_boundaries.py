from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent.candidate_workspace import (
    CandidateWorkspace,
    CandidateWorkspaceError,
)
from agent.executor import (
    _build_environment,
    run_python_candidate,
)
from agent.final_audit import (
    _inventory_output,
    _normalise_required_path,
)


class CandidateWorkspaceSecurityTests(
    unittest.TestCase
):
    def _task(
        self,
        root: Path,
    ) -> Path:
        task_dir = (
            root
            / "task"
        )
        data_dir = (
            task_dir
            / "environment"
            / "data"
        )
        data_dir.mkdir(
            parents=True
        )
        (
            data_dir
            / "input.txt"
        ).write_text(
            "safe input\n",
            encoding="utf-8",
        )
        return task_dir

    def test_candidate_id_must_be_positive_integer(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task_dir = self._task(root)

            for invalid in (
                0,
                -1,
                True,
            ):
                with self.assertRaises(
                    ValueError
                ):
                    CandidateWorkspace.create(
                        base_dir=(
                            root / "work"
                        ),
                        candidate_id=invalid,
                        task_dir=task_dir,
                    )

    def test_task_data_symlink_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task_dir = self._task(root)

            outside = (
                root
                / "outside.txt"
            )
            outside.write_text(
                "secret\n",
                encoding="utf-8",
            )

            link = (
                task_dir
                / "environment"
                / "data"
                / "leak.txt"
            )

            try:
                link.symlink_to(
                    outside
                )
            except OSError:
                self.skipTest(
                    "Symlink creation unavailable"
                )

            with self.assertRaises(
                CandidateWorkspaceError
            ):
                CandidateWorkspace.create(
                    base_dir=(
                        root / "work"
                    ),
                    candidate_id=1,
                    task_dir=task_dir,
                )

    def test_copied_input_has_no_write_bits(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task_dir = self._task(root)

            workspace = (
                CandidateWorkspace.create(
                    base_dir=(
                        root / "work"
                    ),
                    candidate_id=1,
                    task_dir=task_dir,
                )
            )

            copied = (
                workspace.input_dir
                / "data"
                / "input.txt"
            )

            self.assertEqual(
                copied.stat().st_mode
                & 0o222,
                0,
            )


class ExecutorSecurityTests(
    unittest.TestCase
):
    def _workspace(
        self,
        root: Path,
    ) -> Path:
        workspace = (
            root / "candidate"
        )
        (
            workspace
            / "src"
        ).mkdir(
            parents=True
        )
        (
            workspace
            / "input"
        ).mkdir()
        (
            workspace
            / "output"
        ).mkdir()
        return workspace

    def _run_code(
        self,
        root: Path,
        code: str,
    ):
        workspace = (
            self._workspace(
                root
            )
        )
        script = (
            workspace
            / "src"
            / "solver.py"
        )
        script.write_text(
            code,
            encoding="utf-8",
        )

        return (
            workspace,
            run_python_candidate(
                script_path=script,
                cwd=workspace,
                timeout_seconds=10,
                env_overrides={
                    "QFBENCH_SEED": "7",
                    "PYTHONHASHSEED": "7",
                },
            ),
        )

    def test_environment_does_not_inherit_model_or_secret_values(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = (
                self._workspace(root)
            )

            with patch.dict(
                os.environ,
                {
                    "MODEL_ENDPOINT": (
                        "https://secret.test"
                    ),
                    "MODEL_NAME": "secret-model",
                    "SUPER_SECRET": "value",
                },
                clear=False,
            ):
                env = (
                    _build_environment(
                        cwd=workspace,
                        env_overrides={
                            "QFBENCH_SEED": "1",
                        },
                    )
                )

            self.assertNotIn(
                "MODEL_ENDPOINT",
                env,
            )
            self.assertNotIn(
                "MODEL_NAME",
                env,
            )
            self.assertNotIn(
                "SUPER_SECRET",
                env,
            )
            self.assertEqual(
                env["INPUT_DIR"],
                str(
                    workspace
                    / "input"
                ),
            )
            self.assertEqual(
                env["OUTPUT_DIR"],
                str(
                    workspace
                    / "output"
                ),
            )

    def test_script_escape_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = (
                self._workspace(root)
            )
            outside = (
                root / "outside.py"
            )
            outside.write_text(
                "print('outside')\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "escapes",
            ):
                run_python_candidate(
                    script_path=outside,
                    cwd=workspace,
                )

    def test_output_write_is_allowed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            workspace, result = (
                self._run_code(
                    root,
                    """
from pathlib import Path
import os

out = Path(os.environ["OUTPUT_DIR"])
(out / "ok.txt").write_text("ok", encoding="utf-8")
""".strip()
                    + "\n",
                )
            )

            self.assertEqual(
                result.return_code,
                0,
                msg=result.stderr,
            )
            self.assertEqual(
                (
                    workspace
                    / "output"
                    / "ok.txt"
                ).read_text(
                    encoding="utf-8"
                ),
                "ok",
            )

    def test_write_escape_is_blocked(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            workspace, result = (
                self._run_code(
                    root,
                    """
from pathlib import Path

Path("../escaped.txt").write_text(
    "bad",
    encoding="utf-8",
)
""".strip()
                    + "\n",
                )
            )

            self.assertNotEqual(
                result.return_code,
                0,
            )
            self.assertFalse(
                (
                    workspace.parent
                    / "escaped.txt"
                ).exists()
            )
            self.assertIn(
                "outside the allowed workspace",
                result.stderr,
            )

    def test_network_is_blocked(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            _, result = (
                self._run_code(
                    root,
                    """
import socket

socket.socket()
""".strip()
                    + "\n",
                )
            )

            self.assertNotEqual(
                result.return_code,
                0,
            )
            self.assertIn(
                "network access is disabled",
                result.stderr,
            )

    def test_child_process_is_blocked(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            _, result = (
                self._run_code(
                    root,
                    """
import subprocess

subprocess.run(
    ["echo", "bad"],
    check=False,
)
""".strip()
                    + "\n",
                )
            )

            self.assertNotEqual(
                result.return_code,
                0,
            )
            self.assertIn(
                "child-process creation is disabled",
                result.stderr,
            )


class FinalAuditBoundaryTests(
    unittest.TestCase
):
    def test_valid_output_mount_paths_normalise(
        self,
    ) -> None:
        self.assertEqual(
            _normalise_required_path(
                "/output/results.csv"
            ),
            "results.csv",
        )
        self.assertEqual(
            _normalise_required_path(
                "/app/output/nested/results.json"
            ),
            "nested/results.json",
        )

    def test_traversal_required_path_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            _normalise_required_path(
                "/output/../reward.json"
            )

    def test_arbitrary_absolute_required_path_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            _normalise_required_path(
                "/etc/passwd"
            )

    def test_output_symlink_is_inventory_failure(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = (
                root / "output"
            )
            output.mkdir()

            outside = (
                root / "outside.txt"
            )
            outside.write_text(
                "secret",
                encoding="utf-8",
            )

            link = (
                output / "results.txt"
            )

            try:
                link.symlink_to(
                    outside
                )
            except OSError:
                self.skipTest(
                    "Symlink creation unavailable"
                )

            files, unsafe = (
                _inventory_output(
                    output
                )
            )

            self.assertEqual(
                files,
                [],
            )
            self.assertEqual(
                unsafe,
                ["results.txt"],
            )


if __name__ == "__main__":
    unittest.main()