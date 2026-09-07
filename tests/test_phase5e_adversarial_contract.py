from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from agent.candidate_workspace import (
    CandidateWorkspace,
    CandidateWorkspaceError,
)
from agent.code_validator import (
    validate_python_code,
)
from agent.executor import (
    MAX_CAPTURE_CHARS,
    _build_environment,
    run_python_candidate,
)
from agent.final_audit import (
    _normalise_required_path,
)
from agent.publish_pipeline import (
    PublishError,
    _safe_relative_path,
)


class Phase5EAdversarialContractTests(
    unittest.TestCase
):
    def _task(
        self,
        root: Path,
    ) -> Path:
        task_dir = root / "task"
        data_dir = task_dir / "environment" / "data"
        data_dir.mkdir(parents=True)

        (data_dir / "input.txt").write_text(
            "ORIGINAL\n",
            encoding="utf-8",
        )

        (task_dir / "instruction.md").write_text(
            "Produce /output/results.txt\n",
            encoding="utf-8",
        )

        return task_dir

    def _workspace(
        self,
        root: Path,
    ) -> CandidateWorkspace:
        return CandidateWorkspace.create(
            base_dir=root / "workspaces",
            candidate_id=1,
            task_dir=self._task(root),
        )

    def _write_solver(
        self,
        workspace: CandidateWorkspace,
        source: str,
    ) -> Path:
        path = workspace.source_dir / "solver.py"
        path.write_text(
            source,
            encoding="utf-8",
        )
        return path

    def test_parent_model_proxy_and_secret_env_are_not_visible(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._workspace(root)

            solver = self._write_solver(
                workspace,
                '''
import json
import os
from pathlib import Path

sensitive = {
    key: os.environ.get(key)
    for key in (
        "MODEL_ENDPOINT",
        "MODEL_NAME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "OPENAI_API_KEY",
        "SECRET_CANARY",
    )
}

output = Path(os.environ["OUTPUT_DIR"])
output.mkdir(parents=True, exist_ok=True)
(output / "env.json").write_text(
    json.dumps(sensitive, sort_keys=True),
    encoding="utf-8",
)
'''.strip(),
            )

            injected = {
                "MODEL_ENDPOINT": "http://secret-model",
                "MODEL_NAME": "secret-model-name",
                "HTTP_PROXY": "http://secret-proxy",
                "HTTPS_PROXY": "http://secret-proxy",
                "ALL_PROXY": "http://secret-proxy",
                "OPENAI_API_KEY": "TOP-SECRET-KEY",
                "SECRET_CANARY": "CANARY-DO-NOT-LEAK",
            }

            with patch.dict(
                os.environ,
                injected,
                clear=False,
            ):
                result = run_python_candidate(
                    solver,
                    cwd=workspace.root_dir,
                    timeout_seconds=5,
                    env_overrides={
                        "QFBENCH_SEED": "7",
                        "PYTHONHASHSEED": "7",
                    },
                )

            self.assertEqual(
                result.return_code,
                0,
                msg=result.stderr,
            )

            observed = json.loads(
                (workspace.output_dir / "env.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertTrue(
                all(
                    value is None
                    for value in observed.values()
                )
            )

    def test_unsafe_environment_override_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "input").mkdir()
            (cwd / "output").mkdir()

            with self.assertRaises(ValueError):
                _build_environment(
                    cwd=cwd,
                    env_overrides={
                        "MODEL_ENDPOINT": "http://forbidden"
                    },
                )

    def test_candidate_cannot_modify_original_task_input(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task_dir = self._task(root)
            workspace = CandidateWorkspace.create(
                base_dir=root / "workspaces",
                candidate_id=1,
                task_dir=task_dir,
            )

            solver = self._write_solver(
                workspace,
                '''
import os
from pathlib import Path

path = Path(os.environ["INPUT_DIR"]) / "data" / "input.txt"
path.write_text(
    "MUTATED\\n",
    encoding="utf-8",
)
'''.strip(),
            )

            result = run_python_candidate(
                solver,
                cwd=workspace.root_dir,
                timeout_seconds=5,
                env_overrides={
                    "QFBENCH_SEED": "1",
                    "PYTHONHASHSEED": "1",
                },
            )

            self.assertNotEqual(
                result.return_code,
                0,
            )

            original = (
                task_dir
                / "environment"
                / "data"
                / "input.txt"
            ).read_text(
                encoding="utf-8"
            )

            self.assertEqual(
                original,
                "ORIGINAL\n",
            )

    def test_candidate_cannot_write_outside_allowed_workspace_roots(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._workspace(root)

            escaped = (
                workspace
                .root_dir
                .parent
                / "escaped.txt"
            )

            solver = self._write_solver(
                workspace,
                '''
from pathlib import Path

Path("../escaped.txt").write_text(
    "escape",
    encoding="utf-8",
)
'''.strip(),
            )

            result = run_python_candidate(
                solver,
                cwd=workspace.root_dir,
                timeout_seconds=5,
                env_overrides={
                    "QFBENCH_SEED": "1",
                    "PYTHONHASHSEED": "1",
                },
            )

            self.assertNotEqual(
                result.return_code,
                0,
            )
            self.assertFalse(
                escaped.exists()
            )

    def test_runtime_network_creation_is_denied(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._workspace(root)

            solver = self._write_solver(
                workspace,
                '''
import socket

socket.socket()
'''.strip(),
            )

            result = run_python_candidate(
                solver,
                cwd=workspace.root_dir,
                timeout_seconds=5,
                env_overrides={
                    "QFBENCH_SEED": "1",
                    "PYTHONHASHSEED": "1",
                },
            )

            self.assertNotEqual(
                result.return_code,
                0,
            )
            self.assertIn(
                "network access",
                result.stderr.lower(),
            )

    def test_runtime_child_process_creation_is_denied(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._workspace(root)

            solver = self._write_solver(
                workspace,
                '''
import subprocess

subprocess.run(
    ["python", "-c", "print('bad')"],
    check=True,
)
'''.strip(),
            )

            result = run_python_candidate(
                solver,
                cwd=workspace.root_dir,
                timeout_seconds=5,
                env_overrides={
                    "QFBENCH_SEED": "1",
                    "PYTHONHASHSEED": "1",
                },
            )

            self.assertNotEqual(
                result.return_code,
                0,
            )
            self.assertIn(
                "child-process",
                result.stderr.lower(),
            )

    def test_runaway_candidate_hits_wall_timeout(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._workspace(root)

            solver = self._write_solver(
                workspace,
                "while True:\n    pass\n",
            )

            result = run_python_candidate(
                solver,
                cwd=workspace.root_dir,
                timeout_seconds=0.25,
                env_overrides={
                    "QFBENCH_SEED": "1",
                    "PYTHONHASHSEED": "1",
                },
            )

            self.assertTrue(
                result.timed_out
            )
            self.assertIsNone(
                result.return_code
            )

    def test_returned_stdout_is_bounded(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._workspace(root)

            solver = self._write_solver(
                workspace,
                (
                    "print('X' * "
                    f"{MAX_CAPTURE_CHARS * 3})\n"
                ),
            )

            result = run_python_candidate(
                solver,
                cwd=workspace.root_dir,
                timeout_seconds=5,
                env_overrides={
                    "QFBENCH_SEED": "1",
                    "PYTHONHASHSEED": "1",
                },
            )

            self.assertEqual(
                result.return_code,
                0,
                msg=result.stderr,
            )
            self.assertLessEqual(
                len(result.stdout),
                (
                    MAX_CAPTURE_CHARS
                    + len("\n...[truncated]...")
                ),
            )
            self.assertIn(
                "[truncated]",
                result.stdout,
            )

    @unittest.skipUnless(
        hasattr(os, "symlink"),
        "symlinks unavailable",
    )
    def test_task_data_symlink_is_rejected_before_copy(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task_dir = root / "task"
            data_dir = task_dir / "environment" / "data"
            data_dir.mkdir(parents=True)

            outside = root / "outside.txt"
            outside.write_text(
                "secret",
                encoding="utf-8",
            )

            (data_dir / "linked.txt").symlink_to(
                outside
            )

            with self.assertRaises(
                CandidateWorkspaceError
            ):
                CandidateWorkspace.create(
                    base_dir=root / "workspaces",
                    candidate_id=1,
                    task_dir=task_dir,
                )

    def test_static_validator_blocks_escape_primitives(
        self,
    ) -> None:
        cases = (
            "import socket\n",
            "import requests\n",
            "import urllib.request\n",
            "import subprocess\n",
            "import ctypes\n",
            "import cffi\n",
            "import importlib\n",
            "__import__('socket')\n",
            "import os\nos.system('echo bad')\n",
        )

        for code in cases:
            with self.subTest(code=code):
                with self.assertRaises(
                    ValueError
                ):
                    validate_python_code(
                        code
                    )

    def test_final_audit_path_normalisation_fails_closed(
        self,
    ) -> None:
        bad_paths = (
            "../escape.csv",
            "/tmp/escape.csv",
            "/etc/passwd",
            "/output/../escape.csv",
            "/app/output/../../escape.csv",
            "",
        )

        for raw in bad_paths:
            with self.subTest(raw=raw):
                with self.assertRaises(
                    ValueError
                ):
                    _normalise_required_path(
                        raw
                    )

        self.assertEqual(
            _normalise_required_path(
                "/output/nested/results.csv"
            ),
            "nested/results.csv",
        )

        self.assertEqual(
            _normalise_required_path(
                "/app/output/nested/results.csv"
            ),
            "nested/results.csv",
        )

    def test_publish_inventory_blocks_verifier_owned_artifacts(
        self,
    ) -> None:
        bad_paths = (
            "reward.json",
            "reward.txt",
            "pytest_report.json",
            "nested/test_outputs.py",
            "nested/test.sh",
            "../escape.csv",
            "/tmp/escape.csv",
        )

        for raw in bad_paths:
            with self.subTest(raw=raw):
                with self.assertRaises(
                    PublishError
                ):
                    _safe_relative_path(
                        raw
                    )


if __name__ == "__main__":
    unittest.main()