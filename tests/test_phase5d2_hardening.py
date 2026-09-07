from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import os
import unittest

from agent.code_validator import validate_python_code
from agent.publish_pipeline import (
    PublishError,
    _copy_audited_files,
    _safe_relative_path,
)
from agent.runtime_adapters import _build_repair_prompt


class Phase5D2HardeningTests(unittest.TestCase):
    def test_validator_allows_normal_quant_code(self) -> None:
        code = """
import os
from pathlib import Path
import numpy as np
import pandas as pd

seed = int(os.environ.get("QFBENCH_SEED", "0"))
output_dir = Path(os.environ.get("OUTPUT_DIR", "/output"))
frame = pd.DataFrame({"x": np.array([1.0, 2.0])})
""".strip()

        self.assertEqual(
            validate_python_code(code),
            code,
        )

    def test_validator_blocks_escape_primitives(self) -> None:
        examples = (
            "import socket\nsocket.socket()\n",
            "import requests\nrequests.get('https://example.com')\n",
            "import subprocess\nsubprocess.run(['echo', 'x'])\n",
            "import ctypes\n",
            "import cffi\n",
            "import os\nos.system('echo x')\n",
            "from os import fork\nfork()\n",
            "__import__('socket')\n",
            "import importlib\n",
        )

        for code in examples:
            with self.subTest(code=code):
                with self.assertRaises(ValueError):
                    validate_python_code(code)

    def test_validator_blocks_verifier_artifacts(self) -> None:
        with self.assertRaises(ValueError):
            validate_python_code(
                "from pathlib import Path\n"
                "Path('pytest_report.json').write_text('x')\n"
            )

    def test_publish_path_rejects_unsafe_names(self) -> None:
        for value in (
            "../escape.csv",
            "/tmp/escape.csv",
            "reward.json",
            "nested/pytest_report.json",
        ):
            with self.subTest(value=value):
                with self.assertRaises(PublishError):
                    _safe_relative_path(value)

    @unittest.skipUnless(
        hasattr(os, "symlink"),
        "symlinks are unavailable",
    )
    def test_publish_rejects_symlink_ancestor(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            outside = root / "outside"
            outside.mkdir()

            (outside / "results.csv").write_text(
                "x\n1\n",
                encoding="utf-8",
            )

            (source / "nested").symlink_to(
                outside,
                target_is_directory=True,
            )

            with self.assertRaises(PublishError):
                _copy_audited_files(
                    source_dir=source,
                    staging_dir=root / "staging",
                    required_files=(
                        "nested/results.csv",
                    ),
                )

    def test_repair_prompt_uses_workspace_contract(self) -> None:
        request = SimpleNamespace(
            specification=SimpleNamespace(
                instruction_text=(
                    "Write /app/output/results.csv"
                )
            ),
            compiled_specification=SimpleNamespace(
                to_dict=lambda: {
                    "deliverables": [
                        {
                            "path": (
                                "/app/output/results.csv"
                            )
                        }
                    ]
                }
            ),
            candidate_id=1,
            candidate_seed=42,
            brief=SimpleNamespace(
                to_dict=lambda: {"strategy": "repair"}
            ),
            source_code="print('x')",
        )

        prompt = _build_repair_prompt(request)

        self.assertIn(
            'os.environ.get("INPUT_DIR", "/input")',
            prompt,
        )
        self.assertIn(
            'os.environ.get("OUTPUT_DIR", "/output")',
            prompt,
        )
        self.assertIn(
            'INPUT_DIR / "data"',
            prompt,
        )
        self.assertIn(
            "Do not create child processes",
            prompt,
        )
        self.assertIn(
            "do not create reward.json",
            prompt,
        )


if __name__ == "__main__":
    unittest.main()