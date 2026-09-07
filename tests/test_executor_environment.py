from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys
import unittest

from agent.executor import run_python_candidate


class ExecutorEnvironmentTests(unittest.TestCase):

    def test_candidate_does_not_inherit_sensitive_parent_environment(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "solver.py"

            script.write_text(
                """
import json
import os

keys = [
    "MODEL_ENDPOINT",
    "MODEL_NAME",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "RANDOM_PARENT_SECRET",
]

print(
    json.dumps(
        {
            key: key in os.environ
            for key in keys
        }
    )
)
""".strip()
                + "\n",
                encoding="utf-8",
            )

            parent_values = {
                "MODEL_ENDPOINT": (
                    "https://example.invalid"
                ),
                "MODEL_NAME": "secret-model",
                "OPENAI_API_KEY": "secret-openai",
                "ANTHROPIC_API_KEY": (
                    "secret-anthropic"
                ),
                "RANDOM_PARENT_SECRET": (
                    "should-not-leak"
                ),
            }

            with patch.dict(
                os.environ,
                parent_values,
                clear=False,
            ):
                result = run_python_candidate(
                    script_path=script,
                    cwd=root,
                    timeout_seconds=5,
                )

            self.assertEqual(
                result.return_code,
                0,
            )

            payload = json.loads(
                result.stdout.strip()
            )

            self.assertEqual(
                payload,
                {
                    "MODEL_ENDPOINT": False,
                    "MODEL_NAME": False,
                    "OPENAI_API_KEY": False,
                    "ANTHROPIC_API_KEY": False,
                    "RANDOM_PARENT_SECRET": False,
                },
            )

    def test_seed_variables_are_allowed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "solver.py"

            script.write_text(
                """
import json
import os

print(
    json.dumps(
        {
            "qfbench": os.environ.get(
                "QFBENCH_SEED"
            ),
            "pythonhash": os.environ.get(
                "PYTHONHASHSEED"
            ),
        }
    )
)
""".strip()
                + "\n",
                encoding="utf-8",
            )

            result = run_python_candidate(
                script_path=script,
                cwd=root,
                timeout_seconds=5,
                env_overrides={
                    "QFBENCH_SEED": "12345",
                    "PYTHONHASHSEED": "12345",
                },
            )

            self.assertEqual(
                result.return_code,
                0,
            )

            payload = json.loads(
                result.stdout.strip()
            )

            self.assertEqual(
                payload["qfbench"],
                "12345",
            )

            self.assertEqual(
                payload["pythonhash"],
                "12345",
            )

    def test_unsafe_environment_override_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "solver.py"

            script.write_text(
                "print('hello')\n",
                encoding="utf-8",
            )

            with self.assertRaises(
                ValueError
            ):
                run_python_candidate(
                    script_path=script,
                    cwd=root,
                    timeout_seconds=5,
                    env_overrides={
                        "MODEL_ENDPOINT": (
                            "https://example.invalid"
                        ),
                    },
                )

    def test_candidate_uses_current_python_interpreter(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "solver.py"

            script.write_text(
                """
import sys
print(sys.executable)
""".strip()
                + "\n",
                encoding="utf-8",
            )

            result = run_python_candidate(
                script_path=script,
                cwd=root,
                timeout_seconds=5,
            )

            self.assertEqual(
                result.return_code,
                0,
            )

            self.assertEqual(
                Path(
                    result.stdout.strip()
                ).resolve(),
                Path(sys.executable).resolve(),
            )


if __name__ == "__main__":
    unittest.main()