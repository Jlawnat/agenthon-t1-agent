from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from agent.executor import (
    MAX_CAPTURE_BYTES,
    MAX_CAPTURE_CHARS,
    _BoundedCaptureBuffer,
    run_python_candidate,
)


class Phase5E2ExecutorHardeningTests(unittest.TestCase):
    def _workspace(self, root: Path) -> tuple[Path, Path]:
        cwd = root / "candidate"
        for name in ("input", "output", "src"):
            (cwd / name).mkdir(parents=True, exist_ok=True)
        return cwd, cwd / "src" / "solver.py"

    def _run(self, *, cwd: Path, solver: Path, timeout: float = 5.0):
        return run_python_candidate(
            solver,
            cwd=cwd,
            timeout_seconds=timeout,
            env_overrides={
                "QFBENCH_SEED": "123",
                "PYTHONHASHSEED": "123",
            },
        )

    def test_capture_buffer_never_retains_more_than_cap(self) -> None:
        capture = _BoundedCaptureBuffer()
        capture.feed(b"x" * (MAX_CAPTURE_BYTES * 20))
        self.assertEqual(capture.stored_bytes, MAX_CAPTURE_BYTES)
        self.assertTrue(capture.truncated)

    def test_huge_stdout_and_stderr_are_bounded(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cwd, solver = self._workspace(root)
            solver.write_text(
                "import sys\n"
                "sys.stdout.write('A' * 5000000)\n"
                "sys.stderr.write('B' * 5000000)\n",
                encoding="utf-8",
            )
            result = self._run(cwd=cwd, solver=solver, timeout=10.0)
            self.assertEqual(result.return_code, 0, msg=result.stderr)
            cap = MAX_CAPTURE_CHARS + len("\n...[truncated]...")
            self.assertLessEqual(len(result.stdout), cap)
            self.assertLessEqual(len(result.stderr), cap)
            self.assertIn("[truncated]", result.stdout)
            self.assertIn("[truncated]", result.stderr)

    def test_timeout_returns_promptly(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cwd, solver = self._workspace(root)
            solver.write_text("while True:\n    pass\n", encoding="utf-8")
            started = time.perf_counter()
            result = self._run(cwd=cwd, solver=solver, timeout=0.25)
            elapsed = time.perf_counter() - started
            self.assertTrue(result.timed_out)
            self.assertIsNone(result.return_code)
            self.assertLess(elapsed, 2.0)

    @unittest.skipUnless(os.name == "posix", "process-group test requires POSIX")
    def test_timeout_kills_native_forked_descendant(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cwd, solver = self._workspace(root)
            marker = cwd / "output" / "survivor.txt"
            child_source = (
                "import ctypes\n"
                "import os\n"
                "from pathlib import Path\n"
                "import time\n"
                "libc = ctypes.CDLL(None)\n"
                "pid = libc.fork()\n"
                "if pid == 0:\n"
                "    time.sleep(1.0)\n"
                "    marker = Path(os.environ['OUTPUT_DIR']) / 'survivor.txt'\n"
                "    marker.write_text('alive', encoding='utf-8')\n"
                "    os._exit(0)\n"
                "while True:\n"
                "    pass\n"
            )
            solver.write_text(child_source, encoding="utf-8")
            result = self._run(cwd=cwd, solver=solver, timeout=0.25)
            self.assertTrue(result.timed_out)
            time.sleep(1.25)
            self.assertFalse(
                marker.exists(),
                "A descendant survived the candidate timeout.",
            )


if __name__ == "__main__":
    unittest.main()