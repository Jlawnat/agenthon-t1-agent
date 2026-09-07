from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from agent.clean_room import (
    clean_room_reexecute,
)
from agent.schema_expectations import (
    SchemaExpectations,
)


def _compiled_spec():
    return SimpleNamespace(
        task_id="clean-room-test",
        category="credit",
        difficulty="easy",
        deliverables=[
            {
                "path": (
                    "/output/results.csv"
                ),
                "format": "csv",
            }
        ],
        required_columns=[
            "option_id",
            "default_probability",
        ],
        required_row_rules=[],
        ordering_rules=[],
        units={},
        conventions=[],
        edge_cases=[],
        invariants=[],
        review_packs=[],
        numerical_risks=[],
        assumptions=[],
        unresolved_questions=[],
        required_dtypes={
            "option_id": "int64",
            "default_probability": (
                "float64"
            ),
        },
    )


def _schema():
    return SchemaExpectations(
        expected_rows=2,
        id_column="option_id",
        expected_ids=[1, 2],
        require_id_order=True,
        expected_dtypes={
            "option_id": "int64",
            "default_probability": (
                "float64"
            ),
        },
    )


def _make_task_dir(
    root: Path,
) -> Path:
    task_dir = root / "task"

    data_dir = (
        task_dir
        / "environment"
        / "data"
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        data_dir
        / "marker.txt"
    ).write_text(
        "clean-room-input",
        encoding="utf-8",
    )

    return task_dir


def _write_solver(
    path: Path,
    code: str,
) -> None:
    path.write_text(
        code.strip() + "\n",
        encoding="utf-8",
    )


class CleanRoomAcceptanceTests(
    unittest.TestCase
):
    def test_clean_solver_reexecutes_and_passes(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _make_task_dir(
                root
            )

            solver = (
                root
                / "selected_solver.py"
            )

            _write_solver(
                solver,
                """
from pathlib import Path
import csv


def main() -> None:
    input_marker = (
        Path("input")
        / "data"
        / "marker.txt"
    )

    if not input_marker.exists():
        raise FileNotFoundError(
            "Fresh task input is missing."
        )

    output_dir = Path("output")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_dir
        / "results.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "option_id",
                "default_probability",
            ]
        )

        writer.writerow(
            [1, 0.20]
        )

        writer.writerow(
            [2, 0.40]
        )


if __name__ == "__main__":
    main()
""",
            )

            result = clean_room_reexecute(
                selected_solver_path=solver,
                task_dir=task_dir,
                base_dir=(
                    root
                    / "clean_rooms"
                ),
                compiled_specification=(
                    _compiled_spec()
                ),
                schema_expectations=(
                    _schema()
                ),
                timeout_seconds=10,
                workspace_id=9001,
            )

            self.assertTrue(
                result.passed
            )

            self.assertEqual(
                result.execution.return_code,
                0,
            )

            self.assertFalse(
                result.execution.timed_out
            )

            self.assertIsNotNone(
                result.audit
            )

            self.assertTrue(
                result.audit.passed
            )

    def test_old_output_is_destroyed_before_reexecution(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _make_task_dir(
                root
            )

            base_dir = (
                root
                / "clean_rooms"
            )

            stale_output = (
                base_dir
                / "candidate_9002"
                / "output"
            )

            stale_output.mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                stale_output
                / "results.csv"
            ).write_text(
                (
                    "option_id,"
                    "default_probability\n"
                    "1,0.20\n"
                    "2,0.40\n"
                ),
                encoding="utf-8",
            )

            solver = (
                root
                / "does_nothing.py"
            )

            _write_solver(
                solver,
                """
def main() -> None:
    pass


if __name__ == "__main__":
    main()
""",
            )

            result = clean_room_reexecute(
                selected_solver_path=solver,
                task_dir=task_dir,
                base_dir=base_dir,
                compiled_specification=(
                    _compiled_spec()
                ),
                schema_expectations=(
                    _schema()
                ),
                timeout_seconds=10,
                workspace_id=9002,
            )

            self.assertFalse(
                result.passed
            )

            self.assertEqual(
                result.execution.return_code,
                0,
            )

            self.assertIsNotNone(
                result.audit
            )

            self.assertFalse(
                result.audit.passed
            )

            self.assertNotIn(
                "results.csv",
                result.audit.produced_files,
            )

    def test_nonzero_execution_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _make_task_dir(
                root
            )

            solver = (
                root
                / "crashing_solver.py"
            )

            _write_solver(
                solver,
                """
raise RuntimeError(
    "intentional clean-room failure"
)
""",
            )

            result = clean_room_reexecute(
                selected_solver_path=solver,
                task_dir=task_dir,
                base_dir=(
                    root
                    / "clean_rooms"
                ),
                compiled_specification=(
                    _compiled_spec()
                ),
                schema_expectations=(
                    _schema()
                ),
                timeout_seconds=10,
                workspace_id=9003,
            )

            self.assertFalse(
                result.passed
            )

            self.assertIsNotNone(
                result.execution
            )

            self.assertNotEqual(
                result.execution.return_code,
                0,
            )

            self.assertIn(
                "return code",
                result.reason.lower(),
            )

    def test_invalid_fresh_output_fails_audit(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _make_task_dir(
                root
            )

            solver = (
                root
                / "bad_output_solver.py"
            )

            _write_solver(
                solver,
                """
from pathlib import Path


def main() -> None:
    output_dir = Path("output")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir
        / "results.csv"
    ).write_text(
        (
            "option_id,"
            "default_probability\\n"
            "2,0.20\\n"
            "1,1.40\\n"
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
""",
            )

            result = clean_room_reexecute(
                selected_solver_path=solver,
                task_dir=task_dir,
                base_dir=(
                    root
                    / "clean_rooms"
                ),
                compiled_specification=(
                    _compiled_spec()
                ),
                schema_expectations=(
                    _schema()
                ),
                timeout_seconds=10,
                workspace_id=9004,
            )

            self.assertFalse(
                result.passed
            )

            self.assertEqual(
                result.execution.return_code,
                0,
            )

            self.assertIsNotNone(
                result.audit
            )

            self.assertFalse(
                result.audit.passed
            )


if __name__ == "__main__":
    unittest.main()