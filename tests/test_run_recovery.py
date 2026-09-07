from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from agent.candidate_contract import (
    CandidateRecord,
)
from agent.candidate_selector import (
    SelectionResult,
)
from agent.orchestrator_state import (
    OrchestratorState,
    PipelineStage,
)
from agent.publish_pipeline import (
    _run_token,
)
from agent.run_context import (
    RunContext,
)
from agent.run_budget import (
    RunBudget,
)
from agent.run_logger import (
    append_jsonl,
    load_json,
    save_json,
)
from agent.run_recovery import (
    RunJournal,
    recover_publish_state,
)


def _paths(
    *,
    root: Path,
    run_id: str,
):
    final_output = (
        root
        / "output"
    )

    token = _run_token(
        run_id
    )

    staging = (
        root
        / (
            f".output.staging."
            f"{token}"
        )
    )

    backup = (
        root
        / (
            f".output.backup."
            f"{token}"
        )
    )

    return (
        final_output,
        staging,
        backup,
    )


def _write_file(
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content,
        encoding="utf-8",
    )


def _state() -> OrchestratorState:
    state = OrchestratorState(
        run_id="recovery-run"
    )

    for stage in (
        PipelineStage.SNAPSHOT,
        PipelineStage.SPECIFICATION,
        PipelineStage.COMPILE,
        PipelineStage.PLAN,
    ):
        state.start_stage(
            stage
        )

        state.complete_stage(
            stage
        )

    return state


class RunRecoveryTests(
    unittest.TestCase
):

    def test_backup_restored_when_final_missing(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            run_id = (
                "recovery-run"
            )

            (
                final_output,
                staging,
                backup,
            ) = _paths(
                root=root,
                run_id=run_id,
            )

            _write_file(
                backup
                / "previous.csv",
                "previous-valid\n",
            )

            _write_file(
                staging
                / "new.csv",
                "unfinished-new\n",
            )

            result = (
                recover_publish_state(
                    final_output_dir=(
                        final_output
                    ),
                    run_id=run_id,
                    expected_files=(
                        "new.csv",
                    ),
                )
            )

            self.assertEqual(
                result.action,
                "restored_previous_output",
            )

            self.assertTrue(
                (
                    final_output
                    / "previous.csv"
                ).exists()
            )

            self.assertEqual(
                (
                    final_output
                    / "previous.csv"
                ).read_text(
                    encoding="utf-8"
                ),
                "previous-valid\n",
            )

            self.assertFalse(
                staging.exists()
            )

            self.assertFalse(
                backup.exists()
            )

            self.assertTrue(
                result.restored_backup
            )

    def test_verified_committed_output_kept_and_backup_removed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            run_id = (
                "recovery-run"
            )

            (
                final_output,
                staging,
                backup,
            ) = _paths(
                root=root,
                run_id=run_id,
            )

            _write_file(
                final_output
                / "results.csv",
                "new-valid\n",
            )

            _write_file(
                backup
                / "previous.csv",
                "previous-valid\n",
            )

            _write_file(
                staging
                / "temporary.csv",
                "stale\n",
            )

            result = (
                recover_publish_state(
                    final_output_dir=(
                        final_output
                    ),
                    run_id=run_id,
                    expected_files=(
                        "results.csv",
                    ),
                )
            )

            self.assertEqual(
                result.action,
                (
                    "kept_verified_"
                    "committed_output"
                ),
            )

            self.assertTrue(
                (
                    final_output
                    / "results.csv"
                ).exists()
            )

            self.assertEqual(
                (
                    final_output
                    / "results.csv"
                ).read_text(
                    encoding="utf-8"
                ),
                "new-valid\n",
            )

            self.assertFalse(
                backup.exists()
            )

            self.assertFalse(
                staging.exists()
            )

            self.assertTrue(
                result.removed_backup
            )

    def test_corrupt_committed_output_rolls_back_backup(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            run_id = (
                "recovery-run"
            )

            (
                final_output,
                staging,
                backup,
            ) = _paths(
                root=root,
                run_id=run_id,
            )

            _write_file(
                final_output
                / "wrong.csv",
                "corrupt-new\n",
            )

            _write_file(
                backup
                / "previous.csv",
                "previous-valid\n",
            )

            result = (
                recover_publish_state(
                    final_output_dir=(
                        final_output
                    ),
                    run_id=run_id,
                    expected_files=(
                        "results.csv",
                    ),
                )
            )

            self.assertEqual(
                result.action,
                (
                    "rolled_back_to_"
                    "previous_output"
                ),
            )

            self.assertFalse(
                (
                    final_output
                    / "wrong.csv"
                ).exists()
            )

            self.assertTrue(
                (
                    final_output
                    / "previous.csv"
                ).exists()
            )

            self.assertEqual(
                (
                    final_output
                    / "previous.csv"
                ).read_text(
                    encoding="utf-8"
                ),
                "previous-valid\n",
            )

            self.assertTrue(
                result.restored_backup
            )

    def test_uncommitted_staging_is_discarded(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            run_id = (
                "recovery-run"
            )

            (
                final_output,
                staging,
                backup,
            ) = _paths(
                root=root,
                run_id=run_id,
            )

            _write_file(
                staging
                / "results.csv",
                "uncommitted\n",
            )

            result = (
                recover_publish_state(
                    final_output_dir=(
                        final_output
                    ),
                    run_id=run_id,
                    expected_files=(
                        "results.csv",
                    ),
                )
            )

            self.assertEqual(
                result.action,
                (
                    "discarded_"
                    "uncommitted_staging"
                ),
            )

            self.assertFalse(
                staging.exists()
            )

            self.assertFalse(
                final_output.exists()
            )

            self.assertFalse(
                backup.exists()
            )

            self.assertTrue(
                result.removed_staging
            )

    def test_unverified_final_without_backup_is_removed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            run_id = (
                "recovery-run"
            )

            (
                final_output,
                staging,
                backup,
            ) = _paths(
                root=root,
                run_id=run_id,
            )

            _write_file(
                final_output
                / "wrong.csv",
                "unverified\n",
            )

            result = (
                recover_publish_state(
                    final_output_dir=(
                        final_output
                    ),
                    run_id=run_id,
                    expected_files=(
                        "results.csv",
                    ),
                )
            )

            self.assertEqual(
                result.action,
                "removed_unverified_output",
            )

            self.assertFalse(
                final_output.exists()
            )

            self.assertFalse(
                staging.exists()
            )

            self.assertFalse(
                backup.exists()
            )

            self.assertFalse(
                result.final_output_exists
            )

    def test_atomic_json_failure_preserves_previous_checkpoint(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            checkpoint = (
                root
                / "checkpoint.json"
            )

            save_json(
                {
                    "version": 1,
                    "status": "good",
                },
                checkpoint,
            )

            original_text = (
                checkpoint.read_text(
                    encoding="utf-8"
                )
            )

            with patch(
                "agent.run_logger.os.replace",
                side_effect=OSError(
                    "simulated atomic replace failure"
                ),
            ):
                with self.assertRaises(
                    OSError
                ):
                    save_json(
                        {
                            "version": 2,
                            "status": "new",
                        },
                        checkpoint,
                    )

            self.assertTrue(
                checkpoint.exists()
            )

            self.assertEqual(
                checkpoint.read_text(
                    encoding="utf-8"
                ),
                original_text,
            )

            loaded = load_json(
                checkpoint
            )

            self.assertEqual(
                loaded["version"],
                1,
            )

            self.assertEqual(
                loaded["status"],
                "good",
            )

    def test_run_journal_records_checkpoint_and_jsonl_event(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            state = _state()

            state.task_id = (
                "journal-test"
            )

            context = RunContext(
                task_id="journal-test",
                card_timeout_seconds=600.0,
                qfbench_seed=12345,
                budget=RunBudget(
                    total_wall_seconds=600.0,
                    safety_margin_seconds=5.0,
                ),
            )

            candidate = CandidateRecord(
                candidate_id=2,
                approach_name="test",
                strategy={},
            )

            candidate.current_status = (
                "validated"
            )

            selection_result = (
                SelectionResult(
                    selected_candidate_id=2,
                    ranked_candidate_ids=(
                        2,
                        1,
                        3,
                    ),
                    first_candidate_valid=False,
                    any_of_three_valid=True,
                    scorecards=(),
                )
            )

            selected_item = (
                SimpleNamespace(
                    candidate=candidate,
                    candidate_seed=777,
                )
            )

            selection = (
                SimpleNamespace(
                    selection=(
                        selection_result
                    ),
                    selected_item=(
                        selected_item
                    ),
                )
            )

            journal = RunJournal(
                run_dir=(
                    root
                    / "run"
                )
            )

            journal.event(
                "selection_completed",
                state=state,
                details={
                    "note": (
                        "structured-event"
                    ),
                },
            )

            payload = (
                journal.checkpoint(
                    state=state,
                    run_context=context,
                    selection=selection,
                    extra={
                        "phase": "4.10",
                    },
                )
            )

            self.assertTrue(
                journal.events_path.exists()
            )

            self.assertTrue(
                journal.checkpoint_path.exists()
            )

            lines = (
                journal.events_path
                .read_text(
                    encoding="utf-8"
                )
                .splitlines()
            )

            self.assertEqual(
                len(lines),
                1,
            )

            event = json.loads(
                lines[0]
            )

            self.assertEqual(
                event[
                    "event_type"
                ],
                "selection_completed",
            )

            self.assertEqual(
                event[
                    "run_id"
                ],
                "recovery-run",
            )

            self.assertEqual(
                event[
                    "details"
                ][
                    "note"
                ],
                "structured-event",
            )

            stored = (
                journal.load_checkpoint()
            )

            self.assertIsNotNone(
                stored
            )

            self.assertEqual(
                stored[
                    "run_context"
                ][
                    "qfbench_seed"
                ],
                12345,
            )

            self.assertEqual(
                stored[
                    "selection"
                ][
                    "selected_candidate_id"
                ],
                2,
            )

            self.assertFalse(
                stored[
                    "selection"
                ][
                    "first_candidate_valid"
                ]
            )

            self.assertTrue(
                stored[
                    "selection"
                ][
                    "any_of_three_valid"
                ]
            )

            self.assertEqual(
                stored[
                    "selected_candidate"
                ][
                    "candidate_seed"
                ],
                777,
            )

            self.assertEqual(
                payload[
                    "extra"
                ][
                    "phase"
                ],
                "4.10",
            )


if __name__ == "__main__":
    unittest.main()