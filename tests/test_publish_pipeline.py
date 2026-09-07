from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from agent.audit_pipeline import (
    CleanRoomAuditResult,
)
from agent.candidate_workspace import (
    CandidateWorkspace,
)
from agent.executor import (
    ExecutionResult,
)
from agent.final_audit import (
    FinalAuditResult,
)
from agent.orchestrator_state import (
    OrchestratorState,
    PipelineStage,
    RunStatus,
)
from agent.publish_pipeline import (
    PublishError,
    run_publish_stage,
)


def _state_at_publish(
) -> OrchestratorState:
    state = OrchestratorState(
        run_id="publish-test-run"
    )

    for stage in (
        PipelineStage.SNAPSHOT,
        PipelineStage.SPECIFICATION,
        PipelineStage.COMPILE,
        PipelineStage.PLAN,
        PipelineStage.GENERATE,
        PipelineStage.VALIDATE_CODE,
        PipelineStage.ISOLATED_EXECUTE,
        PipelineStage.COLLECT_EVIDENCE,
        PipelineStage.EVALUATE,
        PipelineStage.TARGETED_REPAIR,
        PipelineStage.SELECT,
        PipelineStage.CLEAN_ROOM_AUDIT,
    ):
        state.start_stage(
            stage
        )

        state.complete_stage(
            stage
        )

    return state


def _execution(
) -> ExecutionResult:
    return ExecutionResult(
        command=[
            "python",
            "solver.py",
        ],
        return_code=0,
        stdout="",
        stderr="",
        runtime_seconds=0.01,
        timed_out=False,
    )


def _audit_result(
    *,
    root: Path,
    files: dict[str, str],
    passed: bool = True,
    required_files: tuple[str, ...]
    | None = None,
) -> CleanRoomAuditResult:
    task_dir = (
        root
        / "task"
    )

    (
        task_dir
        / "environment"
        / "data"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    workspace = (
        CandidateWorkspace.create(
            base_dir=(
                root
                / "audit_workspace"
            ),
            candidate_id=1,
            task_dir=task_dir,
        )
    )

    solver_path = (
        workspace.source_dir
        / "solver.py"
    )

    solver_path.write_text(
        "print('ok')\n",
        encoding="utf-8",
    )

    for relative, content in files.items():
        path = (
            workspace.output_dir
            / relative
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            content,
            encoding="utf-8",
        )

    produced = tuple(
        sorted(files.keys())
    )

    if required_files is None:
        required = produced

    else:
        required = required_files

    audit = FinalAuditResult(
        passed=passed,
        findings=(),
        required_files=required,
        produced_files=produced,
    )

    return CleanRoomAuditResult(
        selected_candidate_id=1,
        workspace=workspace,
        solver_path=solver_path,
        execution=_execution(),
        audit=audit,
    )


class PublishPipelineTests(
    unittest.TestCase
):

    def test_successful_publish_completes_run(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            audit_result = (
                _audit_result(
                    root=root,
                    files={
                        "results.csv": (
                            "id,value\n"
                            "1,10.0\n"
                        ),
                    },
                )
            )

            final_output = (
                root
                / "final_output"
            )

            state = (
                _state_at_publish()
            )

            result = (
                run_publish_stage(
                    state=state,
                    audit_result=(
                        audit_result
                    ),
                    final_output_dir=(
                        final_output
                    ),
                )
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
                (
                    "id,value\n"
                    "1,10.0\n"
                ),
            )

            self.assertEqual(
                result.published_files,
                ("results.csv",),
            )

            self.assertFalse(
                result
                .replaced_existing_output
            )

            self.assertEqual(
                state.status,
                RunStatus.SUCCEEDED,
            )

    def test_existing_output_is_replaced_only_after_staging(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            final_output = (
                root
                / "final_output"
            )

            final_output.mkdir()

            old_file = (
                final_output
                / "old.csv"
            )

            old_file.write_text(
                "old\n",
                encoding="utf-8",
            )

            audit_result = (
                _audit_result(
                    root=root,
                    files={
                        "results.csv": (
                            "id,value\n"
                            "1,99.0\n"
                        ),
                    },
                )
            )

            result = (
                run_publish_stage(
                    state=(
                        _state_at_publish()
                    ),
                    audit_result=(
                        audit_result
                    ),
                    final_output_dir=(
                        final_output
                    ),
                )
            )

            self.assertTrue(
                result
                .replaced_existing_output
            )

            self.assertFalse(
                (
                    final_output
                    / "old.csv"
                ).exists()
            )

            self.assertTrue(
                (
                    final_output
                    / "results.csv"
                ).exists()
            )

    def test_copy_failure_preserves_previous_output(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            final_output = (
                root
                / "final_output"
            )

            final_output.mkdir()

            previous = (
                final_output
                / "previous.csv"
            )

            previous.write_text(
                "previous-valid-output\n",
                encoding="utf-8",
            )

            audit_result = (
                _audit_result(
                    root=root,
                    files={
                        "results.csv": (
                            "new-output\n"
                        ),
                    },
                )
            )

            state = (
                _state_at_publish()
            )

            with patch(
                "agent.publish_pipeline."
                "shutil.copy2",
                side_effect=OSError(
                    "simulated copy failure"
                ),
            ):
                with self.assertRaises(
                    PublishError
                ):
                    run_publish_stage(
                        state=state,
                        audit_result=(
                            audit_result
                        ),
                        final_output_dir=(
                            final_output
                        ),
                    )

            self.assertTrue(
                previous.exists()
            )

            self.assertEqual(
                previous.read_text(
                    encoding="utf-8"
                ),
                "previous-valid-output\n",
            )

            self.assertEqual(
                state.status,
                RunStatus.FAILED,
            )

    def test_promotion_failure_rolls_back_previous_output(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            final_output = (
                root
                / "final_output"
            )

            final_output.mkdir()

            previous = (
                final_output
                / "previous.csv"
            )

            previous.write_text(
                "previous-valid-output\n",
                encoding="utf-8",
            )

            audit_result = (
                _audit_result(
                    root=root,
                    files={
                        "results.csv": (
                            "new-output\n"
                        ),
                    },
                )
            )

            real_replace = (
                os.replace
            )

            replace_calls = 0

            def controlled_replace(
                source,
                destination,
            ):
                nonlocal replace_calls

                replace_calls += 1

                # First call:
                # final output -> backup
                if replace_calls == 1:
                    return real_replace(
                        source,
                        destination,
                    )

                # Second call:
                # staging -> final output
                if replace_calls == 2:
                    raise OSError(
                        "simulated promotion failure"
                    )

                # Third call:
                # rollback backup -> final output
                return real_replace(
                    source,
                    destination,
                )

            state = (
                _state_at_publish()
            )

            with patch(
                "agent.publish_pipeline."
                "os.replace",
                side_effect=(
                    controlled_replace
                ),
            ):
                with self.assertRaises(
                    PublishError
                ):
                    run_publish_stage(
                        state=state,
                        audit_result=(
                            audit_result
                        ),
                        final_output_dir=(
                            final_output
                        ),
                    )

            restored = (
                final_output
                / "previous.csv"
            )

            self.assertTrue(
                restored.exists()
            )

            self.assertEqual(
                restored.read_text(
                    encoding="utf-8"
                ),
                "previous-valid-output\n",
            )

            self.assertFalse(
                (
                    final_output
                    / "results.csv"
                ).exists()
            )

            self.assertEqual(
                state.status,
                RunStatus.FAILED,
            )

    def test_existing_crash_backup_blocks_publish(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            audit_result = (
                _audit_result(
                    root=root,
                    files={
                        "results.csv": (
                            "new-output\n"
                        ),
                    },
                )
            )

            final_output = (
                root
                / "final_output"
            )

            state = (
                _state_at_publish()
            )

            from agent.publish_pipeline import (
                _run_token,
            )

            token = _run_token(
                state.run_id
            )

            backup_dir = (
                final_output.parent
                / (
                    f".{final_output.name}"
                    f".backup.{token}"
                )
            )

            backup_dir.mkdir(
                parents=True
            )

            (
                backup_dir
                / "previous.csv"
            ).write_text(
                "recover-me\n",
                encoding="utf-8",
            )

            with self.assertRaises(
                PublishError
            ):
                run_publish_stage(
                    state=state,
                    audit_result=(
                        audit_result
                    ),
                    final_output_dir=(
                        final_output
                    ),
                )

            self.assertTrue(
                backup_dir.exists()
            )

            self.assertTrue(
                (
                    backup_dir
                    / "previous.csv"
                ).exists()
            )

            self.assertEqual(
                state.status,
                RunStatus.FAILED,
            )

    def test_unaudited_output_is_never_published(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            audit_result = (
                _audit_result(
                    root=root,
                    files={
                        "results.csv": (
                            "bad-output\n"
                        ),
                    },
                    passed=False,
                )
            )

            final_output = (
                root
                / "final_output"
            )

            state = (
                _state_at_publish()
            )

            with self.assertRaises(
                PublishError
            ):
                run_publish_stage(
                    state=state,
                    audit_result=(
                        audit_result
                    ),
                    final_output_dir=(
                        final_output
                    ),
                )

            self.assertFalse(
                final_output.exists()
            )

            self.assertEqual(
                state.status,
                RunStatus.FAILED,
            )


if __name__ == "__main__":
    unittest.main()