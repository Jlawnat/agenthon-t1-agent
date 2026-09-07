from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent.candidate_contract import (
    CandidateAttempt,
    CandidateRecord,
    VerificationEvidence,
)
from agent.candidate_production import (
    CandidateProductionItem,
    CandidateProductionResult,
)
from agent.candidate_workspace import (
    CandidateWorkspace,
)
from agent.executor import (
    ExecutionResult,
)
from agent.orchestrator_state import (
    OrchestratorState,
    PipelineStage,
    RunStatus,
)
from agent.selection_pipeline import (
    SelectionPipelineError,
    run_selection_stage,
)


def _state_at_select() -> OrchestratorState:
    state = OrchestratorState(
        run_id="selection-run"
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
    ):
        state.start_stage(
            stage
        )

        state.complete_stage(
            stage
        )

    return state


def _execution(
    *,
    return_code: int = 0,
) -> ExecutionResult:
    return ExecutionResult(
        command=[
            "python",
            "solver.py",
        ],
        return_code=return_code,
        stdout="",
        stderr="",
        runtime_seconds=0.01,
        timed_out=False,
    )


def _pass(
    name: str,
) -> VerificationEvidence:
    return VerificationEvidence(
        name=name,
        status="pass",
        severity="info",
        message="pass",
    )


def _fail(
    name: str,
) -> VerificationEvidence:
    return VerificationEvidence(
        name=name,
        status="fail",
        severity="hard_fail",
        message="fail",
    )


def _candidate(
    *,
    candidate_id: int,
    valid: bool,
    repaired: bool = False,
) -> CandidateRecord:
    candidate = CandidateRecord(
        candidate_id=candidate_id,
        approach_name=(
            f"candidate-{candidate_id}"
        ),
        strategy={},
    )

    first = CandidateAttempt(
        attempt_number=1,
        return_code=(
            0
            if valid
            else 1
        ),
        runtime_seconds=0.01,
    )

    if valid:
        first.structural_evidence.extend(
            [
                _pass(
                    "execution"
                ),
                _pass(
                    "schema"
                ),
            ]
        )

    else:
        first.structural_evidence.append(
            _fail(
                "execution"
            )
        )

    candidate.add_attempt(
        first
    )

    if repaired:
        repaired_attempt = (
            CandidateAttempt(
                attempt_number=2,
                return_code=0,
                runtime_seconds=0.01,
                repair_reason=(
                    "repair_runtime_exception"
                ),
            )
        )

        repaired_attempt.structural_evidence.extend(
            [
                _pass(
                    "execution"
                ),
                _pass(
                    "schema"
                ),
            ]
        )

        candidate.add_attempt(
            repaired_attempt
        )

    return candidate


def _item(
    *,
    root: Path,
    candidate: CandidateRecord,
) -> CandidateProductionItem:
    workspace = (
        CandidateWorkspace.create(
            base_dir=(
                root
                / "workspaces"
            ),
            candidate_id=(
                candidate.candidate_id
            ),
            task_dir=(
                root
                / "task"
            ),
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

    latest = (
        candidate.latest_attempt()
    )

    return CandidateProductionItem(
        candidate=candidate,
        candidate_seed=(
            1000
            + candidate.candidate_id
        ),
        raw_code=(
            "print('ok')"
        ),
        validated_code=(
            "print('ok')"
        ),
        workspace=workspace,
        solver_path=solver_path,
        execution=_execution(
            return_code=(
                latest.return_code
                if latest is not None
                else 1
            )
        ),
    )


class SelectionPipelineTests(
    unittest.TestCase
):

    def test_valid_candidate_survives_partial_failures(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            (
                root
                / "task"
                / "environment"
                / "data"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            failed = _candidate(
                candidate_id=1,
                valid=False,
            )

            valid = _candidate(
                candidate_id=2,
                valid=True,
            )

            production = (
                CandidateProductionResult(
                    items=[
                        _item(
                            root=root,
                            candidate=failed,
                        ),
                        _item(
                            root=root,
                            candidate=valid,
                        ),
                    ]
                )
            )

            state = (
                _state_at_select()
            )

            result = (
                run_selection_stage(
                    state=state,
                    production=production,
                )
            )

            self.assertEqual(
                result
                .selected_candidate_id,
                2,
            )

            self.assertFalse(
                failed.selected
            )

            self.assertTrue(
                valid.selected
            )

            self.assertEqual(
                state.expected_stage(),
                PipelineStage
                .CLEAN_ROOM_AUDIT,
            )

    def test_deterministic_tie_selects_lower_candidate_id(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            (
                root
                / "task"
                / "environment"
                / "data"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            one = _candidate(
                candidate_id=1,
                valid=True,
            )

            two = _candidate(
                candidate_id=2,
                valid=True,
            )

            production = (
                CandidateProductionResult(
                    items=[
                        _item(
                            root=root,
                            candidate=two,
                        ),
                        _item(
                            root=root,
                            candidate=one,
                        ),
                    ]
                )
            )

            result = (
                run_selection_stage(
                    state=(
                        _state_at_select()
                    ),
                    production=production,
                )
            )

            self.assertEqual(
                result
                .selected_candidate_id,
                1,
            )

    def test_repaired_candidate_can_be_selected(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            (
                root
                / "task"
                / "environment"
                / "data"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            repaired = (
                _candidate(
                    candidate_id=1,
                    valid=False,
                    repaired=True,
                )
            )

            failed = _candidate(
                candidate_id=2,
                valid=False,
            )

            production = (
                CandidateProductionResult(
                    items=[
                        _item(
                            root=root,
                            candidate=repaired,
                        ),
                        _item(
                            root=root,
                            candidate=failed,
                        ),
                    ]
                )
            )

            result = (
                run_selection_stage(
                    state=(
                        _state_at_select()
                    ),
                    production=production,
                )
            )

            self.assertEqual(
                result
                .selected_candidate_id,
                1,
            )

            self.assertEqual(
                len(
                    repaired.attempts
                ),
                2,
            )

            self.assertTrue(
                repaired.selected
            )

    def test_pass_at_one_and_any_of_three_are_preserved(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            (
                root
                / "task"
                / "environment"
                / "data"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            first = _candidate(
                candidate_id=1,
                valid=False,
            )

            second = _candidate(
                candidate_id=2,
                valid=True,
            )

            production = (
                CandidateProductionResult(
                    items=[
                        _item(
                            root=root,
                            candidate=first,
                        ),
                        _item(
                            root=root,
                            candidate=second,
                        ),
                    ]
                )
            )

            result = (
                run_selection_stage(
                    state=(
                        _state_at_select()
                    ),
                    production=production,
                )
            )

            self.assertFalse(
                result
                .selection
                .first_candidate_valid
            )

            self.assertTrue(
                result
                .selection
                .any_of_three_valid
            )

    def test_no_valid_candidate_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            (
                root
                / "task"
                / "environment"
                / "data"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            production = (
                CandidateProductionResult(
                    items=[
                        _item(
                            root=root,
                            candidate=(
                                _candidate(
                                    candidate_id=1,
                                    valid=False,
                                )
                            ),
                        ),
                        _item(
                            root=root,
                            candidate=(
                                _candidate(
                                    candidate_id=2,
                                    valid=False,
                                )
                            ),
                        ),
                    ]
                )
            )

            state = (
                _state_at_select()
            )

            with self.assertRaises(
                SelectionPipelineError
            ):
                run_selection_stage(
                    state=state,
                    production=production,
                )

            self.assertEqual(
                state.status,
                RunStatus.FAILED,
            )

            self.assertEqual(
                state.history[-1].stage,
                PipelineStage.SELECT,
            )

    def test_selected_candidate_requires_current_artifacts(
        self,
    ) -> None:
        candidate = _candidate(
            candidate_id=1,
            valid=True,
        )

        production = (
            CandidateProductionResult(
                items=[
                    CandidateProductionItem(
                        candidate=candidate,
                        candidate_seed=1001,
                    )
                ]
            )
        )

        state = (
            _state_at_select()
        )

        with self.assertRaises(
            SelectionPipelineError
        ):
            run_selection_stage(
                state=state,
                production=production,
            )

        self.assertEqual(
            state.status,
            RunStatus.FAILED,
        )


if __name__ == "__main__":
    unittest.main()