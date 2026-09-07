from __future__ import annotations

import unittest

from agent.orchestrator_state import (
    InvalidStateTransition,
    OrchestratorState,
    PIPELINE_ORDER,
    PipelineStage,
    RunStatus,
)


class OrchestratorStateTests(unittest.TestCase):

    def test_pipeline_order_matches_phase_4_contract(
        self,
    ) -> None:
        self.assertEqual(
            [
                stage.value
                for stage in PIPELINE_ORDER
            ],
            [
                "snapshot",
                "specification",
                "compile",
                "plan",
                "generate",
                "validate_code",
                "isolated_execute",
                "collect_evidence",
                "structural_quant_evaluate",
                "targeted_repair",
                "select",
                "clean_room_audit",
                "publish",
            ],
        )

    def test_run_starts_at_snapshot(
        self,
    ) -> None:
        state = OrchestratorState(
            run_id="run-1"
        )

        self.assertEqual(
            state.expected_stage(),
            PipelineStage.SNAPSHOT,
        )

        self.assertEqual(
            state.status,
            RunStatus.NOT_STARTED,
        )

    def test_stages_must_execute_in_exact_order(
        self,
    ) -> None:
        state = OrchestratorState(
            run_id="run-1"
        )

        with self.assertRaises(
            InvalidStateTransition
        ):
            state.start_stage(
                PipelineStage.PLAN
            )

        state.start_stage(
            PipelineStage.SNAPSHOT
        )

        state.complete_stage(
            PipelineStage.SNAPSHOT
        )

        self.assertEqual(
            state.expected_stage(),
            PipelineStage.SPECIFICATION,
        )

    def test_cannot_start_new_stage_while_one_is_active(
        self,
    ) -> None:
        state = OrchestratorState(
            run_id="run-1"
        )

        state.start_stage(
            PipelineStage.SNAPSHOT
        )

        with self.assertRaises(
            InvalidStateTransition
        ):
            state.start_stage(
                PipelineStage.SPECIFICATION
            )

    def test_completion_records_history(
        self,
    ) -> None:
        state = OrchestratorState(
            run_id="run-1"
        )

        state.start_stage(
            PipelineStage.SNAPSHOT
        )

        state.complete_stage(
            PipelineStage.SNAPSHOT,
            detail="snapshot created",
        )

        self.assertEqual(
            state.completed_stages,
            [PipelineStage.SNAPSHOT],
        )

        self.assertEqual(
            [
                event.status
                for event in state.history
            ],
            [
                "started",
                "completed",
            ],
        )

        self.assertEqual(
            state.history[-1].detail,
            "snapshot created",
        )

    def test_stage_failure_fails_closed(
        self,
    ) -> None:
        state = OrchestratorState(
            run_id="run-1"
        )

        state.start_stage(
            PipelineStage.SNAPSHOT
        )

        state.fail_stage(
            PipelineStage.SNAPSHOT,
            reason="snapshot failed",
        )

        self.assertEqual(
            state.status,
            RunStatus.FAILED,
        )

        self.assertEqual(
            state.failure_reason,
            "snapshot failed",
        )

        self.assertIsNone(
            state.current_stage
        )

        with self.assertRaises(
            InvalidStateTransition
        ):
            state.start_stage(
                PipelineStage.SNAPSHOT
            )

    def test_full_pipeline_finishes_succeeded(
        self,
    ) -> None:
        state = OrchestratorState(
            run_id="run-1"
        )

        for stage in PIPELINE_ORDER:
            state.start_stage(stage)
            state.complete_stage(stage)

        self.assertEqual(
            state.status,
            RunStatus.SUCCEEDED,
        )

        self.assertIsNone(
            state.expected_stage()
        )

        self.assertEqual(
            state.completed_stages,
            list(PIPELINE_ORDER),
        )

    def test_state_serialization_uses_plain_values(
        self,
    ) -> None:
        state = OrchestratorState(
            run_id="run-1",
            task_id="task-1",
        )

        state.start_stage(
            PipelineStage.SNAPSHOT
        )

        state.complete_stage(
            PipelineStage.SNAPSHOT
        )

        payload = state.to_dict()

        self.assertEqual(
            payload["run_id"],
            "run-1",
        )

        self.assertEqual(
            payload["task_id"],
            "task-1",
        )

        self.assertEqual(
            payload["status"],
            "running",
        )

        self.assertEqual(
            payload["completed_stages"],
            ["snapshot"],
        )

        self.assertEqual(
            payload["history"][0]["stage"],
            "snapshot",
        )


if __name__ == "__main__":
    unittest.main()