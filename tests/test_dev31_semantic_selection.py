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
from agent.candidate_workspace import CandidateWorkspace
from agent.compiled_specification import CompiledSpecification
from agent.executor import ExecutionResult
from agent.orchestrator_state import (
    OrchestratorState,
    PipelineStage,
)
from agent.run_budget import RunBudget
from agent.run_context import RunContext
from agent.selection_pipeline import run_selection_stage
from agent.semantic_selection import (
    SemanticSelectionResult,
    parse_semantic_selection_response,
)
from agent.specification import TaskSpecification


def _state_at_select() -> OrchestratorState:
    state = OrchestratorState(run_id="semantic-select")

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
        state.start_stage(stage)
        state.complete_stage(stage)

    return state


def _candidate(
    *,
    candidate_id: int,
    root: Path,
) -> CandidateProductionItem:
    candidate = CandidateRecord(
        candidate_id=candidate_id,
        approach_name=f"candidate-{candidate_id}",
        strategy={},
    )

    attempt = CandidateAttempt(
        attempt_number=1,
        return_code=0,
        timed_out=False,
        runtime_seconds=1.0,
    )

    attempt.structural_evidence.extend(
        [
            VerificationEvidence(
                name="execution",
                status="pass",
                severity="info",
                message="executed",
            ),
            VerificationEvidence(
                name="required_outputs",
                status="pass",
                severity="info",
                message="outputs exist",
            ),
        ]
    )

    candidate.add_attempt(attempt)
    candidate.current_status = "validated"

    workspace = CandidateWorkspace.create(
        base_dir=root / "workspaces",
        candidate_id=candidate_id,
        task_dir=root / "task",
    )

    solver_path = workspace.source_dir / "solver.py"

    source = (
        "from pathlib import Path\n"
        f"# candidate {candidate_id}\n"
        "print('ok')\n"
    )

    solver_path.write_text(source, encoding="utf-8")

    execution = ExecutionResult(
        command=["python", "solver.py"],
        return_code=0,
        stdout="",
        stderr="",
        runtime_seconds=1.0,
        timed_out=False,
    )

    return CandidateProductionItem(
        candidate=candidate,
        candidate_seed=candidate_id,
        raw_code=source,
        validated_code=source,
        workspace=workspace,
        solver_path=solver_path,
        execution=execution,
    )


def _specification() -> TaskSpecification:
    return TaskSpecification(
        task_id="semantic-test",
        category="risk-management",
        instruction_text=(
            "Compute the requested portfolio risk measure exactly "
            "and write results.json."
        ),
        required_output_paths=["results.json"],
        candidate_schema_fields=[],
        required_output_columns=[],
        runtime_seconds=600.0,
        network_mode="restricted",
        raw_card={},
        required_output_dtypes={},
    )


def _compiled() -> CompiledSpecification:
    return CompiledSpecification(
        task_id="semantic-test",
        category="risk-management",
        difficulty="medium",
        deliverables=[{"path": "results.json", "format": "json"}],
        required_columns=[],
        required_row_rules=[],
        ordering_rules=[],
        units={},
        conventions=[],
        edge_cases=[],
        invariants=[],
        review_packs=["risk-management"],
        numerical_risks=[],
        assumptions=[],
        unresolved_questions=[],
        required_dtypes={},
    )


class SemanticSelectionTests(unittest.TestCase):
    def test_parser_accepts_reasoning_preamble_and_rejects_ineligible_id(
        self,
    ) -> None:
        result = parse_semantic_selection_response(
            (
                "<think>compare</think>\n"
                "Review complete. "
                '{"selected_candidate_id": 2, '
                '"confidence": "high", '
                '"rationale": ["matches the stated convention"]}'
            ),
            eligible_ids=(1, 2),
            tokens_used=123,
        )

        self.assertEqual(result.selected_candidate_id, 2)
        self.assertEqual(result.tokens_used, 123)

        with self.assertRaises(ValueError):
            parse_semantic_selection_response(
                (
                    '{"selected_candidate_id": 9, '
                    '"confidence": "high", '
                    '"rationale": ["no"]}'
                ),
                eligible_ids=(1, 2),
            )

    def test_semantic_choice_can_override_deterministic_tie(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "task").mkdir()

            first = _candidate(candidate_id=1, root=root)
            second = _candidate(candidate_id=2, root=root)

            production = CandidateProductionResult(
                items=[first, second]
            )

            context = RunContext(
                task_id="semantic-test",
                card_timeout_seconds=600.0,
                qfbench_seed=42,
                budget=RunBudget(
                    total_wall_seconds=600.0,
                    max_model_calls=12,
                ),
            )

            seen = {}

            def compare(request):
                seen["ids"] = request.eligible_ids()

                return SemanticSelectionResult(
                    selected_candidate_id=2,
                    confidence="high",
                    rationale=(
                        "candidate 2 follows the explicit finance convention",
                    ),
                    tokens_used=321,
                )

            result = run_selection_stage(
                state=_state_at_select(),
                production=production,
                semantic_compare=compare,
                semantic_compare_uses_model_budget=True,
                run_context=context,
                specification=_specification(),
                compiled_specification=_compiled(),
            )

            self.assertEqual(seen["ids"], (1, 2))
            self.assertEqual(result.selected_candidate_id, 2)
            self.assertFalse(first.candidate.selected)
            self.assertTrue(second.candidate.selected)
            self.assertEqual(
                result.selection.ranked_candidate_ids[0],
                2,
            )
            self.assertEqual(
                context.budget.model_calls_used,
                1,
            )
            self.assertEqual(
                context.budget.tokens_used,
                321,
            )

    def test_semantic_failure_falls_back_to_deterministic_selector(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "task").mkdir()

            first = _candidate(candidate_id=1, root=root)
            second = _candidate(candidate_id=2, root=root)

            def compare(_request):
                raise RuntimeError("simulated model failure")

            result = run_selection_stage(
                state=_state_at_select(),
                production=CandidateProductionResult(
                    items=[first, second]
                ),
                semantic_compare=compare,
                semantic_compare_uses_model_budget=False,
                specification=_specification(),
                compiled_specification=_compiled(),
            )

            self.assertEqual(result.selected_candidate_id, 1)

    def test_semantic_review_is_not_called_when_only_one_candidate_is_valid(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "task").mkdir()

            first = _candidate(candidate_id=1, root=root)
            second = _candidate(candidate_id=2, root=root)

            second.candidate.latest_attempt().structural_evidence.append(
                VerificationEvidence(
                    name="schema",
                    status="fail",
                    severity="hard_fail",
                    message="bad schema",
                )
            )

            calls = []

            def compare(_request):
                calls.append(True)
                return SemanticSelectionResult(
                    selected_candidate_id=1,
                    confidence="high",
                    rationale=("unused",),
                )

            result = run_selection_stage(
                state=_state_at_select(),
                production=CandidateProductionResult(
                    items=[first, second]
                ),
                semantic_compare=compare,
                semantic_compare_uses_model_budget=False,
                specification=_specification(),
                compiled_specification=_compiled(),
            )

            self.assertEqual(calls, [])
            self.assertEqual(result.selected_candidate_id, 1)


if __name__ == "__main__":
    unittest.main()
