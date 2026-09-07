from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent.candidate_production import (
    CandidateProductionError,
    GeneratedCandidate,
    GeneratorAdapter,
    derive_candidate_seed,
    run_candidate_production,
)
from agent.orchestrator_state import (
    OrchestratorState,
    PipelineStage,
    RunStatus,
)
from agent.run_context import RunContext
from agent.spec_compiler import (
    compile_specification,
)
from agent.specification import (
    load_task_specification,
)


def _write_task(
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
        task_dir
        / "instruction.md"
    ).write_text(
        """
Produce the required result.

### /output/results.txt
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (
        task_dir
        / "card.toml"
    ).write_text(
        """
[metadata]
task_id = "candidate-production-test"
category = "risk-management"

[environment]
timeout_seconds = 600
""".strip()
        + "\n",
        encoding="utf-8",
    )

    return task_dir


def _state_at_generate() -> OrchestratorState:
    state = OrchestratorState(
        run_id="run-1"
    )

    for stage in (
        PipelineStage.SNAPSHOT,
        PipelineStage.SPECIFICATION,
        PipelineStage.COMPILE,
        PipelineStage.PLAN,
    ):
        state.start_stage(stage)
        state.complete_stage(stage)

    return state


def _load_front_half_objects(
    task_dir: Path,
):
    specification = (
        load_task_specification(
            task_dir
        )
    )

    compiled = (
        compile_specification(
            specification
        )
    )

    return specification, compiled


def _context(
    specification,
    *,
    max_candidate_attempts: int = 3,
    max_model_calls: int = 6,
) -> RunContext:
    return RunContext.from_task_specification(
        specification,
        environ={
            "QFBENCH_SEED": "123",
        },
        max_candidate_attempts=(
            max_candidate_attempts
        ),
        max_model_calls=max_model_calls,
    )


def _good_code() -> str:
    return """
from pathlib import Path
import os

output_dir = Path("output")

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

seed = os.environ["QFBENCH_SEED"]

(
    output_dir
    / "results.txt"
).write_text(
    seed,
    encoding="utf-8",
)
""".strip()


class CandidateProductionTests(
    unittest.TestCase
):

    def test_candidate_seeds_are_deterministic_and_distinct(
        self,
    ) -> None:
        first = [
            derive_candidate_seed(
                123,
                candidate_id,
            )
            for candidate_id in (
                1,
                2,
                3,
            )
        ]

        second = [
            derive_candidate_seed(
                123,
                candidate_id,
            )
            for candidate_id in (
                1,
                2,
                3,
            )
        ]

        self.assertEqual(
            first,
            second,
        )

        self.assertEqual(
            len(set(first)),
            3,
        )

    def test_local_generator_does_not_use_model_budget(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _write_task(
                root
            )

            specification, compiled = (
                _load_front_half_objects(
                    task_dir
                )
            )

            context = _context(
                specification,
                max_model_calls=0,
            )

            generator = GeneratorAdapter(
                generate=(
                    lambda request:
                    GeneratedCandidate(
                        code=_good_code(),
                        approach_name="local",
                    )
                ),
                uses_model_budget=False,
                name="local-generator",
            )

            result = run_candidate_production(
                task_dir=task_dir,
                workspace_base_dir=(
                    root / "candidates"
                ),
                state=_state_at_generate(),
                run_context=context,
                plan={},
                specification=specification,
                compiled_specification=compiled,
                generator=generator,
                candidate_count=3,
            )

            self.assertEqual(
                len(result.executed_items),
                3,
            )

            self.assertEqual(
                context.budget.model_calls_used,
                0,
            )

    def test_model_generator_consumes_model_budget(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _write_task(
                root
            )

            specification, compiled = (
                _load_front_half_objects(
                    task_dir
                )
            )

            context = _context(
                specification,
                max_model_calls=1,
            )

            generator = GeneratorAdapter(
                generate=(
                    lambda request:
                    GeneratedCandidate(
                        code=_good_code(),
                        tokens_used=10,
                    )
                ),
                uses_model_budget=True,
                name="bounded-model-generator",
            )

            result = run_candidate_production(
                task_dir=task_dir,
                workspace_base_dir=(
                    root / "candidates"
                ),
                state=_state_at_generate(),
                run_context=context,
                plan={},
                specification=specification,
                compiled_specification=compiled,
                generator=generator,
                candidate_count=2,
            )

            self.assertEqual(
                context.budget.model_calls_used,
                1,
            )

            self.assertEqual(
                context.budget.tokens_used,
                10,
            )

            self.assertEqual(
                len(result.executed_items),
                1,
            )

            self.assertEqual(
                (
                    result.items[1]
                    .candidate
                    .current_status
                ),
                "generation_budget_exhausted",
            )

    def test_candidate_seed_reaches_isolated_subprocess(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _write_task(
                root
            )

            specification, compiled = (
                _load_front_half_objects(
                    task_dir
                )
            )

            result = run_candidate_production(
                task_dir=task_dir,
                workspace_base_dir=(
                    root / "candidates"
                ),
                state=_state_at_generate(),
                run_context=_context(
                    specification
                ),
                plan={},
                specification=specification,
                compiled_specification=compiled,
                generator=GeneratorAdapter(
                    generate=(
                        lambda request:
                        GeneratedCandidate(
                            code=_good_code()
                        )
                    ),
                    uses_model_budget=False,
                ),
                candidate_count=3,
            )

            self.assertEqual(
                len(result.executed_items),
                3,
            )

            for item in result.executed_items:
                self.assertIsNotNone(
                    item.workspace
                )

                output_path = (
                    item.workspace.output_dir
                    / "results.txt"
                )

                self.assertTrue(
                    output_path.exists()
                )

                self.assertEqual(
                    output_path.read_text(
                        encoding="utf-8"
                    ),
                    str(item.candidate_seed),
                )

    def test_invalid_code_does_not_block_other_candidates(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _write_task(
                root
            )

            specification, compiled = (
                _load_front_half_objects(
                    task_dir
                )
            )

            def generate(request):
                if request.candidate_id == 2:
                    return GeneratedCandidate(
                        code="def broken(:"
                    )

                return GeneratedCandidate(
                    code=_good_code()
                )

            state = _state_at_generate()

            result = run_candidate_production(
                task_dir=task_dir,
                workspace_base_dir=(
                    root / "candidates"
                ),
                state=state,
                run_context=_context(
                    specification
                ),
                plan={},
                specification=specification,
                compiled_specification=compiled,
                generator=GeneratorAdapter(
                    generate=generate,
                    uses_model_budget=False,
                ),
                candidate_count=3,
            )

            self.assertEqual(
                len(result.executed_items),
                2,
            )

            self.assertEqual(
                (
                    result.items[1]
                    .candidate
                    .current_status
                ),
                "code_invalid",
            )

            self.assertEqual(
                state.expected_stage(),
                PipelineStage.COLLECT_EVIDENCE,
            )

    def test_nonzero_exit_is_preserved_for_evaluation(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _write_task(
                root
            )

            specification, compiled = (
                _load_front_half_objects(
                    task_dir
                )
            )

            def generate(request):
                if request.candidate_id == 1:
                    return GeneratedCandidate(
                        code=(
                            "raise RuntimeError("
                            "'public candidate failure'"
                            ")"
                        )
                    )

                return GeneratedCandidate(
                    code=_good_code()
                )

            state = _state_at_generate()

            result = run_candidate_production(
                task_dir=task_dir,
                workspace_base_dir=(
                    root / "candidates"
                ),
                state=state,
                run_context=_context(
                    specification
                ),
                plan={},
                specification=specification,
                compiled_specification=compiled,
                generator=GeneratorAdapter(
                    generate=generate,
                    uses_model_budget=False,
                ),
                candidate_count=2,
            )

            self.assertEqual(
                len(result.executed_items),
                2,
            )

            first_execution = (
                result
                .executed_items[0]
                .execution
            )

            self.assertIsNotNone(
                first_execution
            )

            self.assertNotEqual(
                first_execution.return_code,
                0,
            )

            self.assertEqual(
                state.status,
                RunStatus.RUNNING,
            )

            self.assertEqual(
                state.expected_stage(),
                PipelineStage.COLLECT_EVIDENCE,
            )

    def test_execution_budget_preserves_partial_work(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _write_task(
                root
            )

            specification, compiled = (
                _load_front_half_objects(
                    task_dir
                )
            )

            context = _context(
                specification,
                max_candidate_attempts=1,
            )

            result = run_candidate_production(
                task_dir=task_dir,
                workspace_base_dir=(
                    root / "candidates"
                ),
                state=_state_at_generate(),
                run_context=context,
                plan={},
                specification=specification,
                compiled_specification=compiled,
                generator=GeneratorAdapter(
                    generate=(
                        lambda request:
                        GeneratedCandidate(
                            code=_good_code()
                        )
                    ),
                    uses_model_budget=False,
                ),
                candidate_count=3,
            )

            self.assertEqual(
                len(result.executed_items),
                1,
            )

            self.assertEqual(
                (
                    result
                    .executed_items[0]
                    .candidate
                    .candidate_id
                ),
                1,
            )

            self.assertEqual(
                (
                    result.items[1]
                    .candidate
                    .current_status
                ),
                "execution_budget_exhausted",
            )

            self.assertEqual(
                (
                    result.items[2]
                    .candidate
                    .current_status
                ),
                "execution_budget_exhausted",
            )

    def test_all_invalid_code_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            task_dir = _write_task(
                root
            )

            specification, compiled = (
                _load_front_half_objects(
                    task_dir
                )
            )

            state = _state_at_generate()

            with self.assertRaises(
                CandidateProductionError
            ):
                run_candidate_production(
                    task_dir=task_dir,
                    workspace_base_dir=(
                        root / "candidates"
                    ),
                    state=state,
                    run_context=_context(
                        specification
                    ),
                    plan={},
                    specification=specification,
                    compiled_specification=compiled,
                    generator=GeneratorAdapter(
                        generate=(
                            lambda request:
                            GeneratedCandidate(
                                code="def broken(:"
                            )
                        ),
                        uses_model_budget=False,
                    ),
                    candidate_count=3,
                )

            self.assertEqual(
                state.status,
                RunStatus.FAILED,
            )

            self.assertEqual(
                state.history[-1].stage,
                PipelineStage.VALIDATE_CODE,
            )


if __name__ == "__main__":
    unittest.main()