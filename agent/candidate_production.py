from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import hashlib

from agent.candidate_contract import CandidateRecord
from agent.candidate_runner import run_candidate
from agent.candidate_workspace import CandidateWorkspace
from agent.code_validator import validate_python_code
from agent.compiled_specification import CompiledSpecification
from agent.executor import ExecutionResult
from agent.orchestrator_state import (
    OrchestratorState,
    PipelineStage,
)
from agent.run_budget import RunBudgetExceeded
from agent.run_context import RunContext
from agent.specification import TaskSpecification


class CandidateProductionError(RuntimeError):
    """Candidate production cannot safely continue."""


@dataclass(frozen=True)
class CandidateGenerationRequest:
    candidate_id: int
    candidate_seed: int

    plan: Any
    specification: TaskSpecification
    compiled_specification: CompiledSpecification


@dataclass(frozen=True)
class GeneratedCandidate:
    code: str

    approach_name: str = "generated"

    strategy: dict[str, Any] = field(
        default_factory=dict
    )

    tokens_used: int = 0

    def __post_init__(self) -> None:
        if self.tokens_used < 0:
            raise ValueError(
                "tokens_used must be non-negative"
            )


GenerationFunction = Callable[
    [CandidateGenerationRequest],
    GeneratedCandidate,
]


@dataclass(frozen=True)
class GeneratorAdapter:
    """
    Provider-agnostic candidate generator.

    `uses_model_budget=True` means the adapter consumes
    the orchestrator's bounded model-call budget.

    The candidate subprocess never receives model
    configuration or credentials.
    """

    generate: GenerationFunction

    uses_model_budget: bool = True

    name: str = "generator"


@dataclass
class CandidateProductionItem:
    candidate: CandidateRecord
    candidate_seed: int

    raw_code: str | None = None
    validated_code: str | None = None

    workspace: CandidateWorkspace | None = None
    solver_path: Path | None = None

    execution: ExecutionResult | None = None


@dataclass
class CandidateProductionResult:
    items: list[CandidateProductionItem]

    @property
    def candidates(self) -> list[CandidateRecord]:
        return [
            item.candidate
            for item in self.items
        ]

    @property
    def executed_items(
        self,
    ) -> list[CandidateProductionItem]:
        return [
            item
            for item in self.items
            if item.execution is not None
        ]


def derive_candidate_seed(
    root_seed: int,
    candidate_id: int,
) -> int:
    """
    Stable deterministic derivation without using
    Python's randomized built-in hash().
    """

    if root_seed < 0:
        raise ValueError(
            "root_seed must be non-negative"
        )

    if candidate_id <= 0:
        raise ValueError(
            "candidate_id must be positive"
        )

    payload = (
        f"{root_seed}:{candidate_id}"
        .encode("utf-8")
    )

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="big",
        signed=False,
    )


def _candidate_environment(
    candidate_seed: int,
) -> dict[str, str]:
    """
    Only variables explicitly permitted by the
    hardened executor are passed to candidate code.
    """

    seed = str(candidate_seed)

    return {
        "QFBENCH_SEED": seed,
        "PYTHONHASHSEED": seed,
    }


def _failed_item(
    *,
    candidate_id: int,
    candidate_seed: int,
    status: str,
    reason: str,
) -> CandidateProductionItem:

    candidate = CandidateRecord(
        candidate_id=candidate_id,
        approach_name=(
            f"candidate-{candidate_id}"
        ),
        strategy={
            "candidate_seed": candidate_seed,
        },
    )

    candidate.current_status = status

    candidate.hard_failures = [
        reason
    ]

    return CandidateProductionItem(
        candidate=candidate,
        candidate_seed=candidate_seed,
    )


def run_candidate_production(
    *,
    task_dir: Path,
    workspace_base_dir: Path,
    state: OrchestratorState,
    run_context: RunContext,
    plan: Any,
    specification: TaskSpecification,
    compiled_specification: CompiledSpecification,
    generator: GeneratorAdapter,
    candidate_count: int = 3,
    execution_timeout_seconds: float = 120.0,
) -> CandidateProductionResult:
    """
    Execute exactly:

        GENERATE
            ->
        VALIDATE_CODE
            ->
        ISOLATED_EXECUTE

    Candidate-local failures do not abort the batch.
    """

    if candidate_count <= 0:
        raise ValueError(
            "candidate_count must be positive"
        )

    task_dir = task_dir.resolve()

    workspace_base_dir = (
        workspace_base_dir.resolve()
    )

    workspace_base_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    items: list[
        CandidateProductionItem
    ] = []

    # ==================================================
    # GENERATE
    # ==================================================

    stage = PipelineStage.GENERATE
    state.start_stage(stage)

    for candidate_id in range(
        1,
        candidate_count + 1,
    ):
        candidate_seed = (
            derive_candidate_seed(
                run_context.qfbench_seed,
                candidate_id,
            )
        )

        if generator.uses_model_budget:
            try:
                run_context.budget.reserve_model_call()

            except RunBudgetExceeded as exc:
                items.append(
                    _failed_item(
                        candidate_id=candidate_id,
                        candidate_seed=candidate_seed,
                        status=(
                            "generation_budget_exhausted"
                        ),
                        reason=str(exc),
                    )
                )
                continue

        request = CandidateGenerationRequest(
            candidate_id=candidate_id,
            candidate_seed=candidate_seed,
            plan=plan,
            specification=specification,
            compiled_specification=(
                compiled_specification
            ),
        )

        try:
            generated = generator.generate(
                request
            )

        except Exception as exc:
            items.append(
                _failed_item(
                    candidate_id=candidate_id,
                    candidate_seed=candidate_seed,
                    status="generation_failed",
                    reason=(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )
            continue

        if generated.tokens_used:
            run_context.budget.record_tokens(
                generated.tokens_used
            )

        strategy = dict(
            generated.strategy
        )

        strategy.update(
            {
                "candidate_seed": candidate_seed,
                "tokens_used": (
                    generated.tokens_used
                ),
                "generator": generator.name,
                "generator_uses_model_budget": (
                    generator.uses_model_budget
                ),
            }
        )

        candidate = CandidateRecord(
            candidate_id=candidate_id,
            approach_name=(
                generated.approach_name
            ),
            strategy=strategy,
        )

        candidate.current_status = "generated"

        items.append(
            CandidateProductionItem(
                candidate=candidate,
                candidate_seed=candidate_seed,
                raw_code=generated.code,
            )
        )

    generated_items = [
        item
        for item in items
        if item.raw_code is not None
    ]

    if not generated_items:
        reason = (
            "No candidate code was generated."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise CandidateProductionError(
            reason
        )

    state.complete_stage(
        stage,
        detail=(
            f"{len(generated_items)} "
            "candidate(s) generated"
        ),
    )

    # ==================================================
    # VALIDATE CODE
    # ==================================================

    stage = PipelineStage.VALIDATE_CODE
    state.start_stage(stage)

    valid_items: list[
        CandidateProductionItem
    ] = []

    for item in generated_items:
        try:
            validated_code = (
                validate_python_code(
                    item.raw_code or ""
                )
            )

        except Exception as exc:
            item.candidate.current_status = (
                "code_invalid"
            )

            item.candidate.hard_failures = [
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            ]

            continue

        item.validated_code = (
            validated_code
        )

        item.candidate.current_status = (
            "code_validated"
        )

        valid_items.append(item)

    if not valid_items:
        reason = (
            "No generated candidate passed "
            "code validation."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise CandidateProductionError(
            reason
        )

    state.complete_stage(
        stage,
        detail=(
            f"{len(valid_items)} candidate(s) "
            "passed code validation"
        ),
    )

    # ==================================================
    # ISOLATED EXECUTE
    # ==================================================

    stage = PipelineStage.ISOLATED_EXECUTE
    state.start_stage(stage)

    executed_count = 0

    for item in valid_items:
        try:
            run_context.budget.reserve_candidate_attempt()

        except RunBudgetExceeded as exc:
            item.candidate.current_status = (
                "execution_budget_exhausted"
            )

            item.candidate.hard_failures = [
                str(exc)
            ]

            continue

        try:
            workspace = (
                CandidateWorkspace.create(
                    base_dir=workspace_base_dir,
                    candidate_id=(
                        item.candidate.candidate_id
                    ),
                    task_dir=task_dir,
                )
            )

            solver_path = (
                workspace.source_dir
                / "solver.py"
            )

            solver_path.write_text(
                item.validated_code or "",
                encoding="utf-8",
            )

            timeout = (
                run_context.bounded_timeout(
                    execution_timeout_seconds
                )
            )

            execution = run_candidate(
                workspace,
                solver_path,
                timeout_seconds=timeout,
                env_overrides=(
                    _candidate_environment(
                        item.candidate_seed
                    )
                ),
            )

        except Exception as exc:
            item.candidate.current_status = (
                "execution_infrastructure_failed"
            )

            item.candidate.hard_failures = [
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            ]

            continue

        item.workspace = workspace
        item.solver_path = solver_path
        item.execution = execution

        # A non-zero return code is NOT judged here.
        # Phase 3 evidence/evaluation owns that decision.
        item.candidate.current_status = (
            "executed_pending_evaluation"
        )

        executed_count += 1

    if executed_count == 0:
        reason = (
            "No validated candidate reached "
            "isolated execution."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise CandidateProductionError(
            reason
        )

    state.complete_stage(
        stage,
        detail=(
            f"{executed_count} candidate(s) "
            "executed in isolated workspaces"
        ),
    )

    return CandidateProductionResult(
        items=items
    )