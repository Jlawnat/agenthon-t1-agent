from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import time

from agent.budget import RepairBudget, RepairBudgetExceeded
from agent.candidate_contract import CandidateAttempt, VerificationEvidence
from agent.candidate_evaluator import evaluate_collected_attempt
from agent.candidate_production import CandidateProductionItem, CandidateProductionResult
from agent.candidate_runner import run_candidate
from agent.candidate_workspace import CandidateWorkspace
from agent.code_validator import validate_python_code
from agent.compiled_specification import CompiledSpecification
from agent.evidence_collector import collect_execution_evidence
from agent.orchestrator_state import OrchestratorState, PipelineStage
from agent.quality_pipeline import InitialQualityResult
from agent.run_budget import RunBudgetExceeded
from agent.run_context import RunContext
from agent.semantic_selection import (
    SemanticSelectionRequest,
    SemanticSelectionResult,
)
from agent.specification import TaskSpecification
from agent.targeted_repair import RepairBrief, build_repair_brief, classify_failure


@dataclass(frozen=True)
class RepairRequest:
    candidate_id: int
    candidate_seed: int
    source_code: str
    brief: RepairBrief
    specification: TaskSpecification
    compiled_specification: CompiledSpecification
    data_inspections: dict[str, dict[str, Any]] | None = None


@dataclass(frozen=True)
class RepairedCandidate:
    code: str
    tokens_used: int = 0

    def __post_init__(self) -> None:
        if self.tokens_used < 0:
            raise ValueError("tokens_used must be non-negative")


RepairFunction = Callable[[RepairRequest], RepairedCandidate]
SemanticSelectionFunction = Callable[
    [SemanticSelectionRequest],
    SemanticSelectionResult,
]


@dataclass(frozen=True)
class RepairAdapter:
    repair: RepairFunction
    uses_model_budget: bool = True
    name: str = "repair-generator"
    semantic_compare: SemanticSelectionFunction | None = None
    semantic_compare_uses_model_budget: bool = True


@dataclass
class TargetedRepairResult:
    production: CandidateProductionResult
    repair_budget: RepairBudget
    attempted_candidate_ids: tuple[int, ...]
    repaired_candidate_ids: tuple[int, ...]
    skipped_candidate_ids: tuple[int, ...]
    failure_messages: dict[int, str]


def _candidate_environment(candidate_seed: int) -> dict[str, str]:
    seed = str(candidate_seed)
    return {
        "QFBENCH_SEED": seed,
        "PYTHONHASHSEED": seed,
    }


def _bounded_error(exc: Exception | str) -> str:
    text = str(exc)
    if len(text) <= 1000:
        return text
    return text[:985] + "...[truncated]"


def _attempt_quality_key(
    attempt: CandidateAttempt,
) -> tuple[int, int, int, int]:
    """Lower is better; repairs must strictly improve this key."""
    admissible_penalty = (
        0
        if (
            not attempt.timed_out
            and attempt.return_code == 0
        )
        else 1
    )

    structural_hard_failures = sum(
        1
        for item in attempt.structural_evidence
        if (
            item.status == "fail"
            and item.severity == "hard_fail"
        )
    )

    quant_hard_failures = sum(
        1
        for item in attempt.quant_evidence
        if (
            item.status == "fail"
            and item.severity == "hard_fail"
        )
    )

    warnings = sum(
        1
        for item in (
            attempt.structural_evidence
            + attempt.quant_evidence
        )
        if (
            item.status == "warning"
            or item.severity == "warning"
        )
    )

    return (
        admissible_penalty,
        structural_hard_failures,
        quant_hard_failures,
        warnings,
    )


def _code_validation_attempt(
    *,
    item: CandidateProductionItem,
    error: str,
    repair_reason: str | None,
    repaired: bool,
) -> CandidateAttempt:
    candidate = item.candidate
    attempt = CandidateAttempt(
        attempt_number=len(candidate.attempts) + 1,
        return_code=None,
        timed_out=False,
        stderr=error,
        repair_reason=repair_reason,
    )
    attempt.structural_evidence.append(
        VerificationEvidence(
            name="repair_code_validation" if repaired else "code_validation",
            status="fail",
            severity="hard_fail",
            message=(
                "Repaired candidate code failed validation."
                if repaired
                else "Generated candidate code failed validation."
            ),
            details={"error": error[:1000]},
        )
    )
    candidate.add_attempt(attempt)
    candidate.current_status = "repair_code_invalid" if repaired else "code_invalid"
    candidate.hard_failures = [error]
    return attempt


def _ensure_initial_validation_attempt(item: CandidateProductionItem) -> None:
    candidate = item.candidate
    if candidate.current_status != "code_invalid":
        return
    if candidate.latest_attempt() is not None:
        return

    error = (
        candidate.hard_failures[0]
        if candidate.hard_failures
        else "Candidate code failed validation."
    )

    _code_validation_attempt(
        item=item,
        error=error,
        repair_reason=None,
        repaired=False,
    )


def _record_repair_strategy(
    *,
    item: CandidateProductionItem,
    brief: RepairBrief,
    repairer: RepairAdapter,
    tokens_used: int,
) -> None:
    strategy = item.candidate.strategy
    previous_tokens = strategy.get("tokens_used", 0)

    if isinstance(previous_tokens, bool) or not isinstance(previous_tokens, (int, float)):
        previous_tokens = 0

    strategy["tokens_used"] = max(0, int(previous_tokens)) + tokens_used

    history = strategy.get("repair_strategies")
    if not isinstance(history, list):
        history = []

    history = list(history)
    history.append(brief.strategy)

    strategy["repair_strategies"] = history
    strategy["last_repair_failure_label"] = brief.failure_label
    strategy["repair_generator"] = repairer.name


def _repair_evaluation_error(exc: Exception) -> VerificationEvidence:
    return VerificationEvidence(
        name="repair_evaluation_infrastructure_error",
        status="fail",
        severity="hard_fail",
        message="Repaired candidate evaluation failed unexpectedly.",
        details={
            "error_type": type(exc).__name__,
            "error": _bounded_error(exc),
        },
    )


@dataclass(frozen=True)
class _RepairOutcome:
    attempted: bool
    validated: bool
    improved: bool
    failure: str | None = None


def _validation_error_for_candidate(
    item: CandidateProductionItem,
) -> str | None:
    candidate = item.candidate

    if candidate.current_status != "code_invalid":
        return None

    attempt = candidate.latest_attempt()

    return (
        candidate.hard_failures[0]
        if candidate.hard_failures
        else (
            attempt.stderr
            if attempt is not None
            else None
        )
    )


def _repair_priority_key(
    item: CandidateProductionItem,
) -> tuple[int, int, int, int, int]:
    attempt = item.candidate.latest_attempt()

    if attempt is None:
        return (
            9,
            9,
            9,
            9,
            item.candidate.candidate_id,
        )

    return (
        *_attempt_quality_key(attempt),
        item.candidate.candidate_id,
    )


def _attempt_is_publishable(
    attempt: CandidateAttempt | None,
) -> bool:
    if attempt is None:
        return False

    quality = _attempt_quality_key(
        attempt
    )

    return (
        quality[0] == 0
        and quality[1] == 0
        and quality[2] == 0
    )


def _repair_one_item(
    *,
    item: CandidateProductionItem,
    task_dir: Path,
    repair_workspace_base_dir: Path,
    run_context: RunContext,
    quality: InitialQualityResult,
    specification: TaskSpecification,
    compiled_specification: CompiledSpecification,
    repairer: RepairAdapter,
    repair_budget: RepairBudget,
    execution_timeout_seconds: float,
    data_inspections: dict[str, dict[str, Any]] | None,
) -> _RepairOutcome:
    candidate = item.candidate
    candidate_id = candidate.candidate_id

    _ensure_initial_validation_attempt(
        item
    )

    source_attempt = candidate.latest_attempt()

    if source_attempt is None:
        return _RepairOutcome(
            attempted=False,
            validated=False,
            improved=False,
            failure=(
                "No candidate attempt was available for repair."
            ),
        )

    validation_error = (
        _validation_error_for_candidate(
            item
        )
    )

    failure_label = classify_failure(
        source_attempt,
        validation_error=validation_error,
    )

    if failure_label is None:
        return _RepairOutcome(
            attempted=False,
            validated=False,
            improved=False,
        )

    source_code = (
        item.validated_code
        or item.raw_code
    )

    if not source_code:
        return _RepairOutcome(
            attempted=False,
            validated=False,
            improved=False,
            failure=(
                "No candidate source code was available for repair."
            ),
        )

    source_attempt_count = len(
        candidate.attempts
    )
    source_status = (
        candidate.current_status
    )
    source_hard_failures = list(
        candidate.hard_failures
    )
    source_warnings = list(
        candidate.warnings
    )
    source_validated_code = (
        item.validated_code
    )
    source_workspace = (
        item.workspace
    )
    source_solver_path = (
        item.solver_path
    )
    source_execution = (
        item.execution
    )

    def restore_source() -> None:
        del candidate.attempts[
            source_attempt_count:
        ]

        candidate.current_status = (
            source_status
        )
        candidate.hard_failures = (
            source_hard_failures
        )
        candidate.warnings = (
            source_warnings
        )

        item.validated_code = (
            source_validated_code
        )
        item.workspace = (
            source_workspace
        )
        item.solver_path = (
            source_solver_path
        )
        item.execution = (
            source_execution
        )

    if not repair_budget.can_reserve():
        return _RepairOutcome(
            attempted=False,
            validated=False,
            improved=False,
            failure=(
                "Targeted repair budget is exhausted."
            ),
        )

    if (
        repairer.uses_model_budget
        and run_context.budget.remaining_model_calls <= 0
    ):
        return _RepairOutcome(
            attempted=False,
            validated=False,
            improved=False,
            failure=(
                "Global model-call budget is exhausted."
            ),
        )

    if run_context.budget.remaining_candidate_attempts <= 0:
        return _RepairOutcome(
            attempted=False,
            validated=False,
            improved=False,
            failure=(
                "Global candidate-attempt budget is exhausted."
            ),
        )

    try:
        brief = build_repair_brief(
            attempt=source_attempt,
            budget=repair_budget,
            validation_error=validation_error,
        )
    except RepairBudgetExceeded as exc:
        return _RepairOutcome(
            attempted=False,
            validated=False,
            improved=False,
            failure=_bounded_error(exc),
        )
    except Exception as exc:
        return _RepairOutcome(
            attempted=False,
            validated=False,
            improved=False,
            failure=(
                "Repair brief construction failed: "
                f"{_bounded_error(exc)}"
            ),
        )

    if brief is None:
        return _RepairOutcome(
            attempted=False,
            validated=False,
            improved=False,
        )

    if repairer.uses_model_budget:
        try:
            run_context.budget.reserve_model_call()
        except RunBudgetExceeded as exc:
            return _RepairOutcome(
                attempted=False,
                validated=False,
                improved=False,
                failure=_bounded_error(exc),
            )

    request = RepairRequest(
        candidate_id=candidate_id,
        candidate_seed=item.candidate_seed,
        source_code=source_code,
        brief=brief,
        specification=specification,
        compiled_specification=(
            compiled_specification
        ),
        data_inspections=(
            data_inspections
        ),
    )

    repair_start = time.perf_counter()

    try:
        repaired_candidate = repairer.repair(
            request
        )
    except Exception as exc:
        elapsed = (
            time.perf_counter()
            - repair_start
        )

        repair_budget.record_usage(
            tokens=0,
            wall_seconds=elapsed,
        )

        candidate.current_status = (
            "repair_generation_failed"
        )

        return _RepairOutcome(
            attempted=True,
            validated=False,
            improved=False,
            failure=_bounded_error(exc),
        )

    repair_elapsed = (
        time.perf_counter()
        - repair_start
    )

    repair_budget.record_usage(
        tokens=repaired_candidate.tokens_used,
        wall_seconds=repair_elapsed,
    )

    if repaired_candidate.tokens_used:
        run_context.budget.record_tokens(
            repaired_candidate.tokens_used
        )

    _record_repair_strategy(
        item=item,
        brief=brief,
        repairer=repairer,
        tokens_used=(
            repaired_candidate.tokens_used
        ),
    )

    try:
        validated_code = validate_python_code(
            repaired_candidate.code
        )
    except Exception as exc:
        error = (
            f"{type(exc).__name__}: "
            f"{_bounded_error(exc)}"
        )

        _code_validation_attempt(
            item=item,
            error=error,
            repair_reason=brief.strategy,
            repaired=True,
        )

        # A repair is advisory. Invalid repaired code must never
        # replace a better executable source attempt.
        restore_source()

        return _RepairOutcome(
            attempted=True,
            validated=False,
            improved=False,
            failure=error,
        )

    try:
        run_context.budget.reserve_candidate_attempt()
    except RunBudgetExceeded as exc:
        candidate.current_status = (
            "repair_execution_budget_exhausted"
        )

        return _RepairOutcome(
            attempted=True,
            validated=False,
            improved=False,
            failure=_bounded_error(exc),
        )

    next_attempt_number = (
        len(candidate.attempts) + 1
    )

    repair_attempt_base = (
        repair_workspace_base_dir
        / f"attempt_{next_attempt_number}"
    )

    try:
        workspace = CandidateWorkspace.create(
            base_dir=repair_attempt_base,
            candidate_id=candidate_id,
            task_dir=task_dir,
        )

        solver_path = (
            workspace.source_dir
            / "solver.py"
        )

        solver_path.write_text(
            validated_code,
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
        candidate.current_status = (
            "repair_execution_infrastructure_failed"
        )

        return _RepairOutcome(
            attempted=True,
            validated=False,
            improved=False,
            failure=_bounded_error(exc),
        )

    item.validated_code = (
        validated_code
    )
    item.workspace = (
        workspace
    )
    item.solver_path = (
        solver_path
    )
    item.execution = (
        execution
    )

    attempt = collect_execution_evidence(
        execution=execution,
        output_dir=workspace.output_dir,
        required_output_paths=(
            specification.required_output_paths
        ),
        attempt_number=(
            next_attempt_number
        ),
        solver_path=solver_path,
    )

    attempt.repair_reason = (
        brief.strategy
    )

    try:
        evaluate_collected_attempt(
            candidate=candidate,
            attempt=attempt,
            output_dir=workspace.output_dir,
            required_output_paths=(
                specification
                .required_output_paths
            ),
            required_columns=(
                compiled_specification
                .required_columns
            ),
            schema_expectations=(
                quality.schema_expectations
            ),
            compiled_specification=(
                compiled_specification
            ),
        )
    except Exception as exc:
        attempt.structural_evidence.append(
            _repair_evaluation_error(
                exc
            )
        )

        if (
            candidate.latest_attempt()
            is not attempt
        ):
            candidate.add_attempt(
                attempt
            )

        # Evaluation failure is not allowed to destroy the
        # previously usable candidate state.
        restore_source()

        return _RepairOutcome(
            attempted=True,
            validated=False,
            improved=False,
            failure=_bounded_error(exc),
        )

    repaired_attempt = (
        candidate.latest_attempt()
    )

    if repaired_attempt is None:
        restore_source()

        return _RepairOutcome(
            attempted=True,
            validated=False,
            improved=False,
            failure=(
                "Repaired candidate produced no evaluated attempt."
            ),
        )

    if (
        _attempt_quality_key(
            repaired_attempt
        )
        >= _attempt_quality_key(
            source_attempt
        )
    ):
        restore_source()

        return _RepairOutcome(
            attempted=True,
            validated=False,
            improved=False,
            failure=(
                "Repair was not a strict improvement; "
                "restored the previous candidate attempt."
            ),
        )

    validated = (
        _attempt_is_publishable(
            repaired_attempt
        )
    )

    return _RepairOutcome(
        attempted=True,
        validated=validated,
        improved=True,
    )

def run_targeted_repair_stage(
    *,
    task_dir: Path,
    repair_workspace_base_dir: Path,
    state: OrchestratorState,
    run_context: RunContext,
    quality: InitialQualityResult,
    specification: TaskSpecification,
    compiled_specification: CompiledSpecification,
    repairer: RepairAdapter,
    repair_budget: RepairBudget | None = None,
    execution_timeout_seconds: float = 120.0,
    data_inspections: dict[str, dict[str, Any]] | None = None,
) -> TargetedRepairResult:
    task_dir = task_dir.resolve()
    repair_workspace_base_dir = repair_workspace_base_dir.resolve()
    repair_workspace_base_dir.mkdir(parents=True, exist_ok=True)

    if repair_budget is None:
        repair_budget = RepairBudget(
            max_attempts=3,
            max_total_tokens=12000,
            max_wall_seconds=min(
                240.0,
                run_context.budget.usable_remaining_seconds,
            ),
        )

    stage = PipelineStage.TARGETED_REPAIR
    state.start_stage(stage)

    attempted: list[int] = []
    repaired: list[int] = []
    skipped: list[int] = []
    failures: dict[int, str] = {}

    all_items = list(
        quality.production.items
    )

    nonrepairable_ids: list[int] = []
    repairable_items: list[
        CandidateProductionItem
    ] = []

    for item in all_items:
        _ensure_initial_validation_attempt(
            item
        )

        attempt = (
            item.candidate
            .latest_attempt()
        )

        if attempt is None:
            nonrepairable_ids.append(
                item.candidate.candidate_id
            )
            continue

        failure_label = classify_failure(
            attempt,
            validation_error=(
                _validation_error_for_candidate(
                    item
                )
            ),
        )

        if failure_label is None:
            nonrepairable_ids.append(
                item.candidate.candidate_id
            )
            continue

        repairable_items.append(
            item
        )

    repairable_items.sort(
        key=_repair_priority_key
    )

    skipped.extend(
        nonrepairable_ids
    )

    # Preserve one of the default three repair slots for a
    # publication-rescue attempt. For three-candidate hard tasks,
    # this means two closest candidates get a first repair, then the
    # reserved slot either deepens an improving candidate or falls
    # back to the next untried candidate.
    available_slots = (
        repair_budget.remaining_attempts
    )

    first_wave_limit = min(
        len(repairable_items),
        max(
            1,
            available_slots - 1,
        )
        if available_slots > 0
        else 0,
    )

    first_wave = (
        repairable_items[
            :first_wave_limit
        ]
    )

    deferred = list(
        repairable_items[
            first_wave_limit:
        ]
    )

    improving_items: list[
        CandidateProductionItem
    ] = []

    attempted_items: list[
        CandidateProductionItem
    ] = []

    for item in first_wave:
        candidate_id = (
            item.candidate.candidate_id
        )

        outcome = _repair_one_item(
            item=item,
            task_dir=task_dir,
            repair_workspace_base_dir=(
                repair_workspace_base_dir
            ),
            run_context=run_context,
            quality=quality,
            specification=specification,
            compiled_specification=(
                compiled_specification
            ),
            repairer=repairer,
            repair_budget=repair_budget,
            execution_timeout_seconds=(
                execution_timeout_seconds
            ),
            data_inspections=(
                data_inspections
            ),
        )

        if outcome.attempted:
            attempted.append(
                candidate_id
            )
            attempted_items.append(
                item
            )
        else:
            skipped.append(
                candidate_id
            )

        if outcome.failure:
            failures[
                candidate_id
            ] = outcome.failure

        if outcome.validated:
            if candidate_id not in repaired:
                repaired.append(
                    candidate_id
                )

        elif outcome.improved:
            improving_items.append(
                item
            )

    has_publishable_candidate = any(
        _attempt_is_publishable(
            item.candidate.latest_attempt()
        )
        for item in all_items
    )

    rescue_item = None

    if (
        not has_publishable_candidate
        and repair_budget.can_reserve()
        and (
            run_context
            .budget
            .remaining_candidate_attempts
            > 0
        )
        and (
            not repairer.uses_model_budget
            or (
                run_context
                .budget
                .remaining_model_calls
                > 0
            )
        )
    ):
        if improving_items:
            rescue_item = min(
                improving_items,
                key=_repair_priority_key,
            )

        elif deferred:
            # If no first repair improved, do not waste the reserved
            # slot: give the next-closest untouched candidate its
            # first repair.
            rescue_item = deferred.pop(0)

        elif attempted_items:
            # One final bounded second shot. The repair prompt sees
            # the incremented repair-attempt number and is explicitly
            # told to re-derive rather than repeat the same patch.
            rescue_item = min(
                attempted_items,
                key=_repair_priority_key,
            )

    if rescue_item is not None:
        candidate_id = (
            rescue_item
            .candidate
            .candidate_id
        )

        outcome = _repair_one_item(
            item=rescue_item,
            task_dir=task_dir,
            repair_workspace_base_dir=(
                repair_workspace_base_dir
            ),
            run_context=run_context,
            quality=quality,
            specification=specification,
            compiled_specification=(
                compiled_specification
            ),
            repairer=repairer,
            repair_budget=repair_budget,
            execution_timeout_seconds=(
                execution_timeout_seconds
            ),
            data_inspections=(
                data_inspections
            ),
        )

        if outcome.attempted:
            attempted.append(
                candidate_id
            )
        else:
            if candidate_id not in skipped:
                skipped.append(
                    candidate_id
                )

        if outcome.failure:
            failures[
                candidate_id
            ] = outcome.failure

        if outcome.validated:
            if candidate_id not in repaired:
                repaired.append(
                    candidate_id
                )

    for item in deferred:
        candidate_id = (
            item.candidate.candidate_id
        )

        if candidate_id not in skipped:
            skipped.append(
                candidate_id
            )

        if candidate_id not in failures:
            if not repair_budget.can_reserve():
                failures[candidate_id] = (
                    "Targeted repair budget is exhausted."
                )

            elif (
                repairer.uses_model_budget
                and (
                    run_context
                    .budget
                    .remaining_model_calls
                    <= 0
                )
            ):
                failures[candidate_id] = (
                    "Global model-call budget is exhausted."
                )

            elif (
                run_context
                .budget
                .remaining_candidate_attempts
                <= 0
            ):
                failures[candidate_id] = (
                    "Global candidate-attempt budget is exhausted."
                )

            elif has_publishable_candidate:
                failures[candidate_id] = (
                    "Skipped because a publishable candidate "
                    "already exists."
                )

            else:
                failures[candidate_id] = (
                    "Skipped after publication-rescue prioritization."
                )

    skipped[:] = sorted(
        set(skipped)
    )

    state.complete_stage(
        stage,
        detail=(
            f"{len(attempted)} repair attempt(s); "
            f"{len(repaired)} candidate(s) validated after repair"
        ),
    )

    return TargetedRepairResult(
        production=quality.production,
        repair_budget=repair_budget,
        attempted_candidate_ids=tuple(attempted),
        repaired_candidate_ids=tuple(repaired),
        skipped_candidate_ids=tuple(skipped),
        failure_messages=failures,
    )
