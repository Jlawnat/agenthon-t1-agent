from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
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


@dataclass(frozen=True)
class RepairedCandidate:
    code: str
    tokens_used: int = 0

    def __post_init__(self) -> None:
        if self.tokens_used < 0:
            raise ValueError("tokens_used must be non-negative")


RepairFunction = Callable[[RepairRequest], RepairedCandidate]


@dataclass(frozen=True)
class RepairAdapter:
    repair: RepairFunction
    uses_model_budget: bool = True
    name: str = "repair-generator"


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

    items = sorted(
        quality.production.items,
        key=lambda item: item.candidate.candidate_id,
    )

    for item in items:
        candidate = item.candidate
        candidate_id = candidate.candidate_id

        _ensure_initial_validation_attempt(item)

        source_attempt = candidate.latest_attempt()
        if source_attempt is None:
            skipped.append(candidate_id)
            continue

        validation_error = None
        if candidate.current_status == "code_invalid":
            validation_error = (
                candidate.hard_failures[0]
                if candidate.hard_failures
                else source_attempt.stderr
            )

        failure_label = classify_failure(
            source_attempt,
            validation_error=validation_error,
        )

        if failure_label is None:
            skipped.append(candidate_id)
            continue

        source_code = item.validated_code or item.raw_code
        if not source_code:
            skipped.append(candidate_id)
            failures[candidate_id] = "No candidate source code was available for repair."
            continue

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

        if not repair_budget.can_reserve():
            skipped.append(candidate_id)
            failures[candidate_id] = "Targeted repair budget is exhausted."
            continue

        if (
            repairer.uses_model_budget
            and run_context.budget.remaining_model_calls <= 0
        ):
            skipped.append(candidate_id)
            failures[candidate_id] = "Global model-call budget is exhausted."
            continue

        try:
            brief = build_repair_brief(
                attempt=source_attempt,
                budget=repair_budget,
                validation_error=validation_error,
            )
        except RepairBudgetExceeded as exc:
            skipped.append(candidate_id)
            failures[candidate_id] = _bounded_error(exc)
            continue
        except Exception as exc:
            skipped.append(candidate_id)
            failures[candidate_id] = (
                "Repair brief construction failed: "
                f"{_bounded_error(exc)}"
            )
            continue

        if brief is None:
            skipped.append(candidate_id)
            continue

        if repairer.uses_model_budget:
            try:
                run_context.budget.reserve_model_call()
            except RunBudgetExceeded as exc:
                skipped.append(candidate_id)
                failures[candidate_id] = _bounded_error(exc)
                continue

        attempted.append(candidate_id)

        request = RepairRequest(
            candidate_id=candidate_id,
            candidate_seed=item.candidate_seed,
            source_code=source_code,
            brief=brief,
            specification=specification,
            compiled_specification=compiled_specification,
        )

        repair_start = time.perf_counter()

        try:
            repaired_candidate = repairer.repair(request)
        except Exception as exc:
            elapsed = time.perf_counter() - repair_start
            repair_budget.record_usage(tokens=0, wall_seconds=elapsed)
            candidate.current_status = "repair_generation_failed"
            failures[candidate_id] = _bounded_error(exc)
            continue

        repair_elapsed = time.perf_counter() - repair_start

        repair_budget.record_usage(
            tokens=repaired_candidate.tokens_used,
            wall_seconds=repair_elapsed,
        )

        if repaired_candidate.tokens_used:
            run_context.budget.record_tokens(repaired_candidate.tokens_used)

        _record_repair_strategy(
            item=item,
            brief=brief,
            repairer=repairer,
            tokens_used=repaired_candidate.tokens_used,
        )

        try:
            validated_code = validate_python_code(repaired_candidate.code)
        except Exception as exc:
            error = f"{type(exc).__name__}: {_bounded_error(exc)}"
            _code_validation_attempt(
                item=item,
                error=error,
                repair_reason=brief.strategy,
                repaired=True,
            )
            failures[candidate_id] = error
            continue

        try:
            run_context.budget.reserve_candidate_attempt()
        except RunBudgetExceeded as exc:
            candidate.current_status = "repair_execution_budget_exhausted"
            failures[candidate_id] = _bounded_error(exc)
            continue

        next_attempt_number = len(candidate.attempts) + 1

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

            solver_path = workspace.source_dir / "solver.py"
            solver_path.write_text(
                validated_code,
                encoding="utf-8",
            )

            timeout = run_context.bounded_timeout(
                execution_timeout_seconds
            )

            execution = run_candidate(
                workspace,
                solver_path,
                timeout_seconds=timeout,
                env_overrides=_candidate_environment(
                    item.candidate_seed
                ),
            )
        except Exception as exc:
            candidate.current_status = "repair_execution_infrastructure_failed"
            failures[candidate_id] = _bounded_error(exc)
            continue

        item.validated_code = validated_code
        item.workspace = workspace
        item.solver_path = solver_path
        item.execution = execution

        attempt = collect_execution_evidence(
            execution=execution,
            output_dir=workspace.output_dir,
            required_output_paths=specification.required_output_paths,
            attempt_number=next_attempt_number,
            solver_path=solver_path,
        )

        attempt.repair_reason = brief.strategy

        try:
            evaluate_collected_attempt(
                candidate=candidate,
                attempt=attempt,
                output_dir=workspace.output_dir,
                required_output_paths=specification.required_output_paths,
                required_columns=compiled_specification.required_columns,
                schema_expectations=quality.schema_expectations,
                compiled_specification=compiled_specification,
            )
        except Exception as exc:
            attempt.structural_evidence.append(
                _repair_evaluation_error(exc)
            )

            if candidate.latest_attempt() is not attempt:
                candidate.add_attempt(attempt)

            candidate.current_status = "repair_evaluation_failed"
            candidate.hard_failures = [
                "Repaired candidate evaluation failed."
            ]
            failures[candidate_id] = _bounded_error(exc)
            continue

        repaired_attempt = (
            candidate.latest_attempt()
        )

        if (
            repaired_attempt is not None
            and _attempt_quality_key(
                repaired_attempt
            )
            >= _attempt_quality_key(
                source_attempt
            )
        ):
            # A repair is advisory. If it is not strictly better,
            # restore the previous candidate and its execution artifacts.
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

            failures[candidate_id] = (
                "Repair was not a strict improvement; "
                "restored the previous candidate attempt."
            )
            continue

        if candidate.current_status == "validated":
            repaired.append(candidate_id)

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
