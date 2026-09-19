from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
from typing import Any, Mapping

from agent.audit_pipeline import (
    CleanRoomAuditResult,
    run_clean_room_audit_stage,
)
from agent.budget import (
    RepairBudget,
)
from agent.candidate_production import (
    CandidateProductionResult,
    GeneratorAdapter,
    run_candidate_production,
)
from agent.front_half import (
    FrontHalfDependencies,
    FrontHalfResult,
    run_deterministic_front_half,
)
from agent.orchestrator_state import (
    OrchestratorState,
)
from agent.publish_pipeline import (
    PublishResult,
    _run_token,
    run_publish_stage,
)
from agent.quality_pipeline import (
    InitialQualityResult,
    run_initial_quality_evaluation,
)
from agent.repair_pipeline import (
    RepairAdapter,
    TargetedRepairResult,
    run_targeted_repair_stage,
)
from agent.run_recovery import (
    PublishRecoveryResult,
    RunJournal,
    recover_publish_state,
)
from agent.selection_pipeline import (
    SelectionPipelineResult,
    run_selection_stage,
)


@dataclass(frozen=True)
class OrchestratorResult:
    state: OrchestratorState

    run_id: str
    run_dir: Path

    recovery: PublishRecoveryResult | None

    front_half: FrontHalfResult
    production: CandidateProductionResult
    quality: InitialQualityResult
    repair: TargetedRepairResult
    selection: SelectionPipelineResult
    audit: CleanRoomAuditResult
    publish: PublishResult


def _derive_run_id(
    *,
    task_dir: Path,
    environ: Mapping[str, str] | None,
) -> str:
    """
    Derive a stable run ID from task location + root seed.

    Re-running the same task with the same QFBENCH_SEED
    therefore reaches the same crash-recovery namespace.
    """

    effective_environment = (
        environ
        if environ is not None
        else os.environ
    )

    seed = str(
        effective_environment.get(
            "QFBENCH_SEED",
            "missing-seed",
        )
    )

    material = (
        f"{task_dir.resolve()}|{seed}"
    )

    digest = sha256(
        material.encode("utf-8")
    ).hexdigest()[:16]

    return (
        f"solve-{digest}"
    )


def _publish_artifact_paths(
    *,
    final_output_dir: Path,
    run_id: str,
) -> tuple[
    Path,
    Path,
]:
    """
    Return staging and backup paths used by Phase 4.9.
    """

    final_output_dir = (
        final_output_dir.resolve()
    )

    token = _run_token(
        run_id
    )

    parent = (
        final_output_dir.parent
    )

    staging = (
        parent
        / (
            f".{final_output_dir.name}"
            f".staging.{token}"
        )
    )

    backup = (
        parent
        / (
            f".{final_output_dir.name}"
            f".backup.{token}"
        )
    )

    return (
        staging,
        backup,
    )


def _checkpoint_expected_files(
    checkpoint: dict
    | None,
) -> tuple[str, ...]:
    """
    Recover the last known audited/published file inventory.

    Pre-publish checkpoints store it under extra so that
    recovery still knows the expected commit inventory if
    the process dies inside publication.
    """

    if not checkpoint:
        return ()

    publish = (
        checkpoint.get(
            "publish"
        )
    )

    if isinstance(
        publish,
        dict,
    ):
        files = (
            publish.get(
                "published_files"
            )
        )

        if isinstance(
            files,
            list,
        ):
            return tuple(
                sorted(
                    str(item)
                    for item in files
                )
            )

    extra = (
        checkpoint.get(
            "extra"
        )
    )

    if isinstance(
        extra,
        dict,
    ):
        files = (
            extra.get(
                "publish_expected_files"
            )
        )

        if isinstance(
            files,
            list,
        ):
            return tuple(
                sorted(
                    str(item)
                    for item in files
                )
            )

    return ()


def _resolve_candidate_count(
    *,
    requested: int | None,
    plan: Any,
    default: int = 3,
) -> int:
    """
    Resolve candidate generation count.

    An explicit caller override wins. Otherwise use
    the deterministic task-plan allocation when the
    runtime plan exposes it. Generic/test plans fall
    back to the safe historical default.
    """

    if requested is not None:
        if (
            isinstance(requested, bool)
            or not isinstance(requested, int)
            or requested <= 0
        ):
            raise ValueError(
                "candidate_count must be a "
                "positive integer"
            )

        return requested

    task_plan = getattr(
        plan,
        "task_plan",
        None,
    )

    planned = getattr(
        task_plan,
        "candidate_count",
        None,
    )

    if (
        isinstance(planned, int)
        and not isinstance(planned, bool)
        and planned > 0
    ):
        return planned

    return default


def _resolve_execution_timeout(
    *,
    requested_seconds: float,
    run_context: Any,
    candidate_count: int,
) -> float:
    # Expand the historical 120-second ceiling only when the task
    # card has enough wall-clock budget. The executor still calls
    # RunContext.bounded_timeout(), so the global deadline and safety
    # margin remain authoritative.
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count <= 0
    ):
        raise ValueError(
            "candidate_count must be a positive integer"
        )

    requested = float(
        requested_seconds
    )

    if requested <= 0:
        raise ValueError(
            "requested_seconds must be positive"
        )

    card_timeout = float(
        getattr(
            run_context,
            "card_timeout_seconds",
            0.0,
        )
    )

    if card_timeout <= 0:
        return requested

    adaptive_share = (
        0.20
        * card_timeout
        / candidate_count
    )

    adaptive = min(
        240.0,
        adaptive_share,
    )

    return max(
        requested,
        adaptive,
    )


def solve_task(
    *,
    task_dir: Path,
    final_output_dir: Path,
    work_root: Path,
    front_half_dependencies: FrontHalfDependencies,
    generator: GeneratorAdapter,
    repairer: RepairAdapter,
    candidate_count: int | None = None,
    execution_timeout_seconds: float = 120.0,
    audit_timeout_seconds: float = 120.0,
    repair_budget: RepairBudget | None = None,
    environ: Mapping[str, str] | None = None,
    run_id: str | None = None,
) -> OrchestratorResult:
    """
    Execute the complete Phase 4 state machine:

        recovery
        snapshot
        specification
        compile
        plan
        generate
        validate code
        isolated execute
        collect evidence
        evaluate
        targeted repair
        select
        clean-room audit
        atomic publish

    All model/provider behavior remains outside this
    function behind GeneratorAdapter and RepairAdapter.
    """

    task_dir = (
        task_dir.resolve()
    )

    final_output_dir = (
        final_output_dir.resolve()
    )

    work_root = (
        work_root.resolve()
    )

    work_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if run_id is None:
        run_id = _derive_run_id(
            task_dir=task_dir,
            environ=environ,
        )

    run_dir = (
        work_root
        / "runs"
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    journal = RunJournal(
        run_dir=run_dir
    )

    state = OrchestratorState(
        run_id=run_id
    )

    previous_checkpoint = (
        journal.load_checkpoint()
    )

    expected_recovery_files = (
        _checkpoint_expected_files(
            previous_checkpoint
        )
    )

    (
        staging_path,
        backup_path,
    ) = _publish_artifact_paths(
        final_output_dir=(
            final_output_dir
        ),
        run_id=run_id,
    )

    recovery: (
        PublishRecoveryResult
        | None
    ) = None

    if (
        staging_path.exists()
        or backup_path.exists()
    ):
        recovery = (
            recover_publish_state(
                final_output_dir=(
                    final_output_dir
                ),
                run_id=run_id,
                expected_files=(
                    expected_recovery_files
                ),
            )
        )

        journal.event(
            "publish_recovery",
            state=state,
            details={
                "action": (
                    recovery.action
                ),
                "restored_backup": (
                    recovery
                    .restored_backup
                ),
                "removed_staging": (
                    recovery
                    .removed_staging
                ),
                "removed_backup": (
                    recovery
                    .removed_backup
                ),
                "final_output_exists": (
                    recovery
                    .final_output_exists
                ),
            },
        )

    journal.event(
        "run_started",
        state=state,
        details={
            "candidate_count": (
                candidate_count
            ),
        },
    )

    front_half: FrontHalfResult | None = None

    try:

        front_half = (
            run_deterministic_front_half(
                task_dir=task_dir,
                state=state,
                dependencies=(
                    front_half_dependencies
                ),
                environ=environ,
            )
        )

        resolved_candidate_count = (
            _resolve_candidate_count(
                requested=candidate_count,
                plan=front_half.plan,
            )
        )

        resolved_execution_timeout = (
            _resolve_execution_timeout(
                requested_seconds=(
                    execution_timeout_seconds
                ),
                run_context=(
                    front_half
                    .run_context
                ),
                candidate_count=(
                    resolved_candidate_count
                ),
            )
        )

        resolved_audit_timeout = (
            _resolve_execution_timeout(
                requested_seconds=(
                    audit_timeout_seconds
                ),
                run_context=(
                    front_half
                    .run_context
                ),
                candidate_count=1,
            )
        )

        journal.event(
            "front_half_completed",
            state=state,
            details={
                "task_id": (
                    state.task_id
                ),
            },
        )

        journal.checkpoint(
            state=state,
            run_context=(
                front_half
                .run_context
            ),
            extra={
                "phase": (
                    "front_half"
                ),
            },
        )

        production = (
            run_candidate_production(
                task_dir=task_dir,
                workspace_base_dir=(
                    run_dir
                    / "candidates"
                ),
                state=state,
                run_context=(
                    front_half
                    .run_context
                ),
                plan=(
                    front_half.plan
                ),
                specification=(
                    front_half
                    .specification
                ),
                compiled_specification=(
                    front_half
                    .compiled_specification
                ),
                generator=generator,
                candidate_count=(
                    resolved_candidate_count
                ),
                execution_timeout_seconds=(
                    resolved_execution_timeout
                ),
            )
        )

        journal.event(
            "candidate_production_completed",
            state=state,
            details={
                "candidate_count": len(
                    production.items
                ),
                "executed_count": len(
                    production
                    .executed_items
                ),
            },
        )

        journal.checkpoint(
            state=state,
            run_context=(
                front_half
                .run_context
            ),
            extra={
                "phase": (
                    "candidate_production"
                ),
            },
        )

        quality = (
            run_initial_quality_evaluation(
                task_dir=task_dir,
                state=state,
                production=production,
                specification=(
                    front_half
                    .specification
                ),
                compiled_specification=(
                    front_half
                    .compiled_specification
                ),
            )
        )

        journal.event(
            "initial_quality_completed",
            state=state,
            details={
                "evaluated_candidate_ids": list(
                    quality
                    .evaluated_candidate_ids
                ),
            },
        )

        journal.checkpoint(
            state=state,
            run_context=(
                front_half
                .run_context
            ),
            extra={
                "phase": (
                    "initial_quality"
                ),
            },
        )

        repair = (
            run_targeted_repair_stage(
                task_dir=task_dir,
                repair_workspace_base_dir=(
                    run_dir
                    / "repairs"
                ),
                state=state,
                run_context=(
                    front_half
                    .run_context
                ),
                quality=quality,
                specification=(
                    front_half
                    .specification
                ),
                compiled_specification=(
                    front_half
                    .compiled_specification
                ),
                repairer=repairer,
                repair_budget=(
                    repair_budget
                ),
                execution_timeout_seconds=(
                    resolved_execution_timeout
                ),
                data_inspections=(
                    getattr(
                        front_half.plan,
                        "data_inspections",
                        None,
                    )
                ),
            )
        )

        journal.event(
            "targeted_repair_completed",
            state=state,
            details={
                "attempted_candidate_ids": list(
                    repair
                    .attempted_candidate_ids
                ),
                "repaired_candidate_ids": list(
                    repair
                    .repaired_candidate_ids
                ),
                "skipped_candidate_ids": list(
                    repair
                    .skipped_candidate_ids
                ),
            },
        )

        journal.checkpoint(
            state=state,
            run_context=(
                front_half
                .run_context
            ),
            extra={
                "phase": (
                    "targeted_repair"
                ),
            },
        )

        selection = (
            run_selection_stage(
                state=state,
                production=(
                    repair.production
                ),
                allow_best_effort=True,
                semantic_compare=(
                    repairer.semantic_compare
                ),
                semantic_compare_uses_model_budget=(
                    repairer.semantic_compare_uses_model_budget
                ),
                run_context=(
                    front_half.run_context
                ),
                specification=(
                    front_half.specification
                ),
                compiled_specification=(
                    front_half.compiled_specification
                ),
            )
        )

        journal.event(
            "selection_completed",
            state=state,
            details={
                "selected_candidate_id": (
                    selection
                    .selected_candidate_id
                ),
                "first_candidate_valid": (
                    selection
                    .selection
                    .first_candidate_valid
                ),
                "any_of_three_valid": (
                    selection
                    .selection
                    .any_of_three_valid
                ),
                "ranked_candidate_ids": list(
                    selection
                    .selection
                    .ranked_candidate_ids
                ),
            },
        )

        journal.checkpoint(
            state=state,
            run_context=(
                front_half
                .run_context
            ),
            selection=selection,
            extra={
                "phase": (
                    "selection"
                ),
            },
        )

        audit = (
            run_clean_room_audit_stage(
                task_dir=task_dir,
                audit_workspace_base_dir=(
                    run_dir
                    / "clean_room_audit"
                ),
                state=state,
                run_context=(
                    front_half
                    .run_context
                ),
                selection=selection,
                compiled_specification=(
                    front_half
                    .compiled_specification
                ),
                schema_expectations=(
                    quality
                    .schema_expectations
                ),
                execution_timeout_seconds=(
                    resolved_audit_timeout
                ),
                allow_quant_only_failures=True,
            )
        )

        expected_publish_files = (
            tuple(
                sorted(
                    audit
                    .audit
                    .required_files
                )
            )
        )

        journal.event(
            "clean_room_audit_completed",
            state=state,
            details={
                "selected_candidate_id": (
                    audit
                    .selected_candidate_id
                ),
                "required_files": list(
                    expected_publish_files
                ),
            },
        )

        journal.checkpoint(
            state=state,
            run_context=(
                front_half
                .run_context
            ),
            selection=selection,
            extra={
                "phase": (
                    "pre_publish"
                ),
                "publish_expected_files": list(
                    expected_publish_files
                ),
            },
        )

        publish = (
            run_publish_stage(
                state=state,
                audit_result=audit,
                final_output_dir=(
                    final_output_dir
                ),
                allow_quant_only_failures=True,
            )
        )

        journal.event(
            "publish_completed",
            state=state,
            details={
                "published_files": list(
                    publish
                    .published_files
                ),
                "replaced_existing_output": (
                    publish
                    .replaced_existing_output
                ),
            },
        )

        journal.checkpoint(
            state=state,
            run_context=(
                front_half
                .run_context
            ),
            selection=selection,
            publish=publish,
            extra={
                "phase": (
                    "completed"
                ),
            },
        )

        journal.event(
            "run_completed",
            state=state,
            details={
                "selected_candidate_id": (
                    selection
                    .selected_candidate_id
                ),
                "pass_at_1": (
                    selection
                    .selection
                    .first_candidate_valid
                ),
                "any_of_three": (
                    selection
                    .selection
                    .any_of_three_valid
                ),
            },
        )

        return OrchestratorResult(
            state=state,
            run_id=run_id,
            run_dir=run_dir,
            recovery=recovery,
            front_half=front_half,
            production=production,
            quality=quality,
            repair=repair,
            selection=selection,
            audit=audit,
            publish=publish,
        )

    except Exception as exc:

        try:
            journal.event(
                "run_failed",
                state=state,
                details={
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error": (
                        str(exc)[:1000]
                    ),
                },
            )

        except Exception:
            pass

        try:
            journal.checkpoint(
                state=state,
                run_context=(
                    front_half.run_context
                    if front_half is not None
                    else None
                ),
                extra={
                    "phase": (
                        "failed"
                    ),
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error": (
                        str(exc)[:1000]
                    ),
                },
            )

        except Exception:
            pass

        raise
