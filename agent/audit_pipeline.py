from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.candidate_runner import (
    run_candidate,
)
from agent.candidate_workspace import (
    CandidateWorkspace,
)
from agent.compiled_specification import (
    CompiledSpecification,
)
from agent.executor import (
    ExecutionResult,
)
from agent.final_audit import (
    FinalAuditResult,
    audit_final_output,
)
from agent.orchestrator_state import (
    OrchestratorState,
    PipelineStage,
)
from agent.run_budget import (
    RunBudgetExceeded,
)
from agent.run_context import (
    RunContext,
)
from agent.schema_expectations import (
    SchemaExpectations,
)
from agent.selection_pipeline import (
    SelectionPipelineResult,
)


class CleanRoomAuditError(RuntimeError):
    """Raised when final clean-room audit fails closed."""


@dataclass(frozen=True)
class CleanRoomAuditResult:
    selected_candidate_id: int

    workspace: CandidateWorkspace
    solver_path: Path

    execution: ExecutionResult
    audit: FinalAuditResult

    @property
    def audited_output_dir(
        self,
    ) -> Path:
        return self.workspace.output_dir


def _candidate_environment(
    candidate_seed: int,
) -> dict[str, str]:
    seed = str(
        candidate_seed
    )

    return {
        "QFBENCH_SEED": seed,
        "PYTHONHASHSEED": seed,
    }


def _audit_failure_summary(
    audit: FinalAuditResult,
) -> str:
    hard_failures = [
        finding.message
        for finding in audit.findings
        if (
            finding.status == "fail"
            and finding.severity
            == "hard_fail"
        )
    ]

    if not hard_failures:
        return (
            "Final clean-room audit failed."
        )

    joined = "; ".join(
        hard_failures[:5]
    )

    if len(hard_failures) > 5:
        joined += (
            f"; +{len(hard_failures) - 5} "
            "additional hard failure(s)"
        )

    return joined


def run_clean_room_audit_stage(
    *,
    task_dir: Path,
    audit_workspace_base_dir: Path,
    state: OrchestratorState,
    run_context: RunContext,
    selection: SelectionPipelineResult,
    compiled_specification: CompiledSpecification,
    schema_expectations: SchemaExpectations,
    execution_timeout_seconds: float = 120.0,
    allow_quant_only_failures: bool = False,
) -> CleanRoomAuditResult:
    """
    Re-execute the selected solver in a completely fresh
    candidate workspace, then run the existing Phase 3.13
    final-output audit against only those fresh outputs.

    The selected candidate's existing workspace is never
    modified.
    """

    stage = (
        PipelineStage
        .CLEAN_ROOM_AUDIT
    )

    state.start_stage(
        stage
    )

    task_dir = (
        task_dir.resolve()
    )

    audit_workspace_base_dir = (
        audit_workspace_base_dir
        .resolve()
    )

    audit_workspace_base_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_item = (
        selection.selected_item
    )

    candidate_id = (
        selected_item
        .candidate
        .candidate_id
    )

    candidate_seed = (
        selected_item
        .candidate_seed
    )

    source_solver = (
        selected_item
        .solver_path
    )

    if source_solver is None:
        reason = (
            "Selected candidate has no "
            "solver path for clean-room "
            "re-execution."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise CleanRoomAuditError(
            reason
        )

    source_solver = (
        source_solver.resolve()
    )

    if (
        not source_solver.exists()
        or not source_solver.is_file()
    ):
        reason = (
            "Selected candidate solver "
            "does not exist."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise CleanRoomAuditError(
            reason
        )

    try:
        solver_code = (
            source_solver.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:
        reason = (
            "Selected candidate solver "
            "could not be read: "
            f"{type(exc).__name__}: {exc}"
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise CleanRoomAuditError(
            reason
        ) from exc


    try:
        workspace = (
            CandidateWorkspace.create(
                base_dir=(
                    audit_workspace_base_dir
                ),
                candidate_id=(
                    candidate_id
                ),
                task_dir=task_dir,
            )
        )

        clean_solver = (
            workspace.source_dir
            / "solver.py"
        )

        clean_solver.write_text(
            solver_code,
            encoding="utf-8",
        )

    except Exception as exc:
        reason = (
            "Clean-room workspace "
            "creation failed: "
            f"{type(exc).__name__}: {exc}"
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise CleanRoomAuditError(
            reason
        ) from exc


    try:
        timeout = (
            run_context
            .bounded_timeout(
                execution_timeout_seconds
            )
        )

        execution = (
            run_candidate(
                workspace,
                clean_solver,
                timeout_seconds=(
                    timeout
                ),
                env_overrides=(
                    _candidate_environment(
                        candidate_seed
                    )
                ),
            )
        )

    except RunBudgetExceeded as exc:
        reason = (
            "Insufficient remaining "
            "wall-clock budget for "
            "clean-room audit."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise CleanRoomAuditError(
            reason
        ) from exc

    except Exception as exc:
        reason = (
            "Clean-room execution "
            "infrastructure failed: "
            f"{type(exc).__name__}: {exc}"
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise CleanRoomAuditError(
            reason
        ) from exc

    if execution.timed_out:
        reason = (
            "Selected candidate timed out "
            "during clean-room "
            "re-execution."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise CleanRoomAuditError(
            reason
        )

    if execution.return_code != 0:
        reason = (
            "Selected candidate failed "
            "during clean-room "
            "re-execution with return "
            f"code {execution.return_code}."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise CleanRoomAuditError(
            reason
        )


    try:
        audit = (
            audit_final_output(
                output_dir=(
                    workspace.output_dir
                ),
                compiled_specification=(
                    compiled_specification
                ),
                schema_expectations=(
                    schema_expectations
                ),
            )
        )

    except Exception as exc:
        reason = (
            "Final output audit "
            "infrastructure failed: "
            f"{type(exc).__name__}: {exc}"
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise CleanRoomAuditError(
            reason
        ) from exc

    if (
        not audit.passed
        and not (
            allow_quant_only_failures
            and audit.publishable
        )
    ):
        reason = (
            _audit_failure_summary(
                audit
            )
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise CleanRoomAuditError(
            reason
        )

    state.complete_stage(
        stage,
        detail=(
            "candidate "
            f"{candidate_id} passed "
            "clean-room re-execution "
            "and final admissibility audit"
        ),
    )

    return CleanRoomAuditResult(
        selected_candidate_id=(
            candidate_id
        ),
        workspace=workspace,
        solver_path=(
            clean_solver
        ),
        execution=execution,
        audit=audit,
    )