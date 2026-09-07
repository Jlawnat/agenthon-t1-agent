from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
from typing import Any

from agent.candidate_workspace import (
    CandidateWorkspace,
)
from agent.compiled_specification import (
    CompiledSpecification,
)
from agent.executor import (
    ExecutionResult,
    run_python_candidate,
)
from agent.final_audit import (
    FinalAuditResult,
    audit_final_output,
)
from agent.schema_expectations import (
    SchemaExpectations,
)


@dataclass(frozen=True)
class CleanRoomResult:
    passed: bool
    reason: str

    workspace_root: str | None
    clean_solver_path: str | None

    execution: ExecutionResult | None
    audit: FinalAuditResult | None

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(self)


def _first_audit_failure(
    audit: FinalAuditResult,
) -> str | None:
    for finding in audit.findings:
        if (
            finding.status == "fail"
            and finding.severity
            == "hard_fail"
        ):
            return finding.message

    return None


def clean_room_reexecute(
    *,
    selected_solver_path: Path,
    task_dir: Path,
    base_dir: Path,
    compiled_specification:
        CompiledSpecification,
    schema_expectations:
        SchemaExpectations,
    timeout_seconds: int = 120,
    workspace_id: int = 999_999,
) -> CleanRoomResult:
    selected_solver_path = (
        selected_solver_path.resolve()
    )

    task_dir = task_dir.resolve()
    base_dir = base_dir.resolve()

    if (
        not selected_solver_path.exists()
        or not selected_solver_path.is_file()
    ):
        return CleanRoomResult(
            passed=False,
            reason=(
                "Selected solver does "
                "not exist."
            ),
            workspace_root=None,
            clean_solver_path=None,
            execution=None,
            audit=None,
        )

    if (
        not task_dir.exists()
        or not task_dir.is_dir()
    ):
        return CleanRoomResult(
            passed=False,
            reason=(
                "Task directory does "
                "not exist."
            ),
            workspace_root=None,
            clean_solver_path=None,
            execution=None,
            audit=None,
        )

    if timeout_seconds <= 0:
        return CleanRoomResult(
            passed=False,
            reason=(
                "Clean-room timeout must "
                "be greater than zero."
            ),
            workspace_root=None,
            clean_solver_path=None,
            execution=None,
            audit=None,
        )

    try:
        workspace = (
            CandidateWorkspace.create(
                base_dir=base_dir,
                candidate_id=workspace_id,
                task_dir=task_dir,
            )
        )

        clean_solver_path = (
            workspace.source_dir
            / "solver.py"
        )

        shutil.copy2(
            selected_solver_path,
            clean_solver_path,
        )

    except Exception as exc:
        return CleanRoomResult(
            passed=False,
            reason=(
                "Clean-room workspace setup "
                "failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            workspace_root=None,
            clean_solver_path=None,
            execution=None,
            audit=None,
        )

    try:
        execution = run_python_candidate(
            script_path=(
                clean_solver_path
            ),
            cwd=workspace.root_dir,
            timeout_seconds=(
                timeout_seconds
            ),
        )

    except Exception as exc:
        return CleanRoomResult(
            passed=False,
            reason=(
                "Clean-room execution "
                "could not start: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            workspace_root=str(
                workspace.root_dir
            ),
            clean_solver_path=str(
                clean_solver_path
            ),
            execution=None,
            audit=None,
        )

    try:
        audit = audit_final_output(
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

    except Exception as exc:
        return CleanRoomResult(
            passed=False,
            reason=(
                "Final audit raised an "
                "unexpected error: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            workspace_root=str(
                workspace.root_dir
            ),
            clean_solver_path=str(
                clean_solver_path
            ),
            execution=execution,
            audit=None,
        )

    if execution.timed_out:
        return CleanRoomResult(
            passed=False,
            reason=(
                "Clean-room execution "
                "timed out."
            ),
            workspace_root=str(
                workspace.root_dir
            ),
            clean_solver_path=str(
                clean_solver_path
            ),
            execution=execution,
            audit=audit,
        )

    if execution.return_code != 0:
        return CleanRoomResult(
            passed=False,
            reason=(
                "Clean-room execution "
                "failed with return code "
                f"{execution.return_code}."
            ),
            workspace_root=str(
                workspace.root_dir
            ),
            clean_solver_path=str(
                clean_solver_path
            ),
            execution=execution,
            audit=audit,
        )

    if not audit.passed:
        first_failure = (
            _first_audit_failure(
                audit
            )
        )

        return CleanRoomResult(
            passed=False,
            reason=(
                first_failure
                or (
                    "Final admissibility "
                    "audit failed."
                )
            ),
            workspace_root=str(
                workspace.root_dir
            ),
            clean_solver_path=str(
                clean_solver_path
            ),
            execution=execution,
            audit=audit,
        )

    return CleanRoomResult(
        passed=True,
        reason=(
            "Clean-room re-execution "
            "and final audit passed."
        ),
        workspace_root=str(
            workspace.root_dir
        ),
        clean_solver_path=str(
            clean_solver_path
        ),
        execution=execution,
        audit=audit,
    )