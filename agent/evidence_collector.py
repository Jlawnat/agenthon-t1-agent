from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.candidate_contract import (
    CandidateAttempt,
    VerificationEvidence,
)
from agent.executor import ExecutionResult


def _relative_files(root: Path) -> list[str]:
    files: list[str] = []

    if not root.exists():
        return files

    for path in root.rglob("*"):
        if path.is_file():
            files.append(
                str(path.relative_to(root))
            )

    return sorted(files)


def _file_metadata(
    root: Path,
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}

    if not root.exists():
        return metadata

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        rel = str(path.relative_to(root))

        metadata[rel] = {
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix.lower(),
        }

    return metadata


def collect_execution_evidence(
    *,
    execution: ExecutionResult,
    output_dir: Path,
    required_output_paths: list[str],
    attempt_number: int,
    solver_path: Path,
) -> CandidateAttempt:

    attempt = CandidateAttempt(
        attempt_number=attempt_number,
        solver_path=str(solver_path),
        return_code=execution.return_code,
        timed_out=execution.timed_out,
        runtime_seconds=execution.runtime_seconds,
        stdout=execution.stdout,
        stderr=execution.stderr,
    )

    produced_files = _relative_files(
        output_dir
    )

    attempt.output_files = produced_files

    # ---------------------------------------------------------
    # Execution status
    # ---------------------------------------------------------
    if execution.timed_out:
        attempt.structural_evidence.append(
            VerificationEvidence(
                name="execution_timeout",
                status="fail",
                severity="hard_fail",
                message="Candidate execution timed out.",
            )
        )

    elif execution.return_code != 0:
        attempt.structural_evidence.append(
            VerificationEvidence(
                name="execution_return_code",
                status="fail",
                severity="hard_fail",
                message=(
                    "Candidate exited with "
                    f"return code "
                    f"{execution.return_code}."
                ),
                details={
                    "stderr": execution.stderr,
                },
            )
        )

    else:
        attempt.structural_evidence.append(
            VerificationEvidence(
                name="execution",
                status="pass",
                severity="info",
                message=(
                    "Candidate executed successfully."
                ),
            )
        )

    # ---------------------------------------------------------
    # Required output files
    # ---------------------------------------------------------
    required_names: list[str] = []

    for path in required_output_paths:
        required_names.append(
            Path(path).name
        )

    produced_names = {
        Path(path).name
        for path in produced_files
    }

    missing = [
        name
        for name in required_names
        if name not in produced_names
    ]

    if missing:
        attempt.structural_evidence.append(
            VerificationEvidence(
                name="required_outputs",
                status="fail",
                severity="hard_fail",
                message=(
                    "Required output files are missing."
                ),
                details={
                    "missing": missing,
                },
            )
        )

    else:
        attempt.structural_evidence.append(
            VerificationEvidence(
                name="required_outputs",
                status="pass",
                severity="info",
                message=(
                    "All required output files exist."
                ),
            )
        )

    # ---------------------------------------------------------
    # Unexpected output files
    # ---------------------------------------------------------
    unexpected = [
        path
        for path in produced_files
        if Path(path).name
        not in set(required_names)
    ]

    if unexpected:
        attempt.structural_evidence.append(
            VerificationEvidence(
                name="unexpected_outputs",
                status="warning",
                severity="warning",
                message=(
                    "Candidate produced unexpected "
                    "output files."
                ),
                details={
                    "unexpected": unexpected,
                },
            )
        )

    # ---------------------------------------------------------
    # Output file metadata
    # ---------------------------------------------------------
    metadata = _file_metadata(
        output_dir
    )

    attempt.structural_evidence.append(
        VerificationEvidence(
            name="output_inventory",
            status="pass",
            severity="info",
            message="Output inventory collected.",
            details={
                "files": metadata,
            },
        )
    )

    return attempt