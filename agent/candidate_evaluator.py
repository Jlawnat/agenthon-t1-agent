from __future__ import annotations

from pathlib import Path

from agent.candidate_contract import (
    CandidateRecord,
    CandidateAttempt,
)
from agent.compiled_specification import (
    CompiledSpecification,
)
from agent.evidence_collector import (
    collect_execution_evidence,
)
from agent.executor import ExecutionResult
from agent.quant_invariants import (
    QuantInvariantConfig,
    evaluate_quant_invariants,
)
from agent.schema_expectations import (
    SchemaExpectations,
)
from agent.structural_verifier import (
    verify_structural_output,
)


def _has_hard_failure(
    attempt: CandidateAttempt,
) -> bool:
    all_evidence = (
        attempt.structural_evidence
        + attempt.quant_evidence
    )

    return any(
        item.severity == "hard_fail"
        and item.status == "fail"
        for item in all_evidence
    )


def _collect_messages(
    attempt: CandidateAttempt,
    *,
    severity: str,
) -> list[str]:
    messages: list[str] = []

    for item in (
        attempt.structural_evidence
        + attempt.quant_evidence
    ):
        if (
            item.severity == severity
            and item.status
            in {"fail", "warning"}
        ):
            messages.append(
                item.message
            )

    return messages


def evaluate_collected_attempt(
    *,
    candidate: CandidateRecord,
    attempt: CandidateAttempt,
    output_dir: Path,
    required_output_paths: list[str],
    required_columns: list[str],
    schema_expectations: SchemaExpectations,
    compiled_specification: (
        CompiledSpecification | None
    ) = None,
) -> CandidateAttempt:
    """
    Evaluate an already-collected execution attempt.

    This exists so Phase 4 can represent:

        COLLECT_EVIDENCE
            ->
        EVALUATE

    as two explicit state-machine stages.

    The older evaluate_candidate_attempt() API remains
    available and delegates here after collecting
    execution evidence.
    """

    quant_config = None

    if compiled_specification is not None:
        quant_config = (
            QuantInvariantConfig
            .from_compiled_specification(
                compiled_specification
            )
        )

    for required_path in required_output_paths:
        output_name = Path(
            required_path
        ).name

        actual_output = (
            output_dir
            / output_name
        )

        structural_evidence = (
            verify_structural_output(
                output_path=actual_output,
                required_columns=required_columns,
                expected_rows=(
                    schema_expectations
                    .expected_rows
                ),
                id_column=(
                    schema_expectations
                    .id_column
                ),
                expected_ids=(
                    schema_expectations
                    .expected_ids
                ),
                require_id_order=(
                    schema_expectations
                    .require_id_order
                ),
                expected_dtypes=(
                    schema_expectations
                    .expected_dtypes
                ),
            )
        )

        attempt.structural_evidence.extend(
            structural_evidence
        )

        if quant_config is not None:
            quant_evidence = (
                evaluate_quant_invariants(
                    output_path=actual_output,
                    config=quant_config,
                )
            )

            attempt.quant_evidence.extend(
                quant_evidence
            )

    candidate.add_attempt(
        attempt
    )

    if _has_hard_failure(
        attempt
    ):
        candidate.current_status = (
            "failed"
        )

        candidate.hard_failures = (
            _collect_messages(
                attempt,
                severity="hard_fail",
            )
        )

    else:
        if quant_config is None:
            candidate.current_status = (
                "structurally_valid"
            )

        else:
            candidate.current_status = (
                "validated"
            )

        candidate.hard_failures = []

    candidate.warnings = (
        _collect_messages(
            attempt,
            severity="warning",
        )
    )

    return attempt


def evaluate_candidate_attempt(
    *,
    candidate: CandidateRecord,
    execution: ExecutionResult,
    output_dir: Path,
    solver_path: Path,
    required_output_paths: list[str],
    required_columns: list[str],
    schema_expectations: SchemaExpectations,
    attempt_number: int,
    compiled_specification: (
        CompiledSpecification | None
    ) = None,
) -> CandidateAttempt:
    """
    Backward-compatible Phase 3 API.

    Collect execution evidence, then delegate to
    evaluate_collected_attempt().
    """

    attempt = (
        collect_execution_evidence(
            execution=execution,
            output_dir=output_dir,
            required_output_paths=(
                required_output_paths
            ),
            attempt_number=(
                attempt_number
            ),
            solver_path=solver_path,
        )
    )

    return evaluate_collected_attempt(
        candidate=candidate,
        attempt=attempt,
        output_dir=output_dir,
        required_output_paths=(
            required_output_paths
        ),
        required_columns=(
            required_columns
        ),
        schema_expectations=(
            schema_expectations
        ),
        compiled_specification=(
            compiled_specification
        ),
    )