from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.candidate_contract import (
    CandidateAttempt,
    VerificationEvidence,
)
from agent.candidate_evaluator import (
    evaluate_collected_attempt,
)
from agent.candidate_production import (
    CandidateProductionResult,
)
from agent.compiled_specification import (
    CompiledSpecification,
)
from agent.evidence_collector import (
    collect_execution_evidence,
)
from agent.orchestrator_state import (
    OrchestratorState,
    PipelineStage,
)
from agent.schema_expectations import (
    SchemaExpectations,
    infer_schema_expectations,
)
from agent.specification import (
    TaskSpecification,
)


class QualityPipelineError(RuntimeError):
    """Raised when the quality pipeline cannot continue."""


@dataclass
class InitialQualityResult:
    production: CandidateProductionResult
    schema_expectations: SchemaExpectations
    attempts_by_candidate_id: dict[
        int,
        CandidateAttempt,
    ]

    @property
    def evaluated_candidate_ids(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                self.attempts_by_candidate_id
            )
        )


def _evaluation_failure_evidence(
    exc: Exception,
) -> VerificationEvidence:
    message = str(exc)

    if len(message) > 1000:
        message = (
            message[:985]
            + "...[truncated]"
        )

    return VerificationEvidence(
        name="evaluation_infrastructure_error",
        status="fail",
        severity="hard_fail",
        message=(
            "Candidate evaluation failed "
            "unexpectedly."
        ),
        details={
            "error_type": (
                type(exc).__name__
            ),
            "error": message,
        },
    )


def run_initial_quality_evaluation(
    *,
    task_dir: Path,
    state: OrchestratorState,
    production: CandidateProductionResult,
    specification: TaskSpecification,
    compiled_specification: CompiledSpecification,
) -> InitialQualityResult:
    """
    Run the explicit Phase 4 stages:

        COLLECT_EVIDENCE
            ->
        EVALUATE

    Execution evidence is collected exactly once.
    Evaluation then reuses that same CandidateAttempt.
    """

    task_dir = task_dir.resolve()

    schema_expectations = (
        infer_schema_expectations(
            task_dir=task_dir,
            candidate_schema_fields=(
                specification
                .candidate_schema_fields
            ),
            required_dtypes=(
                compiled_specification
                .required_dtypes
            ),
        )
    )

    attempts: dict[
        int,
        CandidateAttempt,
    ] = {}

    # ==================================================
    # COLLECT EVIDENCE
    # ==================================================

    stage = PipelineStage.COLLECT_EVIDENCE

    state.start_stage(stage)

    for item in production.executed_items:
        if (
            item.execution is None
            or item.workspace is None
            or item.solver_path is None
        ):
            continue

        candidate_id = (
            item.candidate.candidate_id
        )

        attempt = (
            collect_execution_evidence(
                execution=item.execution,
                output_dir=(
                    item.workspace.output_dir
                ),
                required_output_paths=(
                    specification
                    .required_output_paths
                ),
                attempt_number=1,
                solver_path=(
                    item.solver_path
                ),
            )
        )

        attempts[
            candidate_id
        ] = attempt

    if not attempts:
        reason = (
            "No executed candidate was available "
            "for evidence collection."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise QualityPipelineError(
            reason
        )

    state.complete_stage(
        stage,
        detail=(
            f"{len(attempts)} candidate "
            "attempt(s) collected"
        ),
    )

    # ==================================================
    # EVALUATE
    # ==================================================

    stage = PipelineStage.EVALUATE

    state.start_stage(stage)

    evaluated_count = 0

    for item in production.executed_items:
        candidate_id = (
            item.candidate.candidate_id
        )

        attempt = attempts.get(
            candidate_id
        )

        if attempt is None:
            continue

        if item.workspace is None:
            continue

        try:
            evaluate_collected_attempt(
                candidate=item.candidate,
                attempt=attempt,
                output_dir=(
                    item.workspace.output_dir
                ),
                required_output_paths=(
                    specification
                    .required_output_paths
                ),
                required_columns=(
                    compiled_specification
                    .required_columns
                ),
                schema_expectations=(
                    schema_expectations
                ),
                compiled_specification=(
                    compiled_specification
                ),
            )

        except Exception as exc:
            attempt.structural_evidence.append(
                _evaluation_failure_evidence(
                    exc
                )
            )

            if (
                item.candidate.latest_attempt()
                is not attempt
            ):
                item.candidate.add_attempt(
                    attempt
                )

            item.candidate.current_status = (
                "evaluation_failed"
            )

            item.candidate.hard_failures = [
                (
                    "Candidate evaluation "
                    "failed unexpectedly."
                )
            ]

        evaluated_count += 1

    if evaluated_count == 0:
        reason = (
            "No collected candidate attempt "
            "could be evaluated."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise QualityPipelineError(
            reason
        )

    state.complete_stage(
        stage,
        detail=(
            f"{evaluated_count} candidate "
            "attempt(s) evaluated"
        ),
    )

    return InitialQualityResult(
        production=production,
        schema_expectations=(
            schema_expectations
        ),
        attempts_by_candidate_id=(
            attempts
        ),
    )