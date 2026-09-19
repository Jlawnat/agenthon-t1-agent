from __future__ import annotations

from dataclasses import dataclass

from agent.candidate_production import (
    CandidateProductionItem,
    CandidateProductionResult,
)
from agent.candidate_selector import (
    SelectionResult,
    select_best_candidate,
)
from agent.orchestrator_state import (
    OrchestratorState,
    PipelineStage,
)


class SelectionPipelineError(RuntimeError):
    """Raised when no admissible candidate can be selected."""


@dataclass(frozen=True)
class SelectionPipelineResult:
    selection: SelectionResult
    selected_item: CandidateProductionItem

    @property
    def selected_candidate_id(self) -> int:
        return (
            self.selected_item
            .candidate
            .candidate_id
        )


def run_selection_stage(
    *,
    state: OrchestratorState,
    production: CandidateProductionResult,
    allow_best_effort: bool = False,
) -> SelectionPipelineResult:
    """
    Run the Phase 4 SELECT stage using the existing
    deterministic Phase 3.12 selector.

    Selection fails closed when no valid candidate exists.
    """

    stage = PipelineStage.SELECT

    state.start_stage(
        stage
    )

    candidates = (
        production.candidates
    )

    if not candidates:
        reason = (
            "Candidate selection received "
            "an empty candidate set."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise SelectionPipelineError(
            reason
        )

    selection = (
        select_best_candidate(
            candidates
        )
    )

    selected_id = (
        selection
        .selected_candidate_id
    )

    if (
        selected_id is None
        and allow_best_effort
    ):
        fallback = next(
            (
                scorecard
                for scorecard
                in selection.scorecards
                if (
                    scorecard.admissible_execution
                    and scorecard.structural_hard_failures == 0
                )
            ),
            None,
        )

        if fallback is not None:
            selected_id = (
                fallback.candidate_id
            )

            for candidate in candidates:
                candidate.selected = (
                    candidate.candidate_id
                    == selected_id
                )

            selection = SelectionResult(
                selected_candidate_id=(
                    selected_id
                ),
                ranked_candidate_ids=(
                    selection.ranked_candidate_ids
                ),
                first_candidate_valid=(
                    selection.first_candidate_valid
                ),
                any_of_three_valid=(
                    selection.any_of_three_valid
                ),
                scorecards=(
                    selection.scorecards
                ),
            )

    if selected_id is None:
        reason = (
            "No valid or structurally publishable candidate survived "
            "execution, evaluation, and targeted repair."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise SelectionPipelineError(
            reason
        )

    matches = [
        item
        for item in production.items
        if (
            item.candidate.candidate_id
            == selected_id
        )
    ]

    if len(matches) != 1:
        reason = (
            "Selected candidate could not "
            "be mapped uniquely to its "
            "production artifact."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise SelectionPipelineError(
            reason
        )

    selected_item = (
        matches[0]
    )

    latest_attempt = (
        selected_item
        .candidate
        .latest_attempt()
    )

    if latest_attempt is None:
        reason = (
            "Selected candidate has no "
            "evaluated attempt."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise SelectionPipelineError(
            reason
        )

    if (
        selected_item.workspace is None
        or selected_item.solver_path is None
        or selected_item.execution is None
    ):
        reason = (
            "Selected candidate is missing "
            "its current execution artifacts."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise SelectionPipelineError(
            reason
        )

    state.complete_stage(
        stage,
        detail=(
            f"selected candidate {selected_id}; "
            f"pass@1="
            f"{selection.first_candidate_valid}; "
            f"any_of_three="
            f"{selection.any_of_three_valid}"
        ),
    )

    return SelectionPipelineResult(
        selection=selection,
        selected_item=selected_item,
    )