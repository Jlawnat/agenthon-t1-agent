from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.candidate_production import (
    CandidateProductionItem,
    CandidateProductionResult,
)
from agent.candidate_selector import (
    SelectionResult,
    select_best_candidate,
)
from agent.compiled_specification import CompiledSpecification
from agent.orchestrator_state import (
    OrchestratorState,
    PipelineStage,
)
from agent.run_budget import RunBudgetExceeded
from agent.run_context import RunContext
from agent.semantic_selection import (
    SemanticCandidate,
    SemanticSelectionRequest,
    SemanticSelectionResult,
)
from agent.specification import TaskSpecification


class SelectionPipelineError(RuntimeError):
    """Raised when no admissible candidate can be selected."""


SemanticCompareFunction = Callable[
    [SemanticSelectionRequest],
    SemanticSelectionResult,
]


def _semantic_candidates(
    *,
    production: CandidateProductionResult,
    selection: SelectionResult,
) -> tuple[SemanticCandidate, ...]:
    valid_ids = {
        scorecard.candidate_id
        for scorecard in selection.scorecards
        if scorecard.is_valid
    }

    deterministic_rank = {
        candidate_id: index
        for index, candidate_id in enumerate(
            selection.ranked_candidate_ids,
            start=1,
        )
    }

    candidates: list[SemanticCandidate] = []

    for item in production.items:
        candidate = item.candidate

        if candidate.candidate_id not in valid_ids:
            continue

        source_code = (
            item.validated_code
            or item.raw_code
            or ""
        )

        if not source_code.strip():
            continue

        attempt = candidate.latest_attempt()

        if attempt is None:
            continue

        evidence = tuple(
            evidence_item.to_dict()
            for evidence_item in (
                attempt.structural_evidence
                + attempt.quant_evidence
            )
        )

        candidates.append(
            SemanticCandidate(
                candidate_id=candidate.candidate_id,
                approach_name=candidate.approach_name,
                source_code=source_code,
                evidence=evidence,
                deterministic_rank=deterministic_rank.get(
                    candidate.candidate_id,
                    999,
                ),
            )
        )

    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.deterministic_rank,
                item.candidate_id,
            ),
        )
    )


def _apply_semantic_choice(
    *,
    candidates,
    selection: SelectionResult,
    semantic: SemanticSelectionResult,
) -> SelectionResult:
    selected_id = semantic.selected_candidate_id

    valid_ids = {
        scorecard.candidate_id
        for scorecard in selection.scorecards
        if scorecard.is_valid
    }

    if selected_id not in valid_ids:
        return selection

    for candidate in candidates:
        candidate.selected = (
            candidate.candidate_id
            == selected_id
        )

        if candidate.candidate_id == selected_id:
            candidate.strategy[
                "semantic_selection"
            ] = {
                "confidence": semantic.confidence,
                "rationale": list(
                    semantic.rationale
                ),
            }

    reordered = (
        selected_id,
        *(
            candidate_id
            for candidate_id
            in selection.ranked_candidate_ids
            if candidate_id != selected_id
        ),
    )

    return SelectionResult(
        selected_candidate_id=selected_id,
        ranked_candidate_ids=tuple(reordered),
        first_candidate_valid=selection.first_candidate_valid,
        any_of_three_valid=selection.any_of_three_valid,
        scorecards=selection.scorecards,
    )


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
    semantic_compare: SemanticCompareFunction | None = None,
    semantic_compare_uses_model_budget: bool = True,
    run_context: RunContext | None = None,
    specification: TaskSpecification | None = None,
    compiled_specification: CompiledSpecification | None = None,
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

    semantic_candidates = _semantic_candidates(
        production=production,
        selection=selection,
    )

    if (
        len(semantic_candidates) >= 2
        and semantic_compare is not None
        and specification is not None
        and compiled_specification is not None
    ):
        budget_reserved = False

        if semantic_compare_uses_model_budget:
            if run_context is not None:
                try:
                    run_context.budget.reserve_model_call()
                    budget_reserved = True
                except RunBudgetExceeded:
                    budget_reserved = False
        else:
            budget_reserved = True

        if budget_reserved:
            request = SemanticSelectionRequest(
                instruction_text=specification.instruction_text,
                compiled_specification=compiled_specification.to_dict(),
                candidates=semantic_candidates,
            )

            try:
                semantic = semantic_compare(request)
            except Exception:
                semantic = None

            if semantic is not None:
                if (
                    run_context is not None
                    and semantic.tokens_used
                ):
                    run_context.budget.record_tokens(
                        semantic.tokens_used
                    )

                selection = _apply_semantic_choice(
                    candidates=candidates,
                    selection=selection,
                    semantic=semantic,
                )

                selected_id = (
                    selection.selected_candidate_id
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