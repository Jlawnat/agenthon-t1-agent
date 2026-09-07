from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf
from typing import Any, Iterable

from agent.candidate_contract import (
    CandidateAttempt,
    CandidateRecord,
    VerificationEvidence,
)


@dataclass(frozen=True)
class CandidateScorecard:
    candidate_id: int
    latest_attempt_number: int | None

    admissible_execution: bool
    structural_hard_failures: int
    quant_hard_failures: int

    independent_consistency_passes: int
    warning_count: int
    repair_count: int

    runtime_seconds: float
    token_cost: int

    is_valid: bool
    rank_key: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionResult:
    selected_candidate_id: int | None
    ranked_candidate_ids: tuple[int, ...]
    first_candidate_valid: bool
    any_of_three_valid: bool
    scorecards: tuple[CandidateScorecard, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _latest_attempt(
    candidate: CandidateRecord,
) -> CandidateAttempt | None:
    return candidate.latest_attempt()


def _hard_failures(
    evidence: Iterable[VerificationEvidence],
) -> list[VerificationEvidence]:
    return [
        item
        for item in evidence
        if (
            item.status == "fail"
            and item.severity == "hard_fail"
        )
    ]


def _warning_count(
    attempt: CandidateAttempt,
) -> int:
    evidence = (
        attempt.structural_evidence
        + attempt.quant_evidence
    )

    return sum(
        1
        for item in evidence
        if (
            item.status == "warning"
            or item.severity == "warning"
        )
    )


def _is_independent_consistency_pass(
    item: VerificationEvidence,
) -> bool:
    if item.status != "pass":
        return False

    details = item.details or {}

    if details.get(
        "independent_recomputation"
    ) is True:
        return True

    provenance = str(
        details.get("provenance", "")
    ).strip().lower()

    name = str(
        item.name
    ).strip().lower()

    consistency_tokens = (
        "reconciliation",
        "consistency",
        "parity",
        "identity",
        "compounding",
        "accounting",
    )

    return (
        provenance
        == "deterministic_output_invariant"
        and any(
            token in name
            for token in consistency_tokens
        )
    )


def _independent_consistency_passes(
    attempt: CandidateAttempt,
) -> int:
    return sum(
        1
        for item in attempt.quant_evidence
        if _is_independent_consistency_pass(
            item
        )
    )


def _attempt_is_admissible(
    attempt: CandidateAttempt | None,
) -> bool:
    if attempt is None:
        return False

    return (
        not attempt.timed_out
        and attempt.return_code == 0
    )


def _candidate_token_cost(
    candidate: CandidateRecord,
) -> int:
    strategy = candidate.strategy or {}

    possible_keys = (
        "tokens_used",
        "token_usage",
        "model_tokens",
        "total_tokens",
    )

    for key in possible_keys:
        value = strategy.get(key)

        if isinstance(value, bool):
            continue

        if isinstance(
            value,
            (int, float),
        ):
            return max(
                0,
                int(value),
            )

    return 0


def _candidate_runtime_seconds(
    candidate: CandidateRecord,
) -> float:
    runtime = 0.0
    found = False

    for attempt in candidate.attempts:
        if attempt.runtime_seconds is None:
            continue

        runtime += max(
            0.0,
            float(
                attempt.runtime_seconds
            ),
        )

        found = True

    if not found:
        return inf

    return runtime


_RUNTIME_RANK_BUCKET_SECONDS = 1.0


def _runtime_rank_bucket(
    runtime_seconds: float,
) -> int | float:
    """Return a stable runtime tier for candidate ranking.

    Raw wall-clock timings are intentionally not used directly because
    millisecond-level OS/process jitter can change replay ordering even when
    the task, seed, generated code, and evidence are identical.

    A one-second bucket preserves the Phase 3.12 preference for materially
    faster candidates (for example, 2s beats 8s) while treating tiny runtime
    differences as ties. candidate_id remains the final deterministic
    tie-breaker.
    """
    if runtime_seconds == inf:
        return inf

    runtime = max(
        0.0,
        float(runtime_seconds),
    )

    return int(
        runtime
        // _RUNTIME_RANK_BUCKET_SECONDS
    )


def _build_rank_key(
    scorecard: CandidateScorecard,
) -> tuple[Any, ...]:
    return (
        0
        if scorecard.admissible_execution
        else 1,

        scorecard.structural_hard_failures,

        scorecard.quant_hard_failures,

        -scorecard.independent_consistency_passes,

        scorecard.warning_count,

        scorecard.repair_count,

        scorecard.token_cost,

        _runtime_rank_bucket(
            scorecard.runtime_seconds
        ),

        scorecard.candidate_id,
    )


def score_candidate(
    candidate: CandidateRecord,
) -> CandidateScorecard:
    attempt = _latest_attempt(
        candidate
    )

    if attempt is None:
        base = CandidateScorecard(
            candidate_id=(
                candidate.candidate_id
            ),
            latest_attempt_number=None,
            admissible_execution=False,
            structural_hard_failures=1,
            quant_hard_failures=1,
            independent_consistency_passes=0,
            warning_count=0,
            repair_count=0,
            runtime_seconds=inf,
            token_cost=(
                _candidate_token_cost(
                    candidate
                )
            ),
            is_valid=False,
            rank_key=(),
        )

        return CandidateScorecard(
            **{
                **base.to_dict(),
                "rank_key": (
                    _build_rank_key(
                        base
                    )
                ),
            }
        )

    structural_hard_failures = len(
        _hard_failures(
            attempt.structural_evidence
        )
    )

    quant_hard_failures = len(
        _hard_failures(
            attempt.quant_evidence
        )
    )

    admissible_execution = (
        _attempt_is_admissible(
            attempt
        )
    )

    is_valid = (
        admissible_execution
        and structural_hard_failures == 0
        and quant_hard_failures == 0
    )

    base = CandidateScorecard(
        candidate_id=(
            candidate.candidate_id
        ),
        latest_attempt_number=(
            attempt.attempt_number
        ),
        admissible_execution=(
            admissible_execution
        ),
        structural_hard_failures=(
            structural_hard_failures
        ),
        quant_hard_failures=(
            quant_hard_failures
        ),
        independent_consistency_passes=(
            _independent_consistency_passes(
                attempt
            )
        ),
        warning_count=(
            _warning_count(
                attempt
            )
        ),
        repair_count=max(
            0,
            len(candidate.attempts) - 1,
        ),
        runtime_seconds=(
            _candidate_runtime_seconds(
                candidate
            )
        ),
        token_cost=(
            _candidate_token_cost(
                candidate
            )
        ),
        is_valid=is_valid,
        rank_key=(),
    )

    return CandidateScorecard(
        **{
            **base.to_dict(),
            "rank_key": (
                _build_rank_key(
                    base
                )
            ),
        }
    )


def rank_candidates(
    candidates: list[CandidateRecord],
) -> list[CandidateScorecard]:
    scorecards = [
        score_candidate(
            candidate
        )
        for candidate in candidates
    ]

    return sorted(
        scorecards,
        key=lambda item: item.rank_key,
    )


def select_best_candidate(
    candidates: list[CandidateRecord],
) -> SelectionResult:
    ranked = rank_candidates(
        candidates
    )

    for candidate in candidates:
        candidate.selected = False

    selected_candidate_id = None

    for scorecard in ranked:
        if not scorecard.is_valid:
            continue

        selected_candidate_id = (
            scorecard.candidate_id
        )
        break

    if selected_candidate_id is not None:
        for candidate in candidates:
            if (
                candidate.candidate_id
                == selected_candidate_id
            ):
                candidate.selected = True
                break

    first_candidate_valid = False

    if candidates:
        first_candidate_valid = (
            score_candidate(
                candidates[0]
            ).is_valid
        )

    any_of_three_valid = any(
        score_candidate(
            candidate
        ).is_valid
        for candidate in candidates[:3]
    )

    return SelectionResult(
        selected_candidate_id=(
            selected_candidate_id
        ),
        ranked_candidate_ids=tuple(
            item.candidate_id
            for item in ranked
        ),
        first_candidate_valid=(
            first_candidate_valid
        ),
        any_of_three_valid=(
            any_of_three_valid
        ),
        scorecards=tuple(
            ranked
        ),
    )