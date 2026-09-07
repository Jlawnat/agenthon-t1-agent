from __future__ import annotations

import unittest

from agent.candidate_contract import (
    CandidateAttempt,
    CandidateRecord,
    VerificationEvidence,
)
from agent.candidate_selector import (
    rank_candidates,
    score_candidate,
    select_best_candidate,
)


def _evidence(
    name: str,
    *,
    status: str,
    severity: str,
    message: str = "test evidence",
    **details,
) -> VerificationEvidence:
    return VerificationEvidence(
        name=name,
        status=status,
        severity=severity,
        message=message,
        details=details,
    )


def _candidate(
    candidate_id: int,
    *,
    return_code: int = 0,
    timed_out: bool = False,
    runtime_seconds: float = 1.0,
    structural_evidence=None,
    quant_evidence=None,
    strategy=None,
    attempts: int = 1,
) -> CandidateRecord:
    candidate = CandidateRecord(
        candidate_id=candidate_id,
        approach_name=f"candidate-{candidate_id}",
        strategy=dict(strategy or {}),
    )

    for attempt_number in range(
        1,
        attempts + 1,
    ):
        attempt = CandidateAttempt(
            attempt_number=attempt_number,
            return_code=return_code,
            timed_out=timed_out,
            runtime_seconds=runtime_seconds,
        )

        if attempt_number == attempts:
            attempt.structural_evidence.extend(
                structural_evidence or []
            )

            attempt.quant_evidence.extend(
                quant_evidence or []
            )

        candidate.add_attempt(attempt)

    return candidate


class CandidateSelectorAcceptanceTests(
    unittest.TestCase
):
    def test_crashed_candidate_can_never_win(
        self,
    ) -> None:
        crashed = _candidate(
            1,
            return_code=1,
            runtime_seconds=0.01,
        )

        valid = _candidate(
            2,
            return_code=0,
            runtime_seconds=10.0,
        )

        result = select_best_candidate(
            [crashed, valid]
        )

        self.assertEqual(
            result.selected_candidate_id,
            2,
        )

        self.assertTrue(valid.selected)
        self.assertFalse(crashed.selected)

    def test_missing_output_candidate_can_never_win(
        self,
    ) -> None:
        missing_output = _candidate(
            1,
            structural_evidence=[
                _evidence(
                    "required_outputs",
                    status="fail",
                    severity="hard_fail",
                    message=(
                        "Required output files "
                        "are missing."
                    ),
                    missing=["results.csv"],
                )
            ],
        )

        clean = _candidate(2)

        result = select_best_candidate(
            [missing_output, clean]
        )

        self.assertEqual(
            result.selected_candidate_id,
            2,
        )

    def test_hard_quant_failure_loses_to_clean_candidate(
        self,
    ) -> None:
        quant_failure = _candidate(
            1,
            quant_evidence=[
                _evidence(
                    "pnl_reconciliation",
                    status="fail",
                    severity="hard_fail",
                    message=(
                        "P&L does not reconcile."
                    ),
                )
            ],
        )

        clean = _candidate(2)

        result = select_best_candidate(
            [quant_failure, clean]
        )

        self.assertEqual(
            result.selected_candidate_id,
            2,
        )

    def test_independent_consistency_is_rewarded(
        self,
    ) -> None:
        independently_checked = _candidate(
            1,
            quant_evidence=[
                _evidence(
                    "nav_reconciliation",
                    status="pass",
                    severity="info",
                    provenance=(
                        "deterministic_output_invariant"
                    ),
                )
            ],
        )

        ordinary = _candidate(2)

        result = select_best_candidate(
            [
                ordinary,
                independently_checked,
            ]
        )

        self.assertEqual(
            result.selected_candidate_id,
            1,
        )

    def test_fewer_warnings_wins_when_quality_ties(
        self,
    ) -> None:
        warning_candidate = _candidate(
            1,
            quant_evidence=[
                _evidence(
                    "discount_factor_monotonicity",
                    status="warning",
                    severity="warning",
                )
            ],
        )

        clean = _candidate(2)

        result = select_best_candidate(
            [warning_candidate, clean]
        )

        self.assertEqual(
            result.selected_candidate_id,
            2,
        )

    def test_fewer_repairs_wins_when_quality_ties(
        self,
    ) -> None:
        repaired = _candidate(
            1,
            attempts=2,
        )

        first_pass = _candidate(
            2,
            attempts=1,
        )

        result = select_best_candidate(
            [repaired, first_pass]
        )

        self.assertEqual(
            result.selected_candidate_id,
            2,
        )

    def test_lower_token_cost_wins_after_quality_ties(
        self,
    ) -> None:
        expensive = _candidate(
            1,
            strategy={
                "tokens_used": 4000,
            },
        )

        cheaper = _candidate(
            2,
            strategy={
                "tokens_used": 1500,
            },
        )

        result = select_best_candidate(
            [expensive, cheaper]
        )

        self.assertEqual(
            result.selected_candidate_id,
            2,
        )

    def test_faster_candidate_wins_after_quality_and_cost_tie(
        self,
    ) -> None:
        slow = _candidate(
            1,
            runtime_seconds=8.0,
            strategy={
                "tokens_used": 1000,
            },
        )

        fast = _candidate(
            2,
            runtime_seconds=2.0,
            strategy={
                "tokens_used": 1000,
            },
        )

        result = select_best_candidate(
            [slow, fast]
        )

        self.assertEqual(
            result.selected_candidate_id,
            2,
        )

    def test_candidate_id_breaks_exact_tie_deterministically(
        self,
    ) -> None:
        high_id = _candidate(9)
        low_id = _candidate(3)

        first = select_best_candidate(
            [high_id, low_id]
        )

        second = select_best_candidate(
            [low_id, high_id]
        )

        self.assertEqual(
            first.selected_candidate_id,
            3,
        )

        self.assertEqual(
            second.selected_candidate_id,
            3,
        )

    def test_first_candidate_and_any_of_three_metrics(
        self,
    ) -> None:
        first = _candidate(
            1,
            return_code=1,
        )

        second = _candidate(
            2,
            structural_evidence=[
                _evidence(
                    "required_columns",
                    status="fail",
                    severity="hard_fail",
                )
            ],
        )

        third = _candidate(3)

        result = select_best_candidate(
            [first, second, third]
        )

        self.assertFalse(
            result.first_candidate_valid
        )

        self.assertTrue(
            result.any_of_three_valid
        )

        self.assertEqual(
            result.selected_candidate_id,
            3,
        )

    def test_any_of_three_only_uses_first_three_candidates(
        self,
    ) -> None:
        first = _candidate(
            1,
            return_code=1,
        )

        second = _candidate(
            2,
            return_code=1,
        )

        third = _candidate(
            3,
            return_code=1,
        )

        fourth = _candidate(4)

        result = select_best_candidate(
            [
                first,
                second,
                third,
                fourth,
            ]
        )

        self.assertFalse(
            result.any_of_three_valid
        )

        self.assertEqual(
            result.selected_candidate_id,
            4,
        )

    def test_no_valid_candidate_returns_none(
        self,
    ) -> None:
        first = _candidate(
            1,
            return_code=1,
        )

        second = _candidate(
            2,
            timed_out=True,
        )

        result = select_best_candidate(
            [first, second]
        )

        self.assertIsNone(
            result.selected_candidate_id
        )

        self.assertFalse(first.selected)
        self.assertFalse(second.selected)

    def test_scorecard_detects_hard_failures(
        self,
    ) -> None:
        candidate = _candidate(
            1,
            structural_evidence=[
                _evidence(
                    "required_columns",
                    status="fail",
                    severity="hard_fail",
                )
            ],
            quant_evidence=[
                _evidence(
                    "probability_bounds",
                    status="fail",
                    severity="hard_fail",
                )
            ],
        )

        scorecard = score_candidate(
            candidate
        )

        self.assertEqual(
            scorecard.structural_hard_failures,
            1,
        )

        self.assertEqual(
            scorecard.quant_hard_failures,
            1,
        )

        self.assertFalse(
            scorecard.is_valid
        )

    def test_ranking_is_deterministic(
        self,
    ) -> None:
        candidates = [
            _candidate(7),
            _candidate(2),
            _candidate(5),
        ]

        first = rank_candidates(
            candidates
        )

        second = rank_candidates(
            list(reversed(candidates))
        )

        self.assertEqual(
            [
                item.candidate_id
                for item in first
            ],
            [2, 5, 7],
        )

        self.assertEqual(
            [
                item.candidate_id
                for item in first
            ],
            [
                item.candidate_id
                for item in second
            ],
        )


if __name__ == "__main__":
    unittest.main()