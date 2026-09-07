from __future__ import annotations

import unittest

from agent.budget import (
    RepairBudget,
    RepairBudgetExceeded,
)
from agent.candidate_contract import (
    CandidateAttempt,
    VerificationEvidence,
)
from agent.targeted_repair import (
    FailureLabel,
    UnsafeRepairEvidenceError,
    build_repair_brief,
    classify_failure,
    sanitize_public_traceback,
)


def hard_failure(
    name: str,
    *,
    message: str = "failure",
    details: dict | None = None,
) -> VerificationEvidence:
    return VerificationEvidence(
        name=name,
        status="fail",
        severity="hard_fail",
        message=message,
        details=details or {},
    )


def warning(
    name: str,
) -> VerificationEvidence:
    return VerificationEvidence(
        name=name,
        status="warning",
        severity="warning",
        message="warning",
    )


class TargetedRepairTests(
    unittest.TestCase
):

    def test_syntax_failure(self) -> None:
        attempt = CandidateAttempt(
            attempt_number=1,
            stderr=(
                "SyntaxError: invalid syntax"
            ),
        )

        self.assertEqual(
            classify_failure(attempt),
            FailureLabel.SYNTAX,
        )

    def test_import_failure(self) -> None:
        attempt = CandidateAttempt(
            attempt_number=1,
            return_code=1,
            stderr=(
                "ModuleNotFoundError: "
                "No module named 'abc'"
            ),
        )

        self.assertEqual(
            classify_failure(attempt),
            FailureLabel.IMPORT,
        )

    def test_runtime_failure(self) -> None:
        attempt = CandidateAttempt(
            attempt_number=1,
            return_code=1,
            stderr="ValueError: bad value",
        )

        self.assertEqual(
            classify_failure(attempt),
            FailureLabel.RUNTIME,
        )

    def test_missing_file_failure(self) -> None:
        attempt = CandidateAttempt(
            attempt_number=1,
            return_code=0,
        )

        attempt.structural_evidence.append(
            hard_failure(
                "required_outputs",
                details={
                    "missing": [
                        "result.csv",
                    ],
                },
            )
        )

        self.assertEqual(
            classify_failure(attempt),
            FailureLabel.MISSING_FILE,
        )

    def test_schema_failure(self) -> None:
        attempt = CandidateAttempt(
            attempt_number=1,
            return_code=0,
        )

        attempt.structural_evidence.append(
            hard_failure(
                "required_columns",
            )
        )

        self.assertEqual(
            classify_failure(attempt),
            FailureLabel.SCHEMA,
        )

    def test_dtype_failure(self) -> None:
        attempt = CandidateAttempt(
            attempt_number=1,
            return_code=0,
        )

        attempt.structural_evidence.append(
            hard_failure(
                "dtype_mismatch",
            )
        )

        self.assertEqual(
            classify_failure(attempt),
            FailureLabel.DTYPE,
        )

    def test_numerical_failure(self) -> None:
        attempt = CandidateAttempt(
            attempt_number=1,
            return_code=0,
        )

        attempt.quant_evidence.append(
            hard_failure(
                "nav_reconciliation",
            )
        )

        self.assertEqual(
            classify_failure(attempt),
            (
                FailureLabel
                .NUMERICAL_RECONCILIATION
            ),
        )

    def test_runtime_beats_missing_file(
        self,
    ) -> None:
        attempt = CandidateAttempt(
            attempt_number=1,
            return_code=1,
            stderr="ValueError: failed",
        )

        attempt.structural_evidence.append(
            hard_failure(
                "required_outputs",
                details={
                    "missing": [
                        "result.csv",
                    ],
                },
            )
        )

        self.assertEqual(
            classify_failure(attempt),
            FailureLabel.RUNTIME,
        )

    def test_warning_does_not_repair(
        self,
    ) -> None:
        attempt = CandidateAttempt(
            attempt_number=1,
            return_code=0,
        )

        attempt.quant_evidence.append(
            warning(
                "uncertain_finance_check"
            )
        )

        budget = RepairBudget()

        brief = build_repair_brief(
            attempt=attempt,
            budget=budget,
        )

        self.assertIsNone(brief)
        self.assertEqual(
            budget.attempts_used,
            0,
        )

    def test_traceback_path_redacted(
        self,
    ) -> None:
        raw = (
            "Traceback:\n"
            '  File "/home/user/project/'
            'candidate/solver.py", line 4\n'
            "ValueError: failure"
        )

        cleaned = (
            sanitize_public_traceback(
                raw
            )
        )

        self.assertIn(
            '<candidate>/solver.py',
            cleaned,
        )

        self.assertNotIn(
            "/home/user/project",
            cleaned,
        )

    def test_forbidden_traceback_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            UnsafeRepairEvidenceError
        ):
            sanitize_public_traceback(
                "Read reward.json"
            )

    def test_traceback_is_bounded(
        self,
    ) -> None:
        raw = "\n".join(
            f"line {index}"
            for index in range(50)
        )

        cleaned = (
            sanitize_public_traceback(
                raw,
                max_lines=5,
                max_chars=1000,
            )
        )

        self.assertIn(
            "truncated",
            cleaned,
        )

        self.assertLessEqual(
            len(cleaned.splitlines()),
            6,
        )

    def test_missing_file_brief(
        self,
    ) -> None:
        attempt = CandidateAttempt(
            attempt_number=4,
            return_code=0,
        )

        attempt.structural_evidence.append(
            hard_failure(
                "required_outputs",
                details={
                    "missing": [
                        "submission.csv",
                    ],
                },
            )
        )

        budget = RepairBudget(
            max_attempts=2,
        )

        brief = build_repair_brief(
            attempt=attempt,
            budget=budget,
        )

        self.assertIsNotNone(brief)

        assert brief is not None

        self.assertEqual(
            brief.failure_label,
            "missing_file",
        )

        self.assertEqual(
            brief.strategy,
            "repair_missing_deliverable",
        )

        self.assertEqual(
            brief.schema_delta[
                "missing_files"
            ],
            ["submission.csv"],
        )

        self.assertEqual(
            brief.source_attempt_number,
            4,
        )

        self.assertEqual(
            brief.repair_attempt_number,
            1,
        )

    def test_quant_invariant_brief(
        self,
    ) -> None:
        attempt = CandidateAttempt(
            attempt_number=1,
            return_code=0,
        )

        attempt.quant_evidence.append(
            hard_failure(
                "nav_reconciliation",
                message=(
                    "NAV does not reconcile."
                ),
                details={
                    "violation_count": 2,
                    "tolerance": 1e-8,
                },
            )
        )

        brief = build_repair_brief(
            attempt=attempt,
            budget=RepairBudget(),
        )

        self.assertIsNotNone(brief)

        assert brief is not None

        self.assertEqual(
            brief.failure_label,
            "numerical_reconciliation",
        )

        self.assertEqual(
            len(
                brief.relevant_invariants
            ),
            1,
        )

        self.assertEqual(
            brief.relevant_invariants[
                0
            ]["name"],
            "nav_reconciliation",
        )

    def test_unsafe_evidence_does_not_use_budget(
        self,
    ) -> None:
        attempt = CandidateAttempt(
            attempt_number=1,
            return_code=1,
            stderr=(
                "oracle reference output "
                "was accessed"
            ),
        )

        budget = RepairBudget()

        with self.assertRaises(
            UnsafeRepairEvidenceError
        ):
            build_repair_brief(
                attempt=attempt,
                budget=budget,
            )

        self.assertEqual(
            budget.attempts_used,
            0,
        )

    def test_attempt_budget_cap(
        self,
    ) -> None:
        budget = RepairBudget(
            max_attempts=1,
        )

        budget.reserve_attempt()

        with self.assertRaises(
            RepairBudgetExceeded
        ):
            budget.reserve_attempt()

    def test_token_budget_cap(
        self,
    ) -> None:
        budget = RepairBudget(
            max_total_tokens=100,
        )

        with self.assertRaises(
            RepairBudgetExceeded
        ):
            budget.reserve_attempt(
                estimated_tokens=101,
            )

    def test_wall_time_budget_cap(
        self,
    ) -> None:
        budget = RepairBudget(
            max_wall_seconds=10.0,
        )

        with self.assertRaises(
            RepairBudgetExceeded
        ):
            budget.reserve_attempt(
                estimated_wall_seconds=11.0,
            )

    def test_recorded_usage_affects_budget(
        self,
    ) -> None:
        budget = RepairBudget(
            max_total_tokens=100,
        )

        budget.record_usage(
            tokens=80,
        )

        self.assertTrue(
            budget.can_reserve(
                estimated_tokens=20,
            )
        )

        self.assertFalse(
            budget.can_reserve(
                estimated_tokens=21,
            )
        )


if __name__ == "__main__":
    unittest.main()