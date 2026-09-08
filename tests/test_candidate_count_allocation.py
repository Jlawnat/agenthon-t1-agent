from __future__ import annotations

from types import SimpleNamespace
import unittest

from agent.orchestrator import (
    _resolve_candidate_count,
)


class CandidateCountAllocationTests(
    unittest.TestCase
):
    def test_explicit_override_wins(
        self,
    ) -> None:
        plan = SimpleNamespace(
            task_plan=SimpleNamespace(
                candidate_count=1,
            )
        )

        self.assertEqual(
            _resolve_candidate_count(
                requested=3,
                plan=plan,
            ),
            3,
        )

    def test_planned_candidate_count_used_when_not_overridden(
        self,
    ) -> None:
        for expected in (1, 2, 3):
            with self.subTest(
                expected=expected
            ):
                plan = SimpleNamespace(
                    task_plan=SimpleNamespace(
                        candidate_count=expected,
                    )
                )

                self.assertEqual(
                    _resolve_candidate_count(
                        requested=None,
                        plan=plan,
                    ),
                    expected,
                )

    def test_generic_plan_preserves_safe_default(
        self,
    ) -> None:
        self.assertEqual(
            _resolve_candidate_count(
                requested=None,
                plan={
                    "steps": [
                        "read",
                        "compute",
                        "write",
                    ],
                },
            ),
            3,
        )


if __name__ == "__main__":
    unittest.main()