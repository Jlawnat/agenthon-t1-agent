from __future__ import annotations

import unittest

from agent.run_budget import (
    RunBudget,
    RunBudgetExceeded,
)


class FakeClock:

    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(
        self,
        seconds: float,
    ) -> None:
        self.value += seconds


class RunBudgetTests(unittest.TestCase):

    def test_budget_tracks_elapsed_and_remaining_time(
        self,
    ) -> None:
        clock = FakeClock()

        budget = RunBudget(
            total_wall_seconds=100,
            safety_margin_seconds=10,
            _clock=clock,
        )

        self.assertEqual(
            budget.usable_remaining_seconds,
            90,
        )

        clock.advance(25)

        self.assertEqual(
            budget.elapsed_seconds,
            25,
        )

        self.assertEqual(
            budget.hard_remaining_seconds,
            75,
        )

        self.assertEqual(
            budget.usable_remaining_seconds,
            65,
        )

    def test_safety_margin_closes_ordinary_work_early(
        self,
    ) -> None:
        clock = FakeClock()

        budget = RunBudget(
            total_wall_seconds=100,
            safety_margin_seconds=10,
            _clock=clock,
        )

        clock.advance(90)

        self.assertTrue(
            budget.exhausted
        )

        self.assertEqual(
            budget.hard_remaining_seconds,
            10,
        )

        self.assertEqual(
            budget.usable_remaining_seconds,
            0,
        )

    def test_timeout_is_capped_by_global_deadline(
        self,
    ) -> None:
        clock = FakeClock()

        budget = RunBudget(
            total_wall_seconds=100,
            safety_margin_seconds=10,
            _clock=clock,
        )

        clock.advance(70)

        timeout = budget.bounded_timeout(
            120
        )

        self.assertEqual(
            timeout,
            20,
        )

    def test_candidate_attempt_budget_is_bounded(
        self,
    ) -> None:
        clock = FakeClock()

        budget = RunBudget(
            total_wall_seconds=100,
            max_candidate_attempts=2,
            _clock=clock,
        )

        self.assertEqual(
            budget.reserve_candidate_attempt(),
            1,
        )

        self.assertEqual(
            budget.reserve_candidate_attempt(),
            2,
        )

        with self.assertRaises(
            RunBudgetExceeded
        ):
            budget.reserve_candidate_attempt()

    def test_model_call_budget_is_bounded(
        self,
    ) -> None:
        clock = FakeClock()

        budget = RunBudget(
            total_wall_seconds=100,
            max_model_calls=1,
            _clock=clock,
        )

        self.assertEqual(
            budget.reserve_model_call(),
            1,
        )

        with self.assertRaises(
            RunBudgetExceeded
        ):
            budget.reserve_model_call()

    def test_estimated_tokens_cannot_cross_limit(
        self,
    ) -> None:
        clock = FakeClock()

        budget = RunBudget(
            total_wall_seconds=100,
            max_total_tokens=1000,
            _clock=clock,
        )

        budget.record_tokens(800)

        with self.assertRaises(
            RunBudgetExceeded
        ):
            budget.reserve_model_call(
                estimated_tokens=250
            )

        self.assertEqual(
            budget.model_calls_used,
            0,
        )

    def test_deadline_blocks_future_work(
        self,
    ) -> None:
        clock = FakeClock()

        budget = RunBudget(
            total_wall_seconds=30,
            safety_margin_seconds=5,
            _clock=clock,
        )

        clock.advance(25)

        with self.assertRaises(
            RunBudgetExceeded
        ):
            budget.reserve_candidate_attempt()

        with self.assertRaises(
            RunBudgetExceeded
        ):
            budget.reserve_model_call()

    def test_snapshot_is_serializable_state(
        self,
    ) -> None:
        clock = FakeClock()

        budget = RunBudget(
            total_wall_seconds=100,
            safety_margin_seconds=10,
            max_candidate_attempts=3,
            max_model_calls=4,
            max_total_tokens=5000,
            _clock=clock,
        )

        budget.reserve_candidate_attempt()
        budget.reserve_model_call(
            estimated_tokens=100
        )
        budget.record_tokens(80)

        payload = budget.snapshot()

        self.assertEqual(
            payload["candidate_attempts_used"],
            1,
        )

        self.assertEqual(
            payload["model_calls_used"],
            1,
        )

        self.assertEqual(
            payload["tokens_used"],
            80,
        )

        self.assertEqual(
            payload["remaining_tokens"],
            4920,
        )


if __name__ == "__main__":
    unittest.main()