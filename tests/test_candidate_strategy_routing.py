from __future__ import annotations

import unittest

from agent.runtime_adapters import (
    RuntimePlan,
    _StrategyEnvelope,
    _strategy_for_candidate,
)


class CandidateStrategyRoutingTests(
    unittest.TestCase
):

    def test_candidate_receives_matching_planner_strategy(
        self,
    ) -> None:
        plan = RuntimePlan(
            planning_specification=None,  # type: ignore[arg-type]
            task_plan=None,
            planner_output=None,
            skill_packs=(),
            data_inspections={},
            strategies=(
                _StrategyEnvelope(
                    payload={
                        "candidate_id": 1,
                        "approach_name": "analytical",
                        "numerical_method": "closed-form",
                        "verification_steps": [
                            "check analytical identity",
                        ],
                    }
                ),
                _StrategyEnvelope(
                    payload={
                        "candidate_id": 2,
                        "approach_name": "numerical",
                        "numerical_method": "root-finding",
                        "verification_steps": [
                            "check numerical convergence",
                        ],
                    }
                ),
            ),
        )

        first = _strategy_for_candidate(
            plan,
            candidate_id=1,
            candidate_seed=101,
        ).to_dict()

        second = _strategy_for_candidate(
            plan,
            candidate_id=2,
            candidate_seed=202,
        ).to_dict()

        self.assertEqual(
            first["approach_name"],
            "analytical",
        )
        self.assertEqual(
            second["approach_name"],
            "numerical",
        )

        self.assertEqual(
            first["numerical_method"],
            "closed-form",
        )
        self.assertEqual(
            second["numerical_method"],
            "root-finding",
        )

        self.assertEqual(
            first["candidate_id"],
            1,
        )
        self.assertEqual(
            second["candidate_id"],
            2,
        )

        self.assertEqual(
            first["candidate_seed"],
            101,
        )
        self.assertEqual(
            second["candidate_seed"],
            202,
        )

        self.assertNotEqual(
            first["approach_name"],
            second["approach_name"],
        )
        self.assertNotEqual(
            first["numerical_method"],
            second["numerical_method"],
        )


if __name__ == "__main__":
    unittest.main()