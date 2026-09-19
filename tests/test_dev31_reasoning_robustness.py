from __future__ import annotations

import json
import unittest

from agent.planner_validator import (
    parse_planner_output,
)


def _payload() -> dict:
    return {
        "task_summary": "test",
        "shared_requirements": ["preserve contract"],
        "candidate_strategies": [
            {
                "candidate_id": 1,
                "approach_name": "primary",
                "implementation_steps": ["read", "compute", "write"],
                "verification_steps": ["check"],
                "numerical_method": None,
                "risks": [],
            },
            {
                "candidate_id": 2,
                "approach_name": "independent",
                "implementation_steps": ["read", "compute independently", "write"],
                "verification_steps": ["reconcile"],
                "numerical_method": None,
                "risks": [],
            },
        ],
        "final_checks": ["schema"],
    }


class ReasoningRobustPlannerTests(unittest.TestCase):
    def test_planner_accepts_reasoning_wrapper_before_json(self) -> None:
        text = (
            "<think>brief internal reasoning</think>\n"
            "Here is the requested plan:\n"
            + json.dumps(_payload())
        )

        result = parse_planner_output(
            text,
            expected_candidates=2,
        )

        self.assertEqual(
            len(result.candidate_strategies),
            2,
        )
        self.assertEqual(
            result.candidate_strategies[1].approach_name,
            "independent",
        )

    def test_planner_remains_strict_about_schema(self) -> None:
        bad = _payload()
        bad["unexpected"] = True

        with self.assertRaises(ValueError):
            parse_planner_output(
                json.dumps(bad),
                expected_candidates=2,
            )


if __name__ == "__main__":
    unittest.main()
