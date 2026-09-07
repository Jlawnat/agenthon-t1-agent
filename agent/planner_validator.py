from __future__ import annotations

import json

from agent.planner import (
    CandidateStrategy,
    PlannerOutput,
)


def parse_planner_output(
    text: str,
    expected_candidates: int,
) -> PlannerOutput:

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    try:
        payload = json.loads(cleaned)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Planner response is not valid JSON."
        ) from exc

    required_keys = {
        "task_summary",
        "shared_requirements",
        "candidate_strategies",
        "final_checks",
    }

    if set(payload) != required_keys:
        raise ValueError(
            "Planner response has incorrect top-level keys."
        )

    strategies = payload["candidate_strategies"]

    if not isinstance(strategies, list):
        raise ValueError(
            "candidate_strategies must be a list."
        )

    if len(strategies) != expected_candidates:
        raise ValueError(
            f"Expected {expected_candidates} candidates, "
            f"received {len(strategies)}."
        )

    parsed_strategies = []

    for index, item in enumerate(
        strategies,
        start=1,
    ):
        parsed_strategies.append(
            CandidateStrategy(
                candidate_id=item.get(
                    "candidate_id",
                    index,
                ),
                approach_name=item[
                    "approach_name"
                ],
                implementation_steps=item[
                    "implementation_steps"
                ],
                verification_steps=item[
                    "verification_steps"
                ],
                numerical_method=item.get(
                    "numerical_method"
                ),
                risks=item["risks"],
            )
        )

    return PlannerOutput(
        task_summary=payload[
            "task_summary"
        ],
        shared_requirements=payload[
            "shared_requirements"
        ],
        candidate_strategies=(
            parsed_strategies
        ),
        final_checks=payload[
            "final_checks"
        ],
    )