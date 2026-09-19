from __future__ import annotations

import json
import re

from agent.planner import (
    CandidateStrategy,
    PlannerOutput,
)


_REQUIRED_KEYS = {
    "task_summary",
    "shared_requirements",
    "candidate_strategies",
    "final_checks",
}


def _extract_planner_json(
    text: str,
) -> dict:
    cleaned = str(text).strip()

    # Reasoning-enabled model responses can contain a hidden-style
    # thinking wrapper or a harmless explanatory preamble. Keep the
    # planner contract strict, but recover the intended JSON object.
    cleaned = re.sub(
        r"(?is)<think>.*?</think>",
        "",
        cleaned,
    ).strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None

    if (
        isinstance(payload, dict)
        and set(payload) == _REQUIRED_KEYS
    ):
        return payload

    decoder = json.JSONDecoder()

    for index, character in enumerate(cleaned):
        if character != "{":
            continue

        try:
            candidate, _ = decoder.raw_decode(
                cleaned[index:]
            )
        except json.JSONDecodeError:
            continue

        if (
            isinstance(candidate, dict)
            and set(candidate) == _REQUIRED_KEYS
        ):
            return candidate

    raise ValueError(
        "Planner response is not valid planner JSON."
    )


def parse_planner_output(
    text: str,
    expected_candidates: int,
) -> PlannerOutput:

    payload = _extract_planner_json(
        text
    )

    if set(payload) != _REQUIRED_KEYS:
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