from __future__ import annotations

import json
from typing import Any

from agent.spec_enrichment import (
    SpecificationEnrichment,
)


EXPECTED_KEYS = {
    "deliverables",
    "required_columns",
    "required_row_rules",
    "ordering_rules",
    "units",
    "conventions",
    "edge_cases",
    "invariants",
    "assumptions",
    "unresolved_questions",
    "difficulty",
    "reasoning_summary",
}


def _strip_code_fence(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


def parse_spec_enrichment(
    text: str,
) -> SpecificationEnrichment:
    cleaned = _strip_code_fence(text)

    try:
        payload = json.loads(cleaned)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Specification model response "
            "is not valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "Specification response must "
            "be a JSON object."
        )

    keys = set(payload)

    missing = EXPECTED_KEYS - keys
    extra = keys - EXPECTED_KEYS

    if missing:
        raise ValueError(
            f"Missing specification keys: "
            f"{sorted(missing)}"
        )

    if extra:
        raise ValueError(
            f"Unexpected specification keys: "
            f"{sorted(extra)}"
        )

    list_fields = [
        "deliverables",
        "required_columns",
        "required_row_rules",
        "ordering_rules",
        "conventions",
        "edge_cases",
        "invariants",
        "assumptions",
        "unresolved_questions",
    ]

    for field in list_fields:
        if not isinstance(
            payload[field],
            list,
        ):
            raise ValueError(
                f"{field} must be a list."
            )

    if not isinstance(
        payload["units"],
        dict,
    ):
        raise ValueError(
            "units must be an object."
        )

    if payload["difficulty"] not in {
        "easy",
        "medium",
        "hard",
    }:
        raise ValueError(
            "difficulty must be easy, "
            "medium, or hard."
        )

    if not isinstance(
        payload["reasoning_summary"],
        str,
    ):
        raise ValueError(
            "reasoning_summary must "
            "be a string."
        )

    return SpecificationEnrichment(
        deliverables=payload[
            "deliverables"
        ],
        required_columns=payload[
            "required_columns"
        ],
        required_row_rules=payload[
            "required_row_rules"
        ],
        ordering_rules=payload[
            "ordering_rules"
        ],
        units=payload["units"],
        conventions=payload[
            "conventions"
        ],
        edge_cases=payload[
            "edge_cases"
        ],
        invariants=payload[
            "invariants"
        ],
        assumptions=payload[
            "assumptions"
        ],
        unresolved_questions=payload[
            "unresolved_questions"
        ],
        difficulty=payload[
            "difficulty"
        ],
        reasoning_summary=payload[
            "reasoning_summary"
        ],
    )