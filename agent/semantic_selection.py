from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any


@dataclass(frozen=True)
class SemanticCandidate:
    candidate_id: int
    approach_name: str
    source_code: str
    evidence: tuple[dict[str, Any], ...]
    deterministic_rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticSelectionRequest:
    instruction_text: str
    compiled_specification: dict[str, Any]
    candidates: tuple[SemanticCandidate, ...]

    def eligible_ids(self) -> tuple[int, ...]:
        return tuple(
            candidate.candidate_id
            for candidate in self.candidates
        )


@dataclass(frozen=True)
class SemanticSelectionResult:
    selected_candidate_id: int
    confidence: str
    rationale: tuple[str, ...]
    tokens_used: int = 0


def _bounded_value(
    value: Any,
    *,
    depth: int = 0,
) -> Any:
    if depth >= 4:
        return str(value)[:500]

    if isinstance(value, str):
        return value[:2000]

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    if isinstance(value, dict):
        result: dict[str, Any] = {}

        for index, (key, item) in enumerate(value.items()):
            if index >= 40:
                break

            result[str(key)[:120]] = _bounded_value(
                item,
                depth=depth + 1,
            )

        return result

    if isinstance(value, (list, tuple)):
        return [
            _bounded_value(
                item,
                depth=depth + 1,
            )
            for item in list(value)[:40]
        ]

    return str(value)[:1000]


def build_semantic_selection_prompt(
    request: SemanticSelectionRequest,
) -> str:
    payload = {
        "instruction": request.instruction_text,
        "compiled_specification": _bounded_value(
            request.compiled_specification
        ),
        "eligible_candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "approach_name": candidate.approach_name,
                "deterministic_rank": candidate.deterministic_rank,
                "source_code": candidate.source_code[:30000],
                "execution_and_validation_evidence": _bounded_value(
                    candidate.evidence
                ),
            }
            for candidate in request.candidates
        ],
    }

    context = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    return f"""
You are the final semantic reviewer for a quantitative-finance coding agent.

Several candidate programs have ALREADY:
- passed Python/security validation,
- executed successfully,
- produced the required output structure,
- survived the agent's deterministic hard-failure checks.

Your only job is to choose which eligible candidate is most likely to
satisfy the EXPLICIT task correctly.

Do NOT rewrite code.
Do NOT propose a new solution.
Do NOT infer hidden checker logic, oracle values, canaries, reference
outputs, rewards, or grader secrets.
Use only the task instruction, compiled specification, candidate source
code, and sanitized execution/validation evidence supplied below.

Compare candidates in this priority order:
1. Exact compliance with the explicit task contract and output semantics.
2. Correct quantitative-finance mathematics and stated conventions.
3. Correct data cleaning, date/time alignment, causality, and aggregation.
4. Numerical stability, reproducibility, and edge-case handling.
5. Consistency with the supplied deterministic evidence.
6. Avoidance of unjustified assumptions or silent convention changes.

Do NOT prefer a candidate merely because it is shorter, faster, cheaper,
uses fewer tokens, or has a lower candidate id. The deterministic rank is
provided only as a final tie-breaker when semantic correctness appears
indistinguishable.

You MUST choose exactly one id from the eligible candidate ids.

Return ONLY one JSON object with exactly these keys:

{{
  "selected_candidate_id": <integer>,
  "confidence": "low|medium|high",
  "rationale": [
    "<brief task-grounded reason>",
    "<brief task-grounded reason>"
  ]
}}

SEMANTIC REVIEW CONTEXT:

{context}
""".strip()


_REQUIRED_KEYS = {
    "selected_candidate_id",
    "confidence",
    "rationale",
}


def _extract_json_object(
    text: str,
) -> dict[str, Any]:
    cleaned = re.sub(
        r"(?is)<think>.*?</think>",
        "",
        str(text),
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
        "Semantic selector response is not valid selector JSON."
    )


def parse_semantic_selection_response(
    text: str,
    *,
    eligible_ids: tuple[int, ...],
    tokens_used: int = 0,
) -> SemanticSelectionResult:
    payload = _extract_json_object(
        text
    )

    selected = payload["selected_candidate_id"]

    if (
        isinstance(selected, bool)
        or not isinstance(selected, int)
        or selected not in eligible_ids
    ):
        raise ValueError(
            "Semantic selector chose an ineligible candidate."
        )

    confidence = payload["confidence"]

    if confidence not in {"low", "medium", "high"}:
        raise ValueError(
            "Semantic selector confidence is invalid."
        )

    rationale = payload["rationale"]

    if (
        not isinstance(rationale, list)
        or not rationale
        or any(
            not isinstance(item, str)
            or not item.strip()
            for item in rationale
        )
    ):
        raise ValueError(
            "Semantic selector rationale must be a non-empty list of strings."
        )

    return SemanticSelectionResult(
        selected_candidate_id=selected,
        confidence=confidence,
        rationale=tuple(
            item.strip()[:1000]
            for item in rationale[:6]
        ),
        tokens_used=max(0, int(tokens_used)),
    )
