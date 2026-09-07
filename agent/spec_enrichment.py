from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any

from agent.specification import TaskSpecification
from agent.compiled_specification import CompiledSpecification


@dataclass
class SpecificationEnrichment:
    deliverables: list[dict[str, Any]]
    required_columns: list[str]
    required_row_rules: list[str]
    ordering_rules: list[str]

    units: dict[str, str]
    conventions: list[str]

    edge_cases: list[str]
    invariants: list[str]

    assumptions: list[str]
    unresolved_questions: list[str]

    difficulty: str | None
    reasoning_summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_safe_model_payload(
    spec: TaskSpecification,
    compiled_spec: CompiledSpecification,
) -> dict[str, Any]:
    """
    Build the model-facing task context.

    IMPORTANT:
    raw_card is deliberately excluded because card metadata may
    contain canaries, contamination markers, or benchmark-only data.
    """

    return {
        "task": {
            "task_id": spec.task_id,
            "category": spec.category,
            "instruction": spec.instruction_text,
        },
        "deterministic_specification": {
            "required_output_paths": spec.required_output_paths,
            "candidate_schema_fields": spec.candidate_schema_fields,
            "required_output_columns": spec.required_output_columns,
            "runtime_seconds": spec.runtime_seconds,
            "network_mode": spec.network_mode,
        },
        "current_compiled_specification": compiled_spec.to_dict(),
    }


def build_specification_prompt(
    spec: TaskSpecification,
    compiled_spec: CompiledSpecification,
) -> str:
    payload = build_safe_model_payload(
        spec=spec,
        compiled_spec=compiled_spec,
    )

    task_json = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    return f"""
You are the specification compiler for a quantitative-finance coding agent.

Your job is NOT to solve the task.

Your job is to convert the task into a precise implementation and
verification contract.

The deterministic specification contains facts already extracted by
code. Treat explicit deterministic facts and explicit task instructions
as authoritative.

You may enrich missing fields, identify ambiguity, and propose finance
invariants, but do not silently contradict explicit requirements.

Focus on:

1. Required deliverables.
2. Exact output paths and formats.
3. Required columns and schemas.
4. Row-count and identifier rules.
5. Ordering requirements.
6. Units and scaling conventions.
7. Financial/model conventions.
8. Explicit and likely edge cases.
9. Mathematical, numerical, data-causality, and accounting invariants.
10. Assumptions required to implement the task.
11. Ambiguities that should remain unresolved rather than guessed.
12. Difficulty: easy, medium, or hard.

Important rules:

- Do not solve the quantitative-finance problem.
- Do not generate implementation code.
- Do not invent requirements that are unsupported.
- Distinguish explicit requirements from reasonable verification checks.
- Preserve units exactly where specified.
- Be especially careful about dates, timestamps, rates, percentages,
  annualisation, compounding, day-count conventions, and signs.
- For time-series/backtesting tasks, check information timing and leakage.
- For pricing tasks, identify relevant arbitrage and limiting-case checks.
- For simulation tasks, identify reproducibility and convergence checks.
- For trading tasks, identify position, cash, P&L, and NAV reconciliation.
- If something cannot be determined from the task, put it in
  unresolved_questions instead of guessing.

Return ONLY valid JSON with exactly these top-level keys:

{{
  "deliverables": [],
  "required_columns": [],
  "required_row_rules": [],
  "ordering_rules": [],
  "units": {{}},
  "conventions": [],
  "edge_cases": [],
  "invariants": [],
  "assumptions": [],
  "unresolved_questions": [],
  "difficulty": "easy|medium|hard",
  "reasoning_summary": ""
}}

TASK CONTEXT:

{task_json}
""".strip()