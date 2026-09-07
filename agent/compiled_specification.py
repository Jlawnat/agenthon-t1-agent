from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class CompiledSpecification:
    task_id: str | None
    category: str | None
    difficulty: str

    deliverables: list[dict[str, Any]]
    required_columns: list[str]
    required_row_rules: list[str]
    ordering_rules: list[str]

    units: dict[str, str]
    conventions: list[str]

    edge_cases: list[str]
    invariants: list[str]

    review_packs: list[str]
    numerical_risks: list[str]

    assumptions: list[str]
    unresolved_questions: list[str]
    required_dtypes: dict[str, str] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)