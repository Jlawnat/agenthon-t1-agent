from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.compiled_specification import (
    CompiledSpecification,
)
from agent.spec_enrichment import (
    SpecificationEnrichment,
)
from agent.specification import (
    TaskSpecification,
)


def _unique_strings(
    *groups: list[str],
) -> list[str]:
    result: list[str] = []

    for group in groups:
        for value in group:
            if value not in result:
                result.append(value)

    return result


def _merge_deliverables(
    deterministic_paths: list[str],
    existing: list[dict[str, Any]],
    enriched: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}

    for item in existing:
        path = item.get("path")

        if isinstance(path, str):
            by_path[path] = deepcopy(item)

    for item in enriched:
        if not isinstance(item, dict):
            continue

        path = item.get("path")

        if not isinstance(path, str):
            continue

        if path in by_path:
            current = by_path[path]

            for key, value in item.items():
                if (
                    key != "path"
                    and key not in current
                ):
                    current[key] = value

    # Explicit deterministic output paths are authoritative.
    result: list[dict[str, Any]] = []

    for path in deterministic_paths:
        if path in by_path:
            result.append(by_path[path])
        else:
            result.append(
                {
                    "path": path,
                    "format": (
                        path.rsplit(".", 1)[-1]
                        if "." in path
                        else None
                    ),
                }
            )

    return result


def merge_specification(
    deterministic_spec: TaskSpecification,
    compiled_spec: CompiledSpecification,
    enrichment: SpecificationEnrichment,
) -> CompiledSpecification:
    """
    Merge model enrichment into the compiled spec.

    Explicit deterministic facts remain authoritative.
    The model may enrich but not silently replace them.
    """

    return CompiledSpecification(
        task_id=compiled_spec.task_id,
        category=compiled_spec.category,

        difficulty=(
            enrichment.difficulty
            or compiled_spec.difficulty
        ),

        deliverables=_merge_deliverables(
            deterministic_paths=(
                deterministic_spec
                .required_output_paths
            ),
            existing=compiled_spec.deliverables,
            enriched=enrichment.deliverables,
        ),

        required_columns=_unique_strings(
            compiled_spec.required_columns,
            enrichment.required_columns,
        ),

        required_row_rules=_unique_strings(
            compiled_spec.required_row_rules,
            enrichment.required_row_rules,
        ),

        ordering_rules=_unique_strings(
            compiled_spec.ordering_rules,
            enrichment.ordering_rules,
        ),

        units={
            **compiled_spec.units,
            **enrichment.units,
        },

        conventions=_unique_strings(
            compiled_spec.conventions,
            enrichment.conventions,
        ),

        edge_cases=_unique_strings(
            compiled_spec.edge_cases,
            enrichment.edge_cases,
        ),

        invariants=_unique_strings(
            compiled_spec.invariants,
            enrichment.invariants,
        ),

        review_packs=list(
            compiled_spec.review_packs
        ),

        numerical_risks=list(
            compiled_spec.numerical_risks
        ),

        assumptions=_unique_strings(
            compiled_spec.assumptions,
            enrichment.assumptions,
        ),

        unresolved_questions=_unique_strings(
            compiled_spec.unresolved_questions,
            enrichment.unresolved_questions,
        ),
    )