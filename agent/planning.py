from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from agent.compiled_specification import CompiledSpecification


@dataclass
class TaskPlan:
    difficulty: str
    candidate_count: int

    review_packs: list[str]

    require_independent_methods: bool
    require_convergence_checks: bool
    require_perturbation_checks: bool
    require_causality_checks: bool
    require_accounting_checks: bool

    strategy_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_count(difficulty: str) -> int:
    if difficulty == "hard":
        return 3

    if difficulty == "medium":
        return 2

    return 1


def build_task_plan(
    spec: CompiledSpecification,
) -> TaskPlan:
    packs = set(spec.review_packs)

    require_causality = (
        "data-causality" in packs
        or spec.category
        in {
            "factor-research",
            "backtesting",
            "microstructure",
            "nlp-on-finance",
        }
    )

    require_accounting = (
        "accounting" in packs
        or spec.category
        in {
            "backtesting",
            "microstructure",
        }
    )
    has_convergence_risk = any(
        "convergence" in risk.lower()
        for risk in spec.numerical_risks
    )
    require_convergence = (
        spec.difficulty in {"medium", "hard"}
        and(
            bool(
                packs.intersection(
                    {
                        "derivatives",
                        "fixed-income",
                        "credit",
                        "risk-management",
                    }
                )
            )
            or has_convergence_risk
        )
    )


    require_perturbation = (
        "numerical" in packs
        or spec.category
        in {
            "derivatives-pricing",
            "fixed-income",
            "credit",
            "risk-management",
        }
    )
    require_independent_methods = (
        spec.difficulty == "hard"
        or "cross-domain" in packs
        or (
            "derivatives" in packs
            and spec.difficulty == "medium"
        )
    )


    notes: list[str] = []

    if require_independent_methods:
        notes.append(
            "Generate method-diverse candidate solutions "
            "where practical."
        )

    if require_convergence:
        notes.append(
            "Check numerical convergence separately "
            "from sampling error."
        )

    if require_perturbation:
        notes.append(
            "Use bump or perturbation checks for "
            "sensitive outputs."
        )

    if require_causality:
        notes.append(
            "Enforce information_time <= signal_time "
            "< execution_time."
        )

    if require_accounting:
        notes.append(
            "Reconcile positions, cash, P&L, and NAV."
        )

    return TaskPlan(
        difficulty=spec.difficulty,
        candidate_count=_candidate_count(
            spec.difficulty
        ),
        review_packs=list(
            spec.review_packs
        ),
        require_independent_methods=(
            require_independent_methods
        ),
        require_convergence_checks=(
            require_convergence
        ),
        require_perturbation_checks=(
            require_perturbation
        ),
        require_causality_checks=(
            require_causality
        ),
        require_accounting_checks=(
            require_accounting
        ),
        strategy_notes=notes,
    )