from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.specification import load_task_specification
from agent.spec_compiler import compile_specification
from agent.planning import build_task_plan


_DOMAIN_PACKS_FOR_CONVERGENCE = {
    "derivatives",
    "fixed-income",
    "credit",
    "risk-management",
}


@dataclass
class PlannerPolicyRow:
    unit_dir: str
    task_id: str | None
    category: str | None
    difficulty: str | None
    review_packs: list[str]

    candidate_count: int | None

    actual_independent_methods: bool | None
    expected_independent_methods: bool | None

    actual_convergence_checks: bool | None
    expected_convergence_checks: bool | None

    actual_perturbation_checks: bool | None
    expected_perturbation_checks: bool | None

    actual_causality_checks: bool | None
    expected_causality_checks: bool | None

    actual_accounting_checks: bool | None
    expected_accounting_checks: bool | None

    mismatch_types: list[str]
    load_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_units_root(repo: Path) -> Path:
    for candidate in (repo / "units", repo):
        if not candidate.exists() or not candidate.is_dir():
            continue

        task_dirs = [
            path
            for path in candidate.iterdir()
            if path.is_dir()
            and (path / "instruction.md").exists()
        ]

        if task_dirs:
            return candidate

    raise FileNotFoundError(
        f"Could not locate public task units under {repo}."
    )


def _expected_flags(
    *,
    difficulty: str,
    review_packs: list[str],
    numerical_risks: list[str],
) -> dict[str, bool]:
    """
    Routing-consistency policy.

    This intentionally mirrors the CURRENT planning policy, but derives
    domain semantics from the already-routed review packs instead of raw
    category strings. It is therefore an audit for category-alias drift,
    not a redesign of planner policy.
    """

    packs = set(review_packs)

    independent = (
        difficulty == "hard"
        or "cross-domain" in packs
        or (
            "derivatives" in packs
            and difficulty == "medium"
        )
    )
    has_convergence_risk = any(
        "convergence" in risk.lower()
        for risk in numerical_risks
        )
    convergence = (
        difficulty in {"medium", "hard"}
        and (
            bool(
                packs.intersection(
                    _DOMAIN_PACKS_FOR_CONVERGENCE
                )
            )
            or has_convergence_risk
        )
   )



    perturbation = "numerical" in packs
    causality = "data-causality" in packs
    accounting = "accounting" in packs

    return {
        "independent_methods": independent,
        "convergence_checks": convergence,
        "perturbation_checks": perturbation,
        "causality_checks": causality,
        "accounting_checks": accounting,
    }


def audit_unit(task_dir: Path) -> PlannerPolicyRow:
    unit_dir = task_dir.name

    try:
        spec = load_task_specification(task_dir)
        compiled = compile_specification(spec)
        plan = build_task_plan(compiled)

        packs = list(plan.review_packs)

        expected = _expected_flags(
            difficulty=compiled.difficulty,
            review_packs=packs,
            numerical_risks=list(
                compiled.numerical_risks
            )
        )

        actual = {
            "independent_methods": (
                plan.require_independent_methods
            ),
            "convergence_checks": (
                plan.require_convergence_checks
            ),
            "perturbation_checks": (
                plan.require_perturbation_checks
            ),
            "causality_checks": (
                plan.require_causality_checks
            ),
            "accounting_checks": (
                plan.require_accounting_checks
            ),
        }

        mismatch_types = [
            name
            for name, expected_value in expected.items()
            if actual[name] != expected_value
        ]

        return PlannerPolicyRow(
            unit_dir=unit_dir,
            task_id=compiled.task_id,
            category=compiled.category,
            difficulty=compiled.difficulty,
            review_packs=packs,
            candidate_count=plan.candidate_count,

            actual_independent_methods=actual[
                "independent_methods"
            ],
            expected_independent_methods=expected[
                "independent_methods"
            ],

            actual_convergence_checks=actual[
                "convergence_checks"
            ],
            expected_convergence_checks=expected[
                "convergence_checks"
            ],

            actual_perturbation_checks=actual[
                "perturbation_checks"
            ],
            expected_perturbation_checks=expected[
                "perturbation_checks"
            ],

            actual_causality_checks=actual[
                "causality_checks"
            ],
            expected_causality_checks=expected[
                "causality_checks"
            ],

            actual_accounting_checks=actual[
                "accounting_checks"
            ],
            expected_accounting_checks=expected[
                "accounting_checks"
            ],

            mismatch_types=mismatch_types,
            load_error=None,
        )

    except Exception as exc:
        return PlannerPolicyRow(
            unit_dir=unit_dir,
            task_id=None,
            category=None,
            difficulty=None,
            review_packs=[],
            candidate_count=None,

            actual_independent_methods=None,
            expected_independent_methods=None,

            actual_convergence_checks=None,
            expected_convergence_checks=None,

            actual_perturbation_checks=None,
            expected_perturbation_checks=None,

            actual_causality_checks=None,
            expected_causality_checks=None,

            actual_accounting_checks=None,
            expected_accounting_checks=None,

            mismatch_types=[],
            load_error=f"{type(exc).__name__}: {exc}",
        )


def _write_csv(
    path: Path,
    rows: list[PlannerPolicyRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "unit_dir",
        "task_id",
        "category",
        "difficulty",
        "review_packs",
        "candidate_count",
        "actual_independent_methods",
        "expected_independent_methods",
        "actual_convergence_checks",
        "expected_convergence_checks",
        "actual_perturbation_checks",
        "expected_perturbation_checks",
        "actual_causality_checks",
        "expected_causality_checks",
        "actual_accounting_checks",
        "expected_accounting_checks",
        "mismatch_types",
        "load_error",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            payload = row.to_dict()
            payload["review_packs"] = json.dumps(
                payload["review_packs"],
                ensure_ascii=False,
            )
            payload["mismatch_types"] = json.dumps(
                payload["mismatch_types"],
                ensure_ascii=False,
            )
            writer.writerow(payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of deterministic planner flags "
            "against routed skill-pack semantics."
        )
    )

    parser.add_argument(
        "--repo",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--expected-unit-count",
        type=int,
        default=87,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path("benchmark")
            / "planner_policy_audit"
        ),
    )

    args = parser.parse_args()

    units_root = _find_units_root(
        args.repo.resolve()
    )

    task_dirs = sorted(
        [
            path
            for path in units_root.iterdir()
            if path.is_dir()
            and (path / "instruction.md").exists()
        ],
        key=lambda path: path.name,
    )

    rows = [
        audit_unit(task_dir)
        for task_dir in task_dirs
    ]

    load_errors = [
        row
        for row in rows
        if row.load_error is not None
    ]

    mismatches = [
        row
        for row in rows
        if row.mismatch_types
    ]

    mismatch_counts: dict[str, int] = {}

    for row in mismatches:
        for name in row.mismatch_types:
            mismatch_counts[name] = (
                mismatch_counts.get(name, 0)
                + 1
            )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_path = (
        out_dir
        / "planner_policy_audit_rows.csv"
    )
    report_path = (
        out_dir
        / "planner_policy_audit_report.json"
    )

    _write_csv(
        rows_path,
        rows,
    )

    report = {
        "units_audited": len(rows),
        "expected_units": (
            args.expected_unit_count
        ),
        "load_errors": len(load_errors),
        "units_with_policy_mismatch": (
            len(mismatches)
        ),
        "mismatch_counts": mismatch_counts,
        "mismatch_units": [
            {
                "unit_dir": row.unit_dir,
                "category": row.category,
                "difficulty": row.difficulty,
                "review_packs": row.review_packs,
                "mismatch_types": row.mismatch_types,
            }
            for row in mismatches
        ],
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "AGENTHON T1 — PLANNER POLICY AUDIT"
    )
    print("=" * 72)
    print(
        f"Units audited: {len(rows)}"
    )
    print(
        "Expected units:",
        args.expected_unit_count,
    )
    print(
        f"Load errors: {len(load_errors)}"
    )
    print(
        "Units with planner-policy mismatch:",
        len(mismatches),
    )

    print("\nMismatch counts:")

    if mismatch_counts:
        for name, count in sorted(
            mismatch_counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):
            print(
                f"  {name}: {count}"
            )
    else:
        print("  none")

    if mismatches:
        print("\nMismatch units:")

        for row in mismatches:
            print(
                f"\n  {row.unit_dir}"
            )
            print(
                f"    category: {row.category}"
            )
            print(
                f"    difficulty: {row.difficulty}"
            )
            print(
                f"    packs: {row.review_packs}"
            )
            print(
                "    mismatches:",
                row.mismatch_types,
            )

    print("\nSaved:")
    print(f"  {rows_path}")
    print(f"  {report_path}")

    if len(rows) != args.expected_unit_count:
        print(
            "\nWARNING: unit count does not "
            "match expected corpus size."
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
