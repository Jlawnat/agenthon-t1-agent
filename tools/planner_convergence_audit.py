from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.specification import load_task_specification
from agent.spec_compiler import compile_specification
from agent.planning import build_task_plan


_SIGNAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "monte_carlo_simulation": (
        r"\bmonte[\s-]+carlo\s+"
        r"(?:simulation|simulations|validation|verification)\b",
        r"\bn_simulations?\b",
        r"\bn_simulation_paths\b",
        r"\bmc_num_paths\b",
        r"\bsimulate\b[^\n]{0,100}\bpaths?\b",
        r"\bsimulated?\s+"
        r"(?:paths?|trials?|residuals?|returns?|losses?|samples?)\b",
        r"\bgbm\s+paths?\b",
    ),
    "stochastic_bootstrap": (
        r"\bstationary\s+bootstrap\b",
        r"\bblock\s+bootstrap\b",
        r"\bbootstrap\s+"
        r"(?:simulation|resampling|samples?)\b",
    ),
    "numerical_optimisation": (
        r"\bscipy\.optimize\b",
        r"\boptimize\.minimize\b",
        r"\bminimize\b",
        r"\bleast[\s-]+squares\s+optimi[sz]ation\b",
        r"\bconstrained\s+optimi[sz]ation\b",
        r"\bl-bfgs-b\b",
        r"\bslsqp\b",
    ),
    "root_finding": (
        r"\bbrentq\b",
        r"\bbrent(?:'s)?\s+method\b",
        r"\bnewton(?:-raphson|'s\s+method)?\b",
        r"\broot[\s-]+find",
        r"\broot[\s-]+solv",
    ),
    "iterative_convergence": (
        r"\bbaum[\s-]+welch\b",
        r"\bem\s+algorithm\b",
        r"\bmax[_\s-]*iter(?:ations?)?\b",
        r"\bterminate\b[^\n]{0,100}"
        r"\b(?:tol|tolerance|converg)",
    ),
    "tree_or_grid": (
        r"\btrinomial\s+tree\b",
        r"\bbinomial\s+tree\b",
        r"\bpde\b",
        r"\bgrid\s+convergence\b",
        r"\btimestep\s+convergence\b",
    ),
    "numerical_integration": (
        r"\bquadrature\b",
        r"\bnumerical\s+integration\b",
    ),
}

@dataclass
class Row:
    unit_dir: str
    task_id: str | None
    category: str | None
    difficulty: str | None
    review_packs: list[str]
    actual_convergence: bool | None
    method_signals: list[str]
    signal_expected_convergence: bool | None
    classification: str
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


def _method_signals(text: str) -> list[str]:
    lowered = text.lower()

    hits: list[str] = []

    for name, patterns in _SIGNAL_PATTERNS.items():
        if any(
            re.search(pattern, lowered)
            for pattern in patterns
        ):
            hits.append(name)

    return hits


def audit_unit(task_dir: Path) -> Row:
    try:
        spec = load_task_specification(task_dir)
        compiled = compile_specification(spec)
        plan = build_task_plan(compiled)

        signals = _method_signals(
            spec.instruction_text
        )

        expected = (
            compiled.difficulty in {"medium", "hard"}
            and bool(signals)
        )

        actual = (
            plan.require_convergence_checks
        )

        if actual and expected:
            classification = "aligned_required"
        elif (not actual) and (not expected):
            classification = "aligned_not_required"
        elif (not actual) and expected:
            classification = "possible_false_negative"
        else:
            classification = "possible_false_positive"

        return Row(
            unit_dir=task_dir.name,
            task_id=compiled.task_id,
            category=compiled.category,
            difficulty=compiled.difficulty,
            review_packs=list(
                plan.review_packs
            ),
            actual_convergence=actual,
            method_signals=signals,
            signal_expected_convergence=expected,
            classification=classification,
            load_error=None,
        )

    except Exception as exc:
        return Row(
            unit_dir=task_dir.name,
            task_id=None,
            category=None,
            difficulty=None,
            review_packs=[],
            actual_convergence=None,
            method_signals=[],
            signal_expected_convergence=None,
            classification="load_error",
            load_error=f"{type(exc).__name__}: {exc}",
        )


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = list(
        Row.__dataclass_fields__
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            payload = row.to_dict()
            payload["review_packs"] = json.dumps(
                payload["review_packs"],
                ensure_ascii=False,
            )
            payload["method_signals"] = json.dumps(
                payload["method_signals"],
                ensure_ascii=False,
            )
            writer.writerow(payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of convergence-planning flags "
            "against explicit numerical-method signals."
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
            / "planner_convergence_audit"
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
        audit_unit(path)
        for path in task_dirs
    ]

    counts: dict[str, int] = {}

    for row in rows:
        counts[row.classification] = (
            counts.get(row.classification, 0)
            + 1
        )

    review_rows = [
        row
        for row in rows
        if row.classification
        in {
            "possible_false_negative",
            "possible_false_positive",
        }
    ]

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_path = (
        out_dir
        / "planner_convergence_audit_rows.csv"
    )
    report_path = (
        out_dir
        / "planner_convergence_audit_report.json"
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
        "classification_counts": counts,
        "review_units": [
            row.to_dict()
            for row in review_rows
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
        "AGENTHON T1 — PLANNER CONVERGENCE AUDIT"
    )
    print("=" * 72)
    print(
        f"Units audited: {len(rows)}"
    )
    print(
        "Expected units:",
        args.expected_unit_count,
    )

    print("\nClassification counts:")

    for name, count in sorted(
        counts.items()
    ):
        print(
            f"  {name}: {count}"
        )

    print("\nReview units:")

    if not review_rows:
        print("  none")
    else:
        for row in review_rows:
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
                f"    actual convergence: "
                f"{row.actual_convergence}"
            )
            print(
                f"    method signals: "
                f"{row.method_signals}"
            )
            print(
                f"    classification: "
                f"{row.classification}"
            )

    print("\nSaved:")
    print(f"  {rows_path}")
    print(f"  {report_path}")

    print(
        "\nNOTE: Signal classifications are review flags, "
        "not hard failures."
    )

    if len(rows) != args.expected_unit_count:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())