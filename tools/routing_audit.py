from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.specification import load_task_specification
from agent.spec_compiler import compile_specification
from agent.planning import build_task_plan
from agent.skill_packs import load_skill_packs


@dataclass
class RoutingRow:
    unit_dir: str
    task_id: str | None
    category: str | None
    candidate_count: int | None
    review_packs: list[str]
    loaded_skill_packs: list[str]
    empty_review_packs: bool
    load_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    try:
        values = list(value)
    except TypeError:
        return [str(value)]

    result: list[str] = []

    for item in values:
        text = str(item)
        if text not in result:
            result.append(text)

    return result


def _skill_pack_name(pack: Any) -> str:
    for attr in (
        "name",
        "pack_name",
        "id",
        "key",
    ):
        value = getattr(
            pack,
            attr,
            None,
        )

        if value:
            return str(value)

    to_dict = getattr(
        pack,
        "to_dict",
        None,
    )

    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            payload = {}

        if isinstance(
            payload,
            dict,
        ):
            for key in (
                "name",
                "pack_name",
                "id",
                "key",
            ):
                value = payload.get(
                    key
                )

                if value:
                    return str(value)

    return type(pack).__name__


def _find_units_root(
    repo: Path,
) -> Path:
    candidates = (
        repo / "units",
        repo,
    )

    for candidate in candidates:
        if (
            candidate.exists()
            and candidate.is_dir()
        ):
            task_dirs = [
                path
                for path in candidate.iterdir()
                if (
                    path.is_dir()
                    and (
                        path / "instruction.md"
                    ).exists()
                )
            ]

            if task_dirs:
                return candidate

    raise FileNotFoundError(
        "Could not locate public task units. "
        f"Checked {repo / 'units'} and {repo}."
    )


def audit_unit(
    task_dir: Path,
) -> RoutingRow:
    unit_dir = task_dir.name

    try:
        spec = (
            load_task_specification(
                task_dir
            )
        )

        compiled = (
            compile_specification(
                spec
            )
        )

        plan = (
            build_task_plan(
                compiled
            )
        )

        review_packs = (
            _normalise_list(
                getattr(
                    plan,
                    "review_packs",
                    [],
                )
            )
        )

        skill_packs = (
            load_skill_packs(
                review_packs
            )
        )

        loaded_names = [
            _skill_pack_name(pack)
            for pack in skill_packs
        ]

        return RoutingRow(
            unit_dir=unit_dir,
            task_id=getattr(
                spec,
                "task_id",
                None,
            ),
            category=getattr(
                spec,
                "category",
                None,
            ),
            candidate_count=getattr(
                plan,
                "candidate_count",
                None,
            ),
            review_packs=review_packs,
            loaded_skill_packs=(
                loaded_names
            ),
            empty_review_packs=(
                len(review_packs) == 0
            ),
            load_error=None,
        )

    except Exception as exc:
        return RoutingRow(
            unit_dir=unit_dir,
            task_id=None,
            category=None,
            candidate_count=None,
            review_packs=[],
            loaded_skill_packs=[],
            empty_review_packs=True,
            load_error=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )


def _write_csv(
    path: Path,
    rows: list[RoutingRow],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "unit_dir",
        "task_id",
        "category",
        "candidate_count",
        "review_packs",
        "loaded_skill_packs",
        "empty_review_packs",
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
            payload[
                "review_packs"
            ] = json.dumps(
                payload[
                    "review_packs"
                ],
                ensure_ascii=False,
            )
            payload[
                "loaded_skill_packs"
            ] = json.dumps(
                payload[
                    "loaded_skill_packs"
                ],
                ensure_ascii=False,
            )

            writer.writerow(
                payload
            )


def build_report(
    rows: list[RoutingRow],
) -> dict[str, Any]:
    category_counts: Counter[
        str
    ] = Counter()

    candidate_counts: Counter[
        str
    ] = Counter()

    review_pack_counts: Counter[
        str
    ] = Counter()

    loaded_pack_counts: Counter[
        str
    ] = Counter()

    category_routes: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    load_errors = []
    empty_routes = []

    for row in rows:
        category = (
            row.category
            or "<none>"
        )

        category_counts[
            category
        ] += 1

        candidate_counts[
            str(
                row.candidate_count
            )
        ] += 1

        route_key = (
            json.dumps(
                row.review_packs,
                ensure_ascii=False,
            )
        )

        category_routes[
            category
        ][route_key] += 1

        for pack in row.review_packs:
            review_pack_counts[
                pack
            ] += 1

        for pack in (
            row.loaded_skill_packs
        ):
            loaded_pack_counts[
                pack
            ] += 1

        if row.empty_review_packs:
            empty_routes.append(
                row.unit_dir
            )

        if row.load_error:
            load_errors.append(
                {
                    "unit_dir": (
                        row.unit_dir
                    ),
                    "error": (
                        row.load_error
                    ),
                }
            )

    route_summary = {}

    for category, counts in sorted(
        category_routes.items()
    ):
        route_summary[
            category
        ] = [
            {
                "review_packs": (
                    json.loads(
                        route_key
                    )
                ),
                "count": count,
            }
            for (
                route_key,
                count,
            )
            in counts.most_common()
        ]

    return {
        "units_audited": len(rows),
        "load_error_count": (
            len(load_errors)
        ),
        "empty_review_pack_count": (
            len(empty_routes)
        ),
        "category_counts": dict(
            category_counts.most_common()
        ),
        "candidate_count_distribution": dict(
            candidate_counts.most_common()
        ),
        "review_pack_counts": dict(
            review_pack_counts.most_common()
        ),
        "loaded_skill_pack_counts": dict(
            loaded_pack_counts.most_common()
        ),
        "routes_by_category": (
            route_summary
        ),
        "empty_review_pack_units": (
            empty_routes
        ),
        "load_errors": (
            load_errors
        ),
    }


def print_summary(
    report: dict[str, Any],
) -> None:
    print(
        "AGENTHON T1 — ROUTING AUDIT"
    )
    print("=" * 72)

    print(
        "Units audited:",
        report[
            "units_audited"
        ],
    )
    print(
        "Load errors:",
        report[
            "load_error_count"
        ],
    )
    print(
        "Units with empty review packs:",
        report[
            "empty_review_pack_count"
        ],
    )

    print(
        "\nCandidate-count distribution:"
    )
    for key, count in (
        report[
            "candidate_count_distribution"
        ].items()
    ):
        print(
            f"  {key}: {count}"
        )

    print(
        "\nReview-pack usage:"
    )
    if (
        report[
            "review_pack_counts"
        ]
    ):
        for pack, count in (
            report[
                "review_pack_counts"
            ].items()
        ):
            print(
                f"  {pack}: {count}"
            )
    else:
        print(
            "  <none>"
        )

    print(
        "\nRoutes by category:"
    )

    for category, routes in (
        report[
            "routes_by_category"
        ].items()
    ):
        print(
            f"  {category}:"
        )

        for route in routes:
            print(
                "    "
                f"{route['count']} × "
                f"{route['review_packs']}"
            )

    if (
        report[
            "empty_review_pack_units"
        ]
    ):
        print(
            "\nUnits with empty review packs:"
        )

        for unit in (
            report[
                "empty_review_pack_units"
            ]
        ):
            print(
                f"  {unit}"
            )

    if report["load_errors"]:
        print(
            "\nLoad errors:"
        )

        for item in (
            report[
                "load_errors"
            ]
        ):
            print(
                "  "
                f"{item['unit_dir']}: "
                f"{item['error']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit current Agenthon T1 "
            "task-family routing over the "
            "public task corpus."
        )
    )

    parser.add_argument(
        "--repo",
        type=Path,
        default=(
            Path.home()
            / "track1-coding-public"
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path("benchmark")
            / "routing_audit"
        ),
    )

    parser.add_argument(
        "--expected-unit-count",
        type=int,
        default=87,
    )

    args = parser.parse_args()

    units_root = (
        _find_units_root(
            args.repo.expanduser()
        )
    )

    task_dirs = sorted(
        (
            path
            for path
            in units_root.iterdir()
            if (
                path.is_dir()
                and (
                    path
                    / "instruction.md"
                ).exists()
            )
        ),
        key=lambda path: path.name,
    )

    if (
        len(task_dirs)
        != args.expected_unit_count
    ):
        raise RuntimeError(
            "Unexpected public-unit count: "
            f"found {len(task_dirs)}, "
            "expected "
            f"{args.expected_unit_count}."
        )

    rows = [
        audit_unit(task_dir)
        for task_dir in task_dirs
    ]

    report = build_report(
        rows
    )

    out_dir = (
        args.out_dir
        .expanduser()
        .resolve()
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_path = (
        out_dir
        / "routing_audit_rows.csv"
    )
    report_path = (
        out_dir
        / "routing_audit_report.json"
    )

    _write_csv(
        rows_path,
        rows,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print_summary(
        report
    )

    print(
        "\nSaved:"
    )
    print(
        f"  {rows_path}"
    )
    print(
        f"  {report_path}"
    )


if __name__ == "__main__":
    main()