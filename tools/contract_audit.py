from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.specification import load_task_specification


DEFAULT_PUBLIC_REPO = Path.home() / "track1-coding-public"
DEFAULT_INVENTORY = PROJECT_ROOT / "benchmark" / "public_units_inventory.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "benchmark" / "contract_audit"

STRUCTURED_EXTENSIONS = {
    "csv",
    "json",
    "jsonl",
    "parquet",
    "xlsx",
    "xls",
    "feather",
}
VERIFIER_OWNED_BASENAMES = {
    "reward.json",
    "reward.txt",
    "pytest_report.json",
    "test_outputs.py",
    "test.sh",
}
MARKDOWN_SCHEMA_TABLE = re.compile(
    r"(?im)^\s*\|[^|\n]*column[^|\n]*\|[^|\n]*type[^|\n]*\|"
)
JSON_FENCE = re.compile(
    r"```(?:json|JSON)\s*(.*?)```",
    flags=re.DOTALL,
)
JSON_KEY = re.compile(
    r'"([A-Za-z_][A-Za-z0-9_.-]*)"\s*:'
)
NESTED_JSON_OBJECT = re.compile(
    r'"[A-Za-z_][A-Za-z0-9_.-]*"\s*:\s*\{'
)
KEY_LIST_SIGNAL = re.compile(
    r"(?i)\b(?:keys?|fields?|columns?|schema)\b"
)
OUTPUT_PATH_SIGNAL = re.compile(
    r"(?:/app/output|/output|\.?/output)/[A-Za-z0-9_./\\-]+"
)


@dataclass(frozen=True)
class AuditRow:
    unit_dir: str
    task_id_expected: str | None
    task_id_actual: str | None
    category_expected: str | None
    category_actual: str | None
    timeout_expected: float | int | None
    timeout_actual: float | int | None
    network_expected: str | None
    network_actual: str | None

    expected_output_count: int
    actual_output_count: int
    expected_output_paths: list[str]
    actual_output_paths: list[str]
    missing_output_paths: list[str]
    extra_output_paths: list[str]

    structured_output_count: int
    required_output_columns: list[str]
    required_output_dtypes: dict[str, str]
    candidate_schema_fields: list[str]

    markdown_schema_table_signal: bool
    json_schema_signal: bool
    nested_json_signal: bool
    general_schema_signal: bool
    schema_review_needed: bool

    hard_failure_types: list[str]
    hard_contract_pass: bool
    load_error: str | None


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object at {path}"
        )

    return payload


def canonical_output_path(value: str) -> str:
    """
    Compare deliverables by logical location, not by the two
    equivalent Track-1 output-root spellings.

    Examples:
        /app/output/results.csv -> /output/results.csv
        /output/results.csv     -> /output/results.csv
        ./output/results.csv    -> /output/results.csv
    """
    path = str(value).strip().replace("\\", "/")

    while "//" in path:
        path = path.replace("//", "/")

    for prefix in (
        "/app/output/",
        "/output/",
        "./output/",
        "output/",
    ):
        if path.startswith(prefix):
            relative = path[len(prefix):].lstrip("/")
            return f"/output/{relative}"

    if path in {
        "/app/output",
        "/output",
        "./output",
        "output",
    }:
        return "/output"

    return path


def canonical_paths(
    values: list[str],
) -> list[str]:
    result: list[str] = []

    for value in values:
        canonical = (
            canonical_output_path(
                value
            )
        )

        basename = (
            Path(
                canonical.replace(
                    "\\",
                    "/",
                )
            )
            .name
            .lower()
        )

        if (
            basename
            in VERIFIER_OWNED_BASENAMES
        ):
            continue

        if canonical not in result:
            result.append(
                canonical
            )

    return result

def path_extension(value: str) -> str:
    return (
        Path(value)
        .suffix
        .lower()
        .lstrip(".")
    )


def numeric_equal(
    first: float | int | None,
    second: float | int | None,
) -> bool:
    if first is None or second is None:
        return first is second

    try:
        return float(first) == float(second)
    except (TypeError, ValueError):
        return first == second


def get_json_schema_signals(
    instruction: str,
) -> tuple[bool, bool, list[str]]:
    json_blocks = JSON_FENCE.findall(
        instruction
    )

    keys: list[str] = []
    nested = False

    for block in json_blocks:
        if NESTED_JSON_OBJECT.search(
            block
        ):
            nested = True

        for key in JSON_KEY.findall(
            block
        ):
            if key not in keys:
                keys.append(
                    key
                )

    return (
        bool(json_blocks and keys),
        nested,
        keys,
    )


def instruction_schema_signals(
    instruction: str,
    expected_paths: list[str],
) -> dict[str, Any]:
    structured_paths = [
        path
        for path in expected_paths
        if path_extension(path)
        in STRUCTURED_EXTENSIONS
    ]

    markdown_table = bool(
        MARKDOWN_SCHEMA_TABLE.search(
            instruction
        )
    )

    (
        json_signal,
        nested_json,
        json_keys,
    ) = get_json_schema_signals(
        instruction
    )

    general_signal = bool(
        structured_paths
        and KEY_LIST_SIGNAL.search(
            instruction
        )
    )

    return {
        "structured_paths": (
            structured_paths
        ),
        "markdown_table": (
            markdown_table
        ),
        "json_signal": (
            json_signal
        ),
        "nested_json": (
            nested_json
        ),
        "json_keys": (
            json_keys
        ),
        "general_signal": (
            general_signal
        ),
    }


def inventory_units(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    units = payload.get(
        "units"
    )

    if not isinstance(
        units,
        list,
    ):
        raise ValueError(
            "Inventory JSON must contain a 'units' list."
        )

    result: list[
        dict[str, Any]
    ] = []

    for row in units:
        if not isinstance(
            row,
            dict,
        ):
            raise ValueError(
                "Every inventory unit must be a JSON object."
            )
        result.append(
            row
        )

    return result


def audit_unit(
    public_repo: Path,
    inventory_row: dict[str, Any],
) -> AuditRow:
    unit_dir_name = str(
        inventory_row.get(
            "unit_dir"
        )
        or ""
    )

    task_dir = (
        public_repo
        / "units"
        / unit_dir_name
    )

    expected_paths = canonical_paths(
        [
            str(path)
            for path in (
                inventory_row.get(
                    "output_paths"
                )
                or []
            )
        ]
    )

    instruction_path = (
        task_dir
        / "instruction.md"
    )

    instruction = ""

    if instruction_path.is_file():
        instruction = (
            instruction_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )

    signals = (
        instruction_schema_signals(
            instruction,
            expected_paths,
        )
    )

    try:
        spec = (
            load_task_specification(
                task_dir
            )
        )
    except Exception as exc:
        return AuditRow(
            unit_dir=unit_dir_name,
            task_id_expected=(
                inventory_row.get(
                    "task_id"
                )
            ),
            task_id_actual=None,
            category_expected=(
                inventory_row.get(
                    "category"
                )
            ),
            category_actual=None,
            timeout_expected=(
                inventory_row.get(
                    "agent_timeout_sec"
                )
            ),
            timeout_actual=None,
            network_expected=(
                inventory_row.get(
                    "network"
                )
            ),
            network_actual=None,
            expected_output_count=(
                len(
                    expected_paths
                )
            ),
            actual_output_count=0,
            expected_output_paths=(
                expected_paths
            ),
            actual_output_paths=[],
            missing_output_paths=(
                expected_paths
            ),
            extra_output_paths=[],
            structured_output_count=(
                len(
                    signals[
                        "structured_paths"
                    ]
                )
            ),
            required_output_columns=[],
            required_output_dtypes={},
            candidate_schema_fields=[],
            markdown_schema_table_signal=(
                signals[
                    "markdown_table"
                ]
            ),
            json_schema_signal=(
                signals[
                    "json_signal"
                ]
            ),
            nested_json_signal=(
                signals[
                    "nested_json"
                ]
            ),
            general_schema_signal=(
                signals[
                    "general_signal"
                ]
            ),
            schema_review_needed=(
                bool(
                    signals[
                        "structured_paths"
                    ]
                )
            ),
            hard_failure_types=[
                "load_error"
            ],
            hard_contract_pass=False,
            load_error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )

    actual_paths = canonical_paths(
        [
            str(path)
            for path in (
                getattr(
                    spec,
                    "required_output_paths",
                    [],
                )
                or []
            )
        ]
    )

    missing_paths = [
        path
        for path in expected_paths
        if path not in actual_paths
    ]

    extra_paths = [
        path
        for path in actual_paths
        if path not in expected_paths
    ]

    expected_task_id = (
        inventory_row.get(
            "task_id"
        )
    )
    actual_task_id = getattr(
        spec,
        "task_id",
        None,
    )

    expected_category = (
        inventory_row.get(
            "category"
        )
    )
    actual_category = getattr(
        spec,
        "category",
        None,
    )

    expected_timeout = (
        inventory_row.get(
            "agent_timeout_sec"
        )
    )
    actual_timeout = getattr(
        spec,
        "runtime_seconds",
        None,
    )

    expected_network = (
        inventory_row.get(
            "network"
        )
    )
    actual_network = getattr(
        spec,
        "network_mode",
        None,
    )

    required_columns = list(
        getattr(
            spec,
            "required_output_columns",
            [],
        )
        or []
    )

    required_dtypes = dict(
        getattr(
            spec,
            "required_output_dtypes",
            {},
        )
        or {}
    )

    candidate_fields = list(
        getattr(
            spec,
            "candidate_schema_fields",
            [],
        )
        or []
    )

    hard_failure_types: list[
        str
    ] = []

    if missing_paths:
        hard_failure_types.append(
            "missing_output_path"
        )

    if extra_paths:
        hard_failure_types.append(
            "extra_output_path"
        )

    if (
        expected_task_id is not None
        and actual_task_id
        != expected_task_id
    ):
        hard_failure_types.append(
            "task_id_mismatch"
        )

    if (
        expected_category is not None
        and actual_category
        != expected_category
    ):
        hard_failure_types.append(
            "category_mismatch"
        )

    if (
        expected_timeout is not None
        and not numeric_equal(
            expected_timeout,
            actual_timeout,
        )
    ):
        hard_failure_types.append(
            "timeout_mismatch"
        )

    if (
        expected_network is not None
        and actual_network
        != expected_network
    ):
        hard_failure_types.append(
            "network_mismatch"
        )

    structured_count = len(
        signals[
            "structured_paths"
        ]
    )

    explicit_schema_signal = any(
        [
            signals[
                "markdown_table"
            ],
            signals[
                "json_signal"
            ],
            signals[
                "general_signal"
            ],
        ]
    )

    parser_has_output_schema = bool(
        required_columns
        or required_dtypes
    )

    schema_review_needed = bool(
        structured_count
        and explicit_schema_signal
        and not parser_has_output_schema
    )

    return AuditRow(
        unit_dir=unit_dir_name,
        task_id_expected=(
            expected_task_id
        ),
        task_id_actual=(
            actual_task_id
        ),
        category_expected=(
            expected_category
        ),
        category_actual=(
            actual_category
        ),
        timeout_expected=(
            expected_timeout
        ),
        timeout_actual=(
            actual_timeout
        ),
        network_expected=(
            expected_network
        ),
        network_actual=(
            actual_network
        ),
        expected_output_count=(
            len(
                expected_paths
            )
        ),
        actual_output_count=(
            len(
                actual_paths
            )
        ),
        expected_output_paths=(
            expected_paths
        ),
        actual_output_paths=(
            actual_paths
        ),
        missing_output_paths=(
            missing_paths
        ),
        extra_output_paths=(
            extra_paths
        ),
        structured_output_count=(
            structured_count
        ),
        required_output_columns=(
            required_columns
        ),
        required_output_dtypes=(
            required_dtypes
        ),
        candidate_schema_fields=(
            candidate_fields
        ),
        markdown_schema_table_signal=(
            signals[
                "markdown_table"
            ]
        ),
        json_schema_signal=(
            signals[
                "json_signal"
            ]
        ),
        nested_json_signal=(
            signals[
                "nested_json"
            ]
        ),
        general_schema_signal=(
            signals[
                "general_signal"
            ]
        ),
        schema_review_needed=(
            schema_review_needed
        ),
        hard_failure_types=(
            hard_failure_types
        ),
        hard_contract_pass=(
            not hard_failure_types
        ),
        load_error=None,
    )


def csv_value(
    value: Any,
) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        bool,
    ):
        return (
            "true"
            if value
            else "false"
        )

    if isinstance(
        value,
        (list, dict),
    ):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    return str(
        value
    )


def write_rows_csv(
    rows: list[AuditRow],
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dict_rows = [
        asdict(
            row
        )
        for row in rows
    ]

    fieldnames = list(
        dict_rows[0].keys()
        if dict_rows
        else AuditRow.__dataclass_fields__.keys()
    )

    with destination.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in dict_rows:
            writer.writerow(
                {
                    key: csv_value(
                        value
                    )
                    for key, value in row.items()
                }
            )


def build_report(
    *,
    rows: list[AuditRow],
    inventory_payload: dict[str, Any],
    public_repo: Path,
    expected_unit_count: int,
) -> dict[str, Any]:
    hard_failure_counts = Counter(
        failure
        for row in rows
        for failure
        in row.hard_failure_types
    )

    category_failures = Counter(
        row.category_expected
        or "<missing>"
        for row in rows
        if not row.hard_contract_pass
    )

    hard_passes = sum(
        1
        for row in rows
        if row.hard_contract_pass
    )

    schema_reviews = [
        row
        for row in rows
        if row.schema_review_needed
    ]

    nested_json_units = [
        row.unit_dir
        for row in rows
        if row.nested_json_signal
    ]

    return {
        "public_repo": str(
            public_repo
        ),
        "inventory_git_commit": (
            inventory_payload.get(
                "git_commit"
            )
        ),
        "expected_unit_count": (
            expected_unit_count
        ),
        "audited_unit_count": (
            len(
                rows
            )
        ),
        "unit_count_matches": (
            len(
                rows
            )
            == expected_unit_count
        ),
        "hard_contract_pass_count": (
            hard_passes
        ),
        "hard_contract_fail_count": (
            len(
                rows
            )
            - hard_passes
        ),
        "hard_contract_readiness_rate": (
            hard_passes
            / len(
                rows
            )
            if rows
            else 0.0
        ),
        "hard_failure_counts": dict(
            hard_failure_counts
        ),
        "hard_failures_by_category": dict(
            category_failures
        ),
        "schema_review_count": (
            len(
                schema_reviews
            )
        ),
        "schema_review_units": [
            row.unit_dir
            for row in schema_reviews
        ],
        "nested_json_signal_count": (
            len(
                nested_json_units
            )
        ),
        "nested_json_signal_units": (
            nested_json_units
        ),
        "rows": [
            asdict(
                row
            )
            for row in rows
        ],
    }


def print_summary(
    report: dict[str, Any],
) -> None:
    print(
        "=" * 72
    )
    print(
        "AGENTHON T1 — CONTRACT AUDIT"
    )
    print(
        "=" * 72
    )
    print(
        "Units audited: "
        f"{report['audited_unit_count']}"
    )
    print(
        "Expected units: "
        f"{report['expected_unit_count']}"
    )
    print(
        "Hard contract passes: "
        f"{report['hard_contract_pass_count']}"
    )
    print(
        "Hard contract failures: "
        f"{report['hard_contract_fail_count']}"
    )
    print(
        "Hard contract readiness: "
        f"{report['hard_contract_readiness_rate']:.4f}"
    )
    print(
        "Schema-review units: "
        f"{report['schema_review_count']}"
    )
    print(
        "Nested-JSON signal units: "
        f"{report['nested_json_signal_count']}"
    )

    failure_counts = (
        report[
            "hard_failure_counts"
        ]
    )

    if failure_counts:
        print(
            "\nHard failure types:"
        )
        for name, count in sorted(
            failure_counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):
            print(
                f"  {name}: {count}"
            )

    category_failures = (
        report[
            "hard_failures_by_category"
        ]
    )

    if category_failures:
        print(
            "\nHard failures by category:"
        )
        for name, count in sorted(
            category_failures.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):
            print(
                f"  {name}: {count}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the frozen Agenthon T1 specification parser "
            "against the 87-unit public contract inventory."
        )
    )

    parser.add_argument(
        "--repo",
        type=Path,
        default=DEFAULT_PUBLIC_REPO,
        help=(
            "Path to the official public Track-1 repository."
        ),
    )

    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY,
        help=(
            "Path to benchmark/public_units_inventory.json."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    parser.add_argument(
        "--expected-unit-count",
        type=int,
        default=87,
    )

    parser.add_argument(
        "--fail-on-hard-miss",
        action="store_true",
        help=(
            "Return exit code 2 when the hard contract audit "
            "finds a mismatch. Useful after the baseline is recorded."
        ),
    )

    args = parser.parse_args()

    public_repo = (
        args.repo
        .expanduser()
        .resolve()
    )

    inventory_path = (
        args.inventory
        .expanduser()
        .resolve()
    )

    out_dir = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    if not (
        public_repo
        / "units"
    ).is_dir():
        raise SystemExit(
            "Official units directory not found: "
            f"{public_repo / 'units'}"
        )

    if not inventory_path.is_file():
        raise SystemExit(
            "Inventory not found: "
            f"{inventory_path}"
        )

    payload = load_json(
        inventory_path
    )

    units = inventory_units(
        payload
    )

    rows = [
        audit_unit(
            public_repo,
            row,
        )
        for row in units
    ]

    report = build_report(
        rows=rows,
        inventory_payload=payload,
        public_repo=public_repo,
        expected_unit_count=(
            args.expected_unit_count
        ),
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_path = (
        out_dir
        / "contract_audit_rows.csv"
    )
    report_path = (
        out_dir
        / "contract_audit_report.json"
    )
    failures_path = (
        out_dir
        / "contract_audit_failures.csv"
    )

    write_rows_csv(
        rows,
        rows_path,
    )

    write_rows_csv(
        [
            row
            for row in rows
            if (
                not row.hard_contract_pass
                or row.schema_review_needed
            )
        ],
        failures_path,
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
        f"  {failures_path}"
    )
    print(
        f"  {report_path}"
    )

    hard_failure = (
        report[
            "hard_contract_fail_count"
        ]
        > 0
        or not report[
            "unit_count_matches"
        ]
    )

    if (
        args.fail_on_hard_miss
        and hard_failure
    ):
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )