from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import re
import subprocess
import tomllib
from typing import Any


OUTPUT_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(/app/output/[A-Za-z0-9_./\-]+"
    r"|/output/[A-Za-z0-9_./\-]+)"
)

OUTPUT_FILENAME_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.\-])"
    r"([A-Za-z0-9_][A-Za-z0-9_.\-]*\."
    r"(?:csv|json|jsonl|parquet|"
    r"png|html|htm|py|txt|xlsx|xls|"
    r"feather|pkl|pickle|md))"
    r"(?![A-Za-z0-9_.\-])",
    flags=re.IGNORECASE,
)

OUTPUT_HEADING_FILENAME_PATTERN = re.compile(
    r"^#{1,6}\s+.*?\boutput\b.*?"
    r"([A-Za-z0-9_][A-Za-z0-9_.\-]*\."
    r"(?:csv|json|jsonl|parquet|"
    r"png|html|htm|py|txt|xlsx|xls|"
    r"feather|pkl|pickle|md))"
    r"(?![A-Za-z0-9_.\-])",
    flags=(
        re.IGNORECASE
        | re.MULTILINE
    ),
)

OUTPUT_LINE_FILENAME_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:[-*]\s*)?"
    r"(?:output|output file|output files)\s*[:\-]\s*"
    r"[`*_\s]*"
    r"([A-Za-z0-9_][A-Za-z0-9_.\-]*\."
    r"(?:csv|json|jsonl|parquet|"
    r"png|html|htm|py|txt|xlsx|xls|"
    r"feather|pkl|pickle|md))"
    r"(?![A-Za-z0-9_.\-])",
    flags=(
        re.IGNORECASE
        | re.MULTILINE
    ),
)

LEGACY_INPUT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(/app/(?!output(?:/|\b))[A-Za-z0-9_./\-]+"
    r"|/input/environment/data/[A-Za-z0-9_./\-]+)"
)

MARKDOWN_HEADING_PATTERN = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$",
    flags=re.MULTILINE,
)

OUTPUT_SECTION_KEYWORDS = (
    "output",
    "outputs",
    "output file",
    "output files",
    "required output",
    "required outputs",
    "required file",
    "required files",
    "deliverable",
    "deliverables",
    "save output",
    "save outputs",
    "save result",
    "save results",
    "write output",
    "write outputs",
)


def clean_path(
    value: str,
) -> str:
    return value.rstrip(
        "`'\".,;:)]}"
    )


def unique(
    values: list[str],
) -> list[str]:
    result: list[str] = []

    for value in values:
        if value not in result:
            result.append(
                value
            )

    return result


def get_output_section(
    instruction: str,
) -> str:
    """
    Extract the most likely Markdown section that defines output
    deliverables.

    Many T1 instructions declare only `/app/output` and list the
    actual filenames underneath as relative names. This function
    lets the inventory recover those filenames without requiring
    every path to be written in full.
    """

    headings = list(
        MARKDOWN_HEADING_PATTERN.finditer(
            instruction
        )
    )

    for index, heading in enumerate(
        headings
    ):
        heading_text = (
            heading.group(2)
            .strip()
            .lower()
        )

        if not any(
            keyword in heading_text
            for keyword
            in OUTPUT_SECTION_KEYWORDS
        ):
            continue

        heading_level = len(
            heading.group(1)
        )

        section_start = (
            heading.end()
        )

        section_end = len(
            instruction
        )

        for later in headings[
            index + 1:
        ]:
            later_level = len(
                later.group(1)
            )

            if (
                later_level
                <= heading_level
            ):
                section_end = (
                    later.start()
                )
                break

        return instruction[
            section_start:
            section_end
        ]

    return ""


def detect_output_root(
    instruction: str,
    output_section: str,
) -> str:
    """
    Infer the declared output root.

    `/app/output` is the primary T1 convention. `/output` is the
    dual-mounted compatibility alias.
    """

    combined = (
        output_section
        + "\n"
        + instruction
    )

    if re.search(
        r"/app/output(?:/|\b)",
        combined,
    ):
        return "/app/output"

    if re.search(
        r"/output(?:/|\b)",
        combined,
    ):
        return "/output"

    return "/app/output"


def extract_output_paths(
    instruction: str,
) -> list[str]:
    """
    Recover output contracts from the official T1 instruction
    styles.

    Supported forms include:

        /app/output/results.csv

        ## Output Files
        1. calibration.json
        2. summary.json

        ### Output: `true_values.json`
    """

    absolute_paths = unique(
        [
            clean_path(
                match
            )
            for match
            in OUTPUT_PATH_PATTERN.findall(
                instruction
            )
        ]
    )

    output_section = (
        get_output_section(
            instruction
        )
    )

    output_root = (
        detect_output_root(
            instruction,
            output_section,
        )
    )

    section_filenames: list[str] = []

    if output_section:
        section_filenames = unique(
            [
                clean_path(
                    match
                )
                for match
                in OUTPUT_FILENAME_PATTERN.findall(
                    output_section
                )
            ]
        )

    heading_filenames = unique(
        [
            clean_path(
                match
            )
            for match
            in OUTPUT_HEADING_FILENAME_PATTERN.findall(
                instruction
            )
        ]
    )

    line_filenames = unique(
        [
            clean_path(
                match
            )
            for match
            in OUTPUT_LINE_FILENAME_PATTERN.findall(
                instruction
            )
        ]
    )

    filenames = unique(
        section_filenames
        + heading_filenames
        + line_filenames
    )

    relative_paths = [
        (
            f"{output_root}/"
            f"{filename}"
        )
        for filename
        in filenames
    ]

    return unique(
        absolute_paths
        + relative_paths
    )


def get_git_commit(
    repo: Path,
) -> str | None:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

    except Exception:
        return None

    value = (
        completed
        .stdout
        .strip()
    )

    return value or None


def read_toml(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "rb"
    ) as handle:
        payload = tomllib.load(
            handle
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"TOML root must be a table: {path}"
        )

    return payload


def safe_get(
    payload: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    current: Any = payload

    for key in keys:
        if not isinstance(
            current,
            dict,
        ):
            return default

        if key not in current:
            return default

        current = current[
            key
        ]

    return current


def get_data_files(
    unit_dir: Path,
) -> list[str]:
    data_dir = (
        unit_dir
        / "environment"
        / "data"
    )

    if not data_dir.exists():
        return []

    files: list[str] = []

    for path in sorted(
        data_dir.rglob("*")
    ):
        if not path.is_file():
            continue

        files.append(
            path.relative_to(
                data_dir
            ).as_posix()
        )

    return files


def get_extension(
    path: str,
) -> str:
    suffix = (
        Path(path)
        .suffix
        .lower()
        .lstrip(".")
    )

    return (
        suffix
        or "<none>"
    )


def inspect_unit(
    unit_dir: Path,
) -> dict[str, Any]:
    card_path = (
        unit_dir
        / "card.toml"
    )

    instruction_path = (
        unit_dir
        / "instruction.md"
    )

    card = read_toml(
        card_path
    )

    instruction = ""

    if instruction_path.exists():
        instruction = (
            instruction_path
            .read_text(
                encoding="utf-8",
                errors="replace",
            )
        )

    output_paths = (
        extract_output_paths(
            instruction
        )
    )

    legacy_input_paths = unique(
        [
            clean_path(
                match
            )
            for match
            in LEGACY_INPUT_PATTERN.findall(
                instruction
            )
        ]
    )

    data_files = (
        get_data_files(
            unit_dir
        )
    )

    task_id = (
        safe_get(
            card,
            "task",
            "id",
        )
        or unit_dir.name
    )

    return {
        "unit_dir": (
            unit_dir.name
        ),

        "task_id": (
            task_id
        ),

        "title": safe_get(
            card,
            "task",
            "title",
        ),

        "split": safe_get(
            card,
            "task",
            "split",
        ),

        "category": safe_get(
            card,
            "metadata",
            "category",
        ),

        "difficulty": safe_get(
            card,
            "metadata",
            "difficulty",
        ),

        "agent_timeout_sec": safe_get(
            card,
            "agent",
            "timeout_sec",
        ),

        "verifier_timeout_sec": safe_get(
            card,
            "verifier",
            "timeout_sec",
        ),

        "cpus": safe_get(
            card,
            "environment",
            "cpus",
        ),

        "memory": safe_get(
            card,
            "environment",
            "memory",
        ),

        "gpu": safe_get(
            card,
            "environment",
            "gpu",
        ),

        "network": safe_get(
            card,
            "environment",
            "network",
        ),

        "scoring_metric": safe_get(
            card,
            "scoring",
            "metric",
        ),

        "k_values": safe_get(
            card,
            "scoring",
            "params",
            "k_values",
            default=[],
        ),

        "output_count": len(
            output_paths
        ),

        "output_paths": (
            output_paths
        ),

        "output_extensions": unique(
            [
                get_extension(
                    path
                )
                for path
                in output_paths
            ]
        ),

        "legacy_input_count": len(
            legacy_input_paths
        ),

        "legacy_input_paths": (
            legacy_input_paths
        ),

        "data_file_count": len(
            data_files
        ),

        "data_files": (
            data_files
        ),

        "has_manifest": (
            unit_dir
            / "manifest.json"
        ).is_file(),

        "has_checks": (
            unit_dir
            / "checks"
        ).is_dir(),

        "is_example": (
            "EXAMPLE"
            in unit_dir.name
        ),
    }


def join_value(
    value: Any,
) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        list,
    ):
        return "|".join(
            str(item)
            for item
            in value
        )

    return str(
        value
    )


def write_csv(
    rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    fields = [
        "unit_dir",
        "task_id",
        "title",
        "split",
        "category",
        "difficulty",
        "agent_timeout_sec",
        "verifier_timeout_sec",
        "cpus",
        "memory",
        "gpu",
        "network",
        "scoring_metric",
        "k_values",
        "output_count",
        "output_paths",
        "output_extensions",
        "legacy_input_count",
        "legacy_input_paths",
        "data_file_count",
        "data_files",
        "has_manifest",
        "has_checks",
        "is_example",
    ]

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open(
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
            writer.writerow(
                {
                    field: join_value(
                        row.get(
                            field
                        )
                    )
                    for field
                    in fields
                }
            )


def write_json(
    rows: list[dict[str, Any]],
    repo: Path,
    destination: Path,
) -> None:
    payload = {
        "repository": str(
            repo
        ),

        "git_commit": (
            get_git_commit(
                repo
            )
        ),

        "unit_count": len(
            rows
        ),

        "units": (
            rows
        ),
    }

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def print_summary(
    rows: list[dict[str, Any]],
) -> None:
    categories = Counter(
        str(
            row[
                "category"
            ]
            or "<missing>"
        )
        for row
        in rows
    )

    difficulties = Counter(
        str(
            row[
                "difficulty"
            ]
            or "<missing>"
        )
        for row
        in rows
    )

    extensions = Counter(
        extension
        for row
        in rows
        for extension
        in row[
            "output_extensions"
        ]
    )

    networks = Counter(
        str(
            row[
                "network"
            ]
            or "<missing>"
        )
        for row
        in rows
    )

    legacy_units = [
        row
        for row
        in rows
        if (
            row[
                "legacy_input_count"
            ]
            > 0
        )
    ]

    multi_output_units = [
        row
        for row
        in rows
        if (
            row[
                "output_count"
            ]
            > 1
        )
    ]

    no_output_detected = [
        row
        for row
        in rows
        if (
            row[
                "output_count"
            ]
            == 0
        )
    ]

    print(
        "=" * 72
    )

    print(
        "AGENTHON T1 — PUBLIC CORPUS INVENTORY"
    )

    print(
        "=" * 72
    )

    print(
        "Units discovered: "
        f"{len(rows)}"
    )

    print(
        "Units with legacy input paths: "
        f"{len(legacy_units)}"
    )

    print(
        "Units with multiple explicit outputs: "
        f"{len(multi_output_units)}"
    )

    print(
        "Units with no explicit output path detected: "
        f"{len(no_output_detected)}"
    )

    print(
        "\nCategories:"
    )

    for name, count in (
        categories.most_common()
    ):
        print(
            f"  {name}: {count}"
        )

    print(
        "\nDifficulty:"
    )

    for name, count in (
        difficulties.most_common()
    ):
        print(
            f"  {name}: {count}"
        )

    print(
        "\nOutput extensions:"
    )

    for name, count in (
        extensions.most_common()
    ):
        print(
            f"  {name}: {count}"
        )

    print(
        "\nNetwork modes:"
    )

    for name, count in (
        networks.most_common()
    ):
        print(
            f"  {name}: {count}"
        )

    if no_output_detected:
        print(
            "\nUnits requiring manual output-contract review:"
        )

        for row in (
            no_output_detected
        ):
            print(
                f"  {row['unit_dir']}"
            )


def main() -> int:
    parser = (
        argparse.ArgumentParser(
            description=(
                "Inventory the public "
                "Agenthon Track-1 corpus."
            )
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
        default=Path(
            "benchmark"
        ),
    )

    args = (
        parser.parse_args()
    )

    repo = (
        args.repo
        .expanduser()
        .resolve()
    )

    units_dir = (
        repo
        / "units"
    )

    if not units_dir.is_dir():
        raise SystemExit(
            "Official units directory "
            f"not found: {units_dir}"
        )

    unit_dirs = sorted(
        path
        for path
        in units_dir.iterdir()
        if (
            path.is_dir()
            and (
                path
                / "card.toml"
            ).is_file()
        )
    )

    rows = [
        inspect_unit(
            unit_dir
        )
        for unit_dir
        in unit_dirs
    ]

    output_dir = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    csv_path = (
        output_dir
        / "public_units_inventory.csv"
    )

    json_path = (
        output_dir
        / "public_units_inventory.json"
    )

    write_csv(
        rows,
        csv_path,
    )

    write_json(
        rows,
        repo,
        json_path,
    )

    print_summary(
        rows
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {csv_path}"
    )

    print(
        f"  {json_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
