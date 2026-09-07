from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import re
import tomllib


@dataclass
class TaskSpecification:
    task_id: str | None
    category: str | None
    instruction_text: str

    required_output_paths: list[str]
    candidate_schema_fields: list[str]
    required_output_columns: list[str]

    runtime_seconds: int | float | None
    network_mode: str | None

    raw_card: dict[str, Any]
    required_output_dtypes: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.required_output_dtypes is None:
            self.required_output_dtypes = {}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


IGNORED_IDENTIFIERS = {
    "str",
    "int",
    "int32",
    "int64",
    "float",
    "float32",
    "float64",
    "bool",
    "object",
    "numpy",
    "pandas",
    "scipy",
    "pyarrow",
    "numba",
    "polars",
    "sklearn",
    "csv",
    "json",
    "jsonl",
    "parquet",
    "txt",
}


_FORBIDDEN_DELIVERABLE_BASENAMES = {
    "reward.json",
    "reward.txt",
    "pytest_report.json",
    "test_outputs.py",
    "test.sh",
}


_OUTPUT_FILENAME_PATTERN = re.compile(
    (
        r"(?<![A-Za-z0-9_.\-])"
        r"("
        r"[A-Za-z0-9_]"
        r"[A-Za-z0-9_.\-]*"
        r"\."
        r"(?:"
        r"csv|json|jsonl|parquet|"
        r"png|html|htm|py|txt|"
        r"xlsx|xls|feather|pkl|pickle|md"
        r")"
        r")"
        r"(?![A-Za-z0-9_.\-])"
    ),
    flags=re.IGNORECASE,
)


_EXPLICIT_OUTPUT_PATH_PATTERN = re.compile(
    (
        r"`?("
        r"(?:/app/output|/output)"
        r"/[A-Za-z0-9_./\-]+"
        r")`?"
    )
)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []

    for value in values:
        if value not in result:
            result.append(value)

    return result


def _is_output_path_text(
    value: str,
) -> bool:
    lowered = value.lower()

    return (
        "/output/" in lowered
        or "/app/output/" in lowered
    )


def _is_forbidden_deliverable(
    value: str,
) -> bool:
    basename = (
        Path(
            value.replace(
                "\\",
                "/",
            )
        )
        .name
        .lower()
    )

    return (
        basename
        in _FORBIDDEN_DELIVERABLE_BASENAMES
    )


def _strip_explicit_output_paths(
    value: str,
) -> str:
    """
    Remove already-explicit output paths before scanning for
    relative filenames.

    Without this guard, a heading such as

        ### /app/output/nested/results.json

    would correctly produce the nested path and then incorrectly
    produce a second /app/output/results.json path from the basename.
    """

    return _EXPLICIT_OUTPUT_PATH_PATTERN.sub(
        " ",
        value,
    )


def _extract_output_paths(
    instruction: str,
) -> list[str]:
    """
    Extract participant deliverables from Track-1 instructions.

    Supported public-task forms include:

    1. Explicit full paths:
       /app/output/results.csv
       /output/results.csv
       /app/output/nested/results.json

    2. A declared output directory followed by relative filenames:
       ## Required outputs
       Write everything to /app/output/.
       ### calibration.json
       ### prices.csv

    3. Filenames declared directly in output headings:
       ### Output: true_values.json
       ### Output: estimates.csv

    The parser deliberately avoids promoting input/configuration
    filenames to deliverables and always excludes verifier-owned
    artifacts such as reward.json and pytest_report.json.
    """

    # ---------------------------------------------------------
    # 1. Explicit full output paths.
    # ---------------------------------------------------------
    explicit_paths: list[str] = []

    for match in _EXPLICIT_OUTPUT_PATH_PATTERN.finditer(
        instruction
    ):
        path = (
            match.group(1)
            .rstrip(".,;:)")
        )

        if _is_forbidden_deliverable(
            path
        ):
            continue

        explicit_paths.append(
            path
        )

    explicit_paths = _unique(
        explicit_paths
    )

    # ---------------------------------------------------------
    # 2. Locate Markdown headings and output/deliverable sections.
    # ---------------------------------------------------------
    heading_pattern = re.compile(
        r"^(#{1,6})\s+(.+?)\s*$",
        flags=re.MULTILINE,
    )

    headings = list(
        heading_pattern.finditer(
            instruction
        )
    )

    output_sections: list[str] = []
    heading_filenames: list[str] = []

    for index, heading in enumerate(
        headings
    ):
        heading_text = (
            heading.group(2)
            .strip()
        )
        lowered = (
            heading_text.lower()
        )

        if not (
            "output" in lowered
            or "deliverable" in lowered
        ):
            continue

        # A relative filename may be part of the heading:
        #
        #   ### Output: `true_values.json`
        #
        # First remove any already-explicit path so
        #
        #   ### /app/output/nested/results.json
        #
        # does NOT yield an extra root-level results.json.
        heading_scan_text = (
            _strip_explicit_output_paths(
                heading_text
            )
        )

        for filename in (
            _OUTPUT_FILENAME_PATTERN.findall(
                heading_scan_text
            )
        ):
            if _is_forbidden_deliverable(
                filename
            ):
                continue

            heading_filenames.append(
                filename
            )

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

        output_sections.append(
            instruction[
                section_start:
                section_end
            ]
        )

    # ---------------------------------------------------------
    # 3. Determine the root for recovered relative filenames.
    #
    # /app/output is the primary Track-1 convention. /output is
    # the dual-mounted alias and is preserved when that is the
    # only spelling present.
    # ---------------------------------------------------------
    if re.search(
        r"/app/output(?:/|\b)",
        instruction,
    ):
        output_root = (
            "/app/output"
        )
    elif re.search(
        r"/output(?:/|\b)",
        instruction,
    ):
        output_root = (
            "/output"
        )
    else:
        output_root = (
            "/app/output"
        )

    relative_filenames: list[str] = []

    # Filenames declared directly by an output heading are strong
    # contract evidence.
    relative_filenames.extend(
        heading_filenames
    )

    # ---------------------------------------------------------
    # 4. Recover filenames from output sections.
    #
    # If a section declares /app/output or /output, scan only from
    # that declaration onward. This is important for headings such
    # as "Inputs and Outputs": input tables before the declaration
    # may contain params.json, task_rules.json, etc.
    #
    # Explicit full paths are removed before relative scanning so
    # nested paths cannot create duplicate root-level basenames.
    # ---------------------------------------------------------
    for section in output_sections:
        root_match = re.search(
            r"(?:/app/output|/output)(?:/|\b)",
            section,
        )

        if root_match is not None:
            scan_text = (
                section[
                    root_match.start():
                ]
            )

            scan_text = (
                _strip_explicit_output_paths(
                    scan_text
                )
            )

            candidates = (
                _OUTPUT_FILENAME_PATTERN.findall(
                    scan_text
                )
            )
        else:
            # No declared directory: only accept filenames from
            # strong Markdown/output-like lines, not arbitrary prose.
            candidates: list[str] = []

            for line in (
                section.splitlines()
            ):
                stripped = (
                    line.strip()
                )

                strong_output_line = (
                    stripped.startswith(
                        "#"
                    )
                    or re.match(
                        r"^(?:\d+[.)]|[-*+])\s+",
                        stripped,
                    )
                    is not None
                    or re.search(
                        r"\b(?:output|file)\s*\d*\s*:",
                        stripped,
                        flags=re.IGNORECASE,
                    )
                    is not None
                )

                if not strong_output_line:
                    continue

                line_scan_text = (
                    _strip_explicit_output_paths(
                        stripped
                    )
                )

                candidates.extend(
                    _OUTPUT_FILENAME_PATTERN.findall(
                        line_scan_text
                    )
                )

        for filename in candidates:
            if _is_forbidden_deliverable(
                filename
            ):
                continue

            relative_filenames.append(
                filename
            )

    relative_paths = [
        f"{output_root}/{filename}"
        for filename
        in _unique(
            relative_filenames
        )
    ]

    return _unique(
        explicit_paths
        + relative_paths
    )


def _extract_required_columns(
    instruction: str,
) -> list[str]:
    """
    Extract likely schema fields while excluding library names,
    dtype names, filenames, paths, and other code tokens.
    """

    candidates = re.findall(
        r"`([A-Za-z_][A-Za-z0-9_]*)`",
        instruction,
    )

    result: list[str] = []

    for name in candidates:
        if (
            name.lower()
            in IGNORED_IDENTIFIERS
        ):
            continue

        if name not in result:
            result.append(
                name
            )

    return result


def _extract_required_output_columns(
    instruction: str,
) -> list[str]:
    """
    Extract output columns from explicit participant-output
    schema sections.

    Supports both /app/output/... and /output/... headings.
    """

    result: list[str] = []

    lines = (
        instruction.splitlines()
    )

    inside_output_section = False
    inside_column_table = False

    for line in lines:
        stripped = line.strip()
        lowered = (
            stripped.lower()
        )

        if (
            stripped.startswith("#")
            and _is_output_path_text(
                lowered
            )
        ):
            inside_output_section = True
            inside_column_table = False
            continue

        if (
            inside_output_section
            and stripped.startswith("#")
            and not _is_output_path_text(
                lowered
            )
        ):
            inside_output_section = False
            inside_column_table = False
            continue

        if not inside_output_section:
            continue

        if (
            stripped.startswith("|")
            and "column" in lowered
            and "type" in lowered
        ):
            inside_column_table = True
            continue

        if not inside_column_table:
            continue

        if re.fullmatch(
            r"\|[\s:\-|]+\|?",
            stripped,
        ):
            continue

        if not stripped.startswith(
            "|"
        ):
            inside_column_table = False
            continue

        cells = [
            cell.strip()
            for cell
            in stripped
            .strip("|")
            .split("|")
        ]

        if not cells:
            continue

        match = re.fullmatch(
            (
                r"`("
                r"[A-Za-z_]"
                r"[A-Za-z0-9_]*"
                r")`"
            ),
            cells[0],
        )

        if match:
            column_name = (
                match.group(1)
            )

            if (
                column_name
                not in result
            ):
                result.append(
                    column_name
                )

    return result


def _extract_required_output_dtypes(
    instruction: str,
) -> dict[str, str]:
    """
    Extract declared output dtypes from Markdown schema tables
    inside /app/output/... or /output/... sections.

    Example:
    | `id` | `int64` |
    | `price` | `float64` |
    """

    result: dict[str, str] = {}

    lines = instruction.splitlines()

    inside_output_section = False
    inside_column_table = False

    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()

        if (
            stripped.startswith("#")
            and _is_output_path_text(
                lowered
            )
        ):
            inside_output_section = True
            inside_column_table = False
            continue

        if (
            inside_output_section
            and stripped.startswith("#")
            and not _is_output_path_text(
                lowered
            )
        ):
            inside_output_section = False
            inside_column_table = False
            continue

        if not inside_output_section:
            continue

        if (
            stripped.startswith("|")
            and "column" in lowered
            and "type" in lowered
        ):
            inside_column_table = True
            continue

        if not inside_column_table:
            continue

        if re.fullmatch(
            r"\|[\s:\-|]+\|?",
            stripped,
        ):
            continue

        if not stripped.startswith("|"):
            inside_column_table = False
            continue

        cells = [
            cell.strip()
            for cell
            in stripped
            .strip("|")
            .split("|")
        ]

        if len(cells) < 2:
            continue

        column_match = re.fullmatch(
            r"`([A-Za-z_][A-Za-z0-9_]*)`",
            cells[0],
        )

        dtype_match = re.fullmatch(
            r"`([^`]+)`",
            cells[1],
        )

        if (
            column_match is not None
            and dtype_match is not None
        ):
            column = (
                column_match.group(1)
            )
            dtype = (
                dtype_match.group(1)
                .strip()
            )

            if dtype:
                result[column] = dtype

    return result


def load_task_specification(
    task_dir: Path,
) -> TaskSpecification:
    task_dir = (
        task_dir.resolve()
    )

    instruction_path = (
        task_dir
        / "instruction.md"
    )
    card_path = (
        task_dir
        / "card.toml"
    )

    if not instruction_path.exists():
        raise FileNotFoundError(
            "instruction.md not found"
        )

    instruction_text = (
        instruction_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )

    raw_card: dict[
        str,
        Any,
    ] = {}

    if card_path.exists():
        with card_path.open(
            "rb"
        ) as handle:
            raw_card = (
                tomllib.load(
                    handle
                )
            )

    task = raw_card.get(
        "task",
        {},
    )
    metadata = raw_card.get(
        "metadata",
        {},
    )
    agent = raw_card.get(
        "agent",
        {},
    )
    environment = (
        raw_card.get(
            "environment",
            {},
        )
    )

    task_id = (
        task.get("id")
        or raw_card.get(
            "task_id"
        )
        or metadata.get(
            "task_id"
        )
        or metadata.get(
            "id"
        )
        or task_dir.name
    )

    category = (
        metadata.get(
            "category"
        )
    )

    # Track 1 v2: [agent].timeout_sec is authoritative when present.
    # Preserve the historical environment timeout as a compatibility
    # fallback for older fixtures/cards used throughout earlier phases.
    runtime_seconds = (
        agent.get(
            "timeout_sec"
        )
    )

    if runtime_seconds is None:
        runtime_seconds = (
            environment.get(
                "timeout_seconds"
            )
            or environment.get(
                "runtime_seconds"
            )
        )

    network_mode = (
        environment.get(
            "network"
        )
    )

    return TaskSpecification(
        task_id=task_id,
        category=category,
        instruction_text=(
            instruction_text
        ),
        required_output_paths=(
            _extract_output_paths(
                instruction_text
            )
        ),
        candidate_schema_fields=(
            _extract_required_columns(
                instruction_text
            )
        ),
        required_output_columns=(
            _extract_required_output_columns(
                instruction_text
            )
        ),
        required_output_dtypes=(
            _extract_required_output_dtypes(
                instruction_text
            )
        ),
        runtime_seconds=(
            runtime_seconds
        ),
        network_mode=(
            network_mode
        ),
        raw_card=raw_card,
    )
