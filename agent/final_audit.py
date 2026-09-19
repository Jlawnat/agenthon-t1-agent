from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
import os
import stat

from agent.compiled_specification import (
    CompiledSpecification,
)
from agent.quant_invariants import (
    QuantInvariantConfig,
    evaluate_quant_invariants,
)
from agent.schema_expectations import (
    SchemaExpectations,
)
from agent.structural_verifier import (
    verify_structural_output,
)


_FORBIDDEN_OUTPUT_NAMES = {
    "reward.json",
    "test_outputs.py",
    "test.sh",
}

_ALLOWED_ABSOLUTE_PREFIXES = (
    "/output/",
    "/app/output/",
)


@dataclass(frozen=True)
class AuditFinding:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FinalAuditResult:
    passed: bool
    findings: tuple[AuditFinding, ...]
    required_files: tuple[str, ...]
    produced_files: tuple[str, ...]

    @property
    def publishable(self) -> bool:
        # Structural/security/inventory failures remain fail-closed.
        # Quant-only hard failures may be judged by the official checker.
        hard_failures = [
            finding
            for finding in self.findings
            if (
                finding.status == "fail"
                and finding.severity == "hard_fail"
            )
        ]

        return all(
            finding.name.startswith("quant:")
            for finding in hard_failures
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(self)


def _normalise_required_path(
    raw_path: str,
) -> str:
    """
    Convert a declared output path to a safe path relative to
    the selected candidate's output directory.

    Only the two documented output mount prefixes may appear
    as absolute paths. Traversal, NULs, and other absolute
    locations fail closed.
    """

    path = str(
        raw_path
    ).replace(
        "\\",
        "/",
    ).strip()

    if not path:
        raise ValueError(
            "Required output path is empty."
        )

    if "\x00" in path:
        raise ValueError(
            "Required output path contains a NUL byte."
        )

    if path.startswith("/"):
        matched_prefix = None

        for prefix in (
            _ALLOWED_ABSOLUTE_PREFIXES
        ):
            if path.startswith(
                prefix
            ):
                matched_prefix = (
                    prefix
                )
                break

        if matched_prefix is None:
            raise ValueError(
                "Required output path uses an unsupported "
                "absolute location."
            )

        path = path[
            len(matched_prefix):
        ]

    # Accept a conservative relative spelling used by some
    # internal fixtures while still resolving under output_dir.
    if path.startswith(
        "output/"
    ):
        path = path[
            len("output/"):
        ]

    pure = PurePosixPath(
        path
    )

    if pure.is_absolute():
        raise ValueError(
            "Required output path must resolve under output_dir."
        )

    parts = pure.parts

    if (
        not parts
        or any(
            part in {
                "",
                ".",
                "..",
            }
            for part in parts
        )
    ):
        raise ValueError(
            "Required output path contains traversal "
            "or invalid components."
        )

    normalised = (
        pure.as_posix()
    )

    if normalised in {
        "",
        ".",
    }:
        raise ValueError(
            "Required output path does not name a file."
        )

    return normalised


def _required_files(
    compiled_specification:
        CompiledSpecification,
) -> list[str]:
    files: list[str] = []

    for deliverable in (
        compiled_specification
        .deliverables
    ):
        raw_path = (
            deliverable.get(
                "path"
            )
        )

        if not raw_path:
            continue

        normalised = (
            _normalise_required_path(
                str(raw_path)
            )
        )

        if (
            normalised
            not in files
        ):
            files.append(
                normalised
            )

    return files


def _inventory_output(
    output_dir: Path,
) -> tuple[
    list[str],
    list[str],
]:
    """
    Return regular files and unsafe filesystem entries.

    Symlinks are never followed. Non-regular files are unsafe.
    Directories are allowed only as containers for deliverables.
    """

    if not output_dir.exists():
        return (
            [],
            [],
        )

    files: list[str] = []
    unsafe: list[str] = []

    for current_root, dirs, names in os.walk(
        output_dir,
        topdown=True,
        followlinks=False,
    ):
        current = Path(
            current_root
        )

        safe_dirs: list[str] = []

        for name in dirs:
            path = current / name
            relative = (
                path.relative_to(
                    output_dir
                )
                .as_posix()
            )

            if path.is_symlink():
                unsafe.append(
                    relative
                )
                continue

            safe_dirs.append(
                name
            )

        dirs[:] = safe_dirs

        for name in names:
            path = current / name
            relative = (
                path.relative_to(
                    output_dir
                )
                .as_posix()
            )

            if path.is_symlink():
                unsafe.append(
                    relative
                )
                continue

            try:
                mode = (
                    path.stat(
                        follow_symlinks=False
                    ).st_mode
                )
            except OSError:
                unsafe.append(
                    relative
                )
                continue

            if not stat.S_ISREG(
                mode
            ):
                unsafe.append(
                    relative
                )
                continue

            files.append(
                relative
            )

    return (
        sorted(files),
        sorted(unsafe),
    )


def _produced_files(
    output_dir: Path,
) -> list[str]:
    files, _ = (
        _inventory_output(
            output_dir
        )
    )

    return files


def _has_hard_failure(
    findings: list[AuditFinding],
) -> bool:
    return any(
        finding.status == "fail"
        and finding.severity
        == "hard_fail"
        for finding in findings
    )


def _expected_format(
    compiled_specification:
        CompiledSpecification,
    required_file: str,
) -> str | None:
    for deliverable in (
        compiled_specification
        .deliverables
    ):
        raw_path = (
            deliverable.get(
                "path"
            )
        )

        if not raw_path:
            continue

        try:
            normalised = (
                _normalise_required_path(
                    str(raw_path)
                )
            )
        except ValueError:
            continue

        if (
            normalised
            != required_file
        ):
            continue

        value = deliverable.get(
            "format"
        )

        if value is None:
            return None

        return (
            str(value)
            .lower()
            .lstrip(".")
        )

    return None


def _format_matches(
    path: Path,
    expected_format: str | None,
) -> bool:
    if expected_format is None:
        return True

    actual = (
        path.suffix
        .lower()
        .lstrip(".")
    )

    aliases = {
        "jsonlines": "jsonl",
        "ndjson": "jsonl",
    }

    expected = aliases.get(
        expected_format,
        expected_format,
    )

    actual = aliases.get(
        actual,
        actual,
    )

    return (
        actual
        == expected
    )


def _safe_required_path(
    *,
    output_dir: Path,
    relative_path: str,
) -> Path | None:
    candidate = (
        output_dir
        / relative_path
    )

    current = (
        output_dir
    )

    for part in (
        PurePosixPath(
            relative_path
        ).parts
    ):
        current = (
            current
            / part
        )

        if current.is_symlink():
            return None

    try:
        resolved = (
            candidate.resolve()
        )
        resolved.relative_to(
            output_dir.resolve()
        )
    except (
        OSError,
        ValueError,
    ):
        return None

    if not resolved.exists():
        return resolved

    try:
        mode = resolved.stat(
            follow_symlinks=False
        ).st_mode
    except OSError:
        return None

    if not stat.S_ISREG(
        mode
    ):
        return None

    return resolved


def audit_final_output(
    *,
    output_dir: Path,
    compiled_specification:
        CompiledSpecification,
    schema_expectations:
        SchemaExpectations,
) -> FinalAuditResult:
    findings: list[
        AuditFinding
    ] = []

    raw_output_dir = Path(
        output_dir
    )

    if raw_output_dir.is_symlink():
        return FinalAuditResult(
            passed=False,
            findings=(
                AuditFinding(
                    name=(
                        "output_directory"
                    ),
                    status="fail",
                    severity="hard_fail",
                    message=(
                        "Final output directory "
                        "must not be a symlink."
                    ),
                ),
            ),
            required_files=(),
            produced_files=(),
        )

    output_dir = (
        raw_output_dir.resolve()
    )

    required_files: list[str] = []

    try:
        required_files = (
            _required_files(
                compiled_specification
            )
        )

    except ValueError as exc:
        findings.append(
            AuditFinding(
                name=(
                    "required_output_path"
                ),
                status="fail",
                severity="hard_fail",
                message=(
                    "Compiled specification "
                    "contains an unsafe required "
                    "output path."
                ),
                details={
                    "reason": str(
                        exc
                    ),
                },
            )
        )

    single_tabular_output = (
        len(required_files) == 1
        and Path(required_files[0]).suffix.lower()
        in {".csv", ".parquet", ".feather"}
    )

    (
        produced_files,
        unsafe_entries,
    ) = _inventory_output(
        output_dir
    )

    if not output_dir.exists():
        findings.append(
            AuditFinding(
                name="output_directory",
                status="fail",
                severity="hard_fail",
                message=(
                    "Final output directory "
                    "does not exist."
                ),
                details={
                    "output_dir": str(
                        output_dir
                    ),
                },
            )
        )

    elif not output_dir.is_dir():
        findings.append(
            AuditFinding(
                name="output_directory",
                status="fail",
                severity="hard_fail",
                message=(
                    "Final output path is "
                    "not a directory."
                ),
            )
        )

    else:
        findings.append(
            AuditFinding(
                name="output_directory",
                status="pass",
                severity="info",
                message=(
                    "Final output directory "
                    "exists."
                ),
            )
        )

    if unsafe_entries:
        findings.append(
            AuditFinding(
                name="unsafe_output_entries",
                status="fail",
                severity="hard_fail",
                message=(
                    "Final output contains "
                    "symlinks or non-regular "
                    "filesystem entries."
                ),
                details={
                    "unsafe_entries": (
                        unsafe_entries
                    ),
                },
            )
        )

    else:
        findings.append(
            AuditFinding(
                name="unsafe_output_entries",
                status="pass",
                severity="info",
                message=(
                    "Final output contains only "
                    "regular files and directories."
                ),
            )
        )

    if not required_files:
        findings.append(
            AuditFinding(
                name="required_deliverables",
                status="fail",
                severity="hard_fail",
                message=(
                    "Compiled specification "
                    "contains no safe required "
                    "deliverables."
                ),
            )
        )

    else:
        missing_files = [
            path
            for path in required_files
            if path not in (
                produced_files
            )
        ]

        if missing_files:
            findings.append(
                AuditFinding(
                    name=(
                        "required_deliverables"
                    ),
                    status="fail",
                    severity="hard_fail",
                    message=(
                        "Required final output "
                        "files are missing."
                    ),
                    details={
                        "missing_files": (
                            missing_files
                        ),
                    },
                )
            )

        else:
            findings.append(
                AuditFinding(
                    name=(
                        "required_deliverables"
                    ),
                    status="pass",
                    severity="info",
                    message=(
                        "All required final "
                        "output files exist."
                    ),
                )
            )

    unexpected_files = [
        path
        for path in produced_files
        if path not in (
            required_files
        )
    ]

    if unexpected_files:
        findings.append(
            AuditFinding(
                name="unexpected_files",
                status="fail",
                severity="hard_fail",
                message=(
                    "Final output contains "
                    "stale or unrequested "
                    "files."
                ),
                details={
                    "unexpected_files": (
                        unexpected_files
                    ),
                },
            )
        )

    else:
        findings.append(
            AuditFinding(
                name="unexpected_files",
                status="pass",
                severity="info",
                message=(
                    "No stale or unrequested "
                    "output files exist."
                ),
            )
        )

    forbidden_files = [
        path
        for path in produced_files
        if (
            Path(path)
            .name
            .lower()
            in _FORBIDDEN_OUTPUT_NAMES
        )
    ]

    if forbidden_files:
        findings.append(
            AuditFinding(
                name="forbidden_artifacts",
                status="fail",
                severity="hard_fail",
                message=(
                    "Forbidden benchmark or "
                    "reward artifacts exist "
                    "in final output."
                ),
                details={
                    "forbidden_files": (
                        forbidden_files
                    ),
                },
            )
        )

    else:
        findings.append(
            AuditFinding(
                name="forbidden_artifacts",
                status="pass",
                severity="info",
                message=(
                    "No forbidden benchmark "
                    "or reward artifacts "
                    "exist."
                ),
            )
        )

    quant_config = None

    if len(required_files) == 1:
        quant_config = (
            QuantInvariantConfig
            .from_compiled_specification(
                compiled_specification
            )
        )

    unsafe_set = set(
        unsafe_entries
    )

    for required_file in (
        required_files
    ):
        if required_file in (
            unsafe_set
        ):
            continue

        actual_path = (
            _safe_required_path(
                output_dir=(
                    output_dir
                ),
                relative_path=(
                    required_file
                ),
            )
        )

        if (
            actual_path is None
            or not actual_path.exists()
        ):
            continue

        expected_format = (
            _expected_format(
                compiled_specification,
                required_file,
            )
        )

        if not _format_matches(
            actual_path,
            expected_format,
        ):
            findings.append(
                AuditFinding(
                    name="output_format",
                    status="fail",
                    severity="hard_fail",
                    message=(
                        "Final output format "
                        "does not match the "
                        "compiled specification."
                    ),
                    details={
                        "file": (
                            required_file
                        ),
                        "expected_format": (
                            expected_format
                        ),
                        "actual_suffix": (
                            actual_path.suffix
                        ),
                    },
                )
            )

        else:
            findings.append(
                AuditFinding(
                    name="output_format",
                    status="pass",
                    severity="info",
                    message=(
                        "Final output format "
                        "matches the compiled "
                        "specification."
                    ),
                    details={
                        "file": (
                            required_file
                        ),
                    },
                )
            )

        structural = (
            verify_structural_output(
                output_path=(
                    actual_path
                ),
                required_columns=(
                    compiled_specification.required_columns
                    if single_tabular_output
                    else []
                ),
                expected_rows=(
                    schema_expectations.expected_rows
                    if single_tabular_output
                    else None
                ),
                id_column=(
                    schema_expectations.id_column
                    if single_tabular_output
                    else None
                ),
                expected_ids=(
                    schema_expectations.expected_ids
                    if single_tabular_output
                    else None
                ),
                require_id_order=(
                    schema_expectations.require_id_order
                    if single_tabular_output
                    else False
                ),
                expected_dtypes=(
                    schema_expectations.expected_dtypes
                    if single_tabular_output
                    else {}
                ),
            )
        )

        for item in structural:
            findings.append(
                AuditFinding(
                    name=(
                        f"structural:"
                        f"{item.name}"
                    ),
                    status=item.status,
                    severity=(
                        item.severity
                    ),
                    message=item.message,
                    details=dict(
                        item.details
                    ),
                )
            )

        if quant_config is not None:
            quant = (
                evaluate_quant_invariants(
                    output_path=(
                        actual_path
                    ),
                    config=quant_config,
                )
            )

            for item in quant:
                findings.append(
                    AuditFinding(
                        name=(
                            f"quant:"
                            f"{item.name}"
                        ),
                        status=item.status,
                        severity=(
                            item.severity
                        ),
                        message=item.message,
                        details=dict(
                            item.details
                        ),
                    )
                )

    passed = not _has_hard_failure(
        findings
    )

    return FinalAuditResult(
        passed=passed,
        findings=tuple(
            findings
        ),
        required_files=tuple(
            required_files
        ),
        produced_files=tuple(
            produced_files
        ),
    )