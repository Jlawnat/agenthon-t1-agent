from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
    field,
)
from enum import Enum
from typing import Any

import re

from agent.budget import RepairBudget
from agent.candidate_contract import (
    CandidateAttempt,
    VerificationEvidence,
)


class FailureLabel(str, Enum):
    """
    Root-cause failure classes supported
    by the targeted repair engine.
    """

    SYNTAX = "syntax"
    IMPORT = "import"
    RUNTIME = "runtime"
    SCHEMA = "schema"
    DTYPE = "dtype"
    MISSING_FILE = "missing_file"
    NUMERICAL_RECONCILIATION = (
        "numerical_reconciliation"
    )


@dataclass(frozen=True)
class RepairStrategy:
    """
    Deterministic repair strategy selected
    from a classified failure.
    """

    name: str
    objective: str
    instructions: tuple[str, ...]


@dataclass(frozen=True)
class RepairBrief:
    """
    Safe, bounded evidence passed to
    one candidate repair call.
    """

    failure_label: str
    strategy: str
    objective: str

    safe_traceback_excerpt: str = ""

    violated_contract: tuple[
        dict[str, Any],
        ...,
    ] = ()

    schema_delta: dict[str, Any] = (
        field(default_factory=dict)
    )

    relevant_invariants: tuple[
        dict[str, Any],
        ...,
    ] = ()

    instructions: tuple[str, ...] = ()

    source_attempt_number: int | None = None
    repair_attempt_number: int | None = None

    budget: dict[str, Any] = (
        field(default_factory=dict)
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_STRATEGIES: dict[
    FailureLabel,
    RepairStrategy,
] = {

    FailureLabel.SYNTAX: RepairStrategy(
        name="repair_python_syntax",
        objective=(
            "Produce syntactically valid Python "
            "without changing the requested "
            "task contract."
        ),
        instructions=(
            (
                "Fix only the parse/syntax defect "
                "indicated by the safe error excerpt."
            ),
            (
                "Preserve the required input/output "
                "contract and quantitative method "
                "unless a local rewrite is required."
            ),
            (
                "Return complete executable Python "
                "code, not a patch or explanation."
            ),
        ),
    ),

    FailureLabel.IMPORT: RepairStrategy(
        name="repair_import_dependency",
        objective=(
            "Remove or replace an unavailable or "
            "incorrect import using only dependencies "
            "available in the image."
        ),
        instructions=(
            (
                "Correct the failing import or replace "
                "it with an available standard-library "
                "or in-image alternative."
            ),
            (
                "Do not install packages at runtime "
                "and do not access the internet."
            ),
            (
                "Preserve the requested output contract "
                "and computation semantics."
            ),
        ),
    ),

    FailureLabel.RUNTIME: RepairStrategy(
        name="repair_runtime_exception",
        objective=(
            "Correct the public runtime failure while "
            "preserving the requested computation "
            "and outputs."
        ),
        instructions=(
            (
                "Use the safe traceback excerpt to "
                "locate the runtime failure."
            ),
            (
                "Fix the root cause rather than "
                "suppressing the exception or "
                "fabricating outputs."
            ),
            (
                "Keep all task-required filenames, "
                "schemas, units, and conventions "
                "unchanged."
            ),
        ),
    ),

    FailureLabel.MISSING_FILE: RepairStrategy(
        name="repair_missing_deliverable",
        objective=(
            "Create every required deliverable at "
            "the exact requested output filename "
            "and path."
        ),
        instructions=(
            (
                "Write every required output file "
                "under the provided output directory."
            ),
            (
                "Use the exact required filenames; "
                "do not substitute aliases or add "
                "extra directory levels."
            ),
            (
                "Ensure files are fully written "
                "before successful process exit."
            ),
        ),
    ),

    FailureLabel.SCHEMA: RepairStrategy(
        name="repair_output_schema",
        objective=(
            "Make the produced deliverable conform "
            "exactly to the required structural "
            "contract."
        ),
        instructions=(
            (
                "Correct missing or extra structural "
                "elements, row coverage, identifier "
                "coverage, and required ordering shown "
                "in the safe schema delta."
            ),
            (
                "Preserve the quantitative calculation; "
                "change serialization or assembly only "
                "where possible."
            ),
            (
                "Do not hardcode benchmark answers "
                "or hidden reference values."
            ),
        ),
    ),

    FailureLabel.DTYPE: RepairStrategy(
        name="repair_output_dtype",
        objective=(
            "Emit the required data types without "
            "changing the meaning or units of values."
        ),
        instructions=(
            (
                "Cast or construct affected fields "
                "using the required dtype shown in "
                "the safe schema delta."
            ),
            (
                "Avoid string/object fallbacks for "
                "numeric fields and preserve identifier "
                "formatting exactly."
            ),
            (
                "Do not round or rescale values unless "
                "the public task contract requires it."
            ),
        ),
    ),

    FailureLabel.NUMERICAL_RECONCILIATION:
        RepairStrategy(
            name=(
                "repair_numerical_reconciliation"
            ),
            objective=(
                "Correct a deterministic financial "
                "or numerical inconsistency using "
                "an independent recomputation."
            ),
            instructions=(
                (
                    "Recompute the violated identity "
                    "or invariant independently and "
                    "find the discrepancy source."
                ),
                (
                    "Check sign, units, timing, "
                    "compounding, indexing, and "
                    "tolerance conventions first."
                ),
                (
                    "Do not hardcode target values; "
                    "repair the general computation."
                ),
            ),
        ),
}
class UnsafeRepairEvidenceError(RuntimeError):
    """
    Raised when evidence appears to contain
    benchmark-private or oracle-like material.
    """


_TRACEBACK_MAX_CHARS = 3000
_TRACEBACK_MAX_LINES = 24


_FORBIDDEN_EVIDENCE_PATTERNS = (
    r"reward\.json",
    r"checks[/\\]test_outputs\.py",
    r"checks[/\\]test\.sh",
    r"\bhidden[\s_-]*checker\b",
    r"\bchecker[\s_-]*implementation\b",
    r"\bcanary\b",
    r"\boracle\b",
    r"\breference[\s_-]*output\b",
    r"\bexpected[\s_-]*answer\b",
    r"\bground[\s_-]*truth[\s_-]*answer\b",
    r"\bgrader[\s_-]*secret\b",
)


_ANSI_ESCAPE_RE = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
)


def _contains_forbidden_evidence(
    text: str,
) -> bool:
    """
    Detect material that must never enter
    a repair prompt.
    """

    for pattern in _FORBIDDEN_EVIDENCE_PATTERNS:
        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            return True

    return False


def _redact_traceback_paths(
    text: str,
) -> str:
    """
    Preserve useful Python filenames and line
    numbers while removing host-specific paths.
    """

    file_pattern = re.compile(
        r'File "([^"]+)"'
    )

    def replace_file(
        match: re.Match[str],
    ) -> str:
        raw_path = match.group(1)

        normalised = raw_path.replace(
            "\\",
            "/",
        )

        filename = normalised.rsplit(
            "/",
            1,
        )[-1]

        return (
            'File "<candidate>/'
            f'{filename}"'
        )

    return file_pattern.sub(
        replace_file,
        text,
    )


def sanitize_public_traceback(
    text: str,
    *,
    max_chars: int = _TRACEBACK_MAX_CHARS,
    max_lines: int = _TRACEBACK_MAX_LINES,
) -> str:
    """
    Return a bounded traceback excerpt suitable
    for a candidate repair prompt.

    Fail closed if the input appears to contain
    checker, canary, oracle, reference-answer,
    or reward material.
    """

    if not text:
        return ""

    if max_chars <= 0:
        raise ValueError(
            "max_chars must be positive"
        )

    if max_lines <= 0:
        raise ValueError(
            "max_lines must be positive"
        )

    cleaned = _ANSI_ESCAPE_RE.sub(
        "",
        str(text),
    )

    if _contains_forbidden_evidence(
        cleaned
    ):
        raise UnsafeRepairEvidenceError(
            "Repair evidence contained "
            "benchmark-private or oracle-like "
            "material and was rejected."
        )

    cleaned = _redact_traceback_paths(
        cleaned
    )

    lines = cleaned.splitlines()


    if len(lines) > max_lines:
        lines = lines[-max_lines:]

        cleaned = (
            "[traceback excerpt truncated]\n"
            + "\n".join(lines)
        )

    else:
        cleaned = "\n".join(lines)

    if len(cleaned) > max_chars:
        cleaned = (
            "[traceback excerpt truncated]\n"
            + cleaned[-max_chars:]
        )


    if _contains_forbidden_evidence(
        cleaned
    ):
        raise UnsafeRepairEvidenceError(
            "Sanitized repair evidence still "
            "contained prohibited material."
        )

    return cleaned.strip()
def _failed_evidence(
    evidence: list[VerificationEvidence],
) -> list[VerificationEvidence]:
    """
    Return deterministic hard-failure evidence only.
    """

    return [
        item
        for item in evidence
        if (
            item.status == "fail"
            and item.severity == "hard_fail"
        )
    ]


def _text_matches_any(
    text: str,
    patterns: tuple[str, ...],
) -> bool:
    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in patterns
    )


_SYNTAX_ERROR_PATTERNS = (
    r"\bSyntaxError\b",
    r"\bIndentationError\b",
    r"\bTabError\b",
    r"not valid Python",
)


_IMPORT_ERROR_PATTERNS = (
    r"\bModuleNotFoundError\b",
    r"\bImportError\b",
    r"cannot import name",
    r"no module named",
)


_MISSING_FILE_EVIDENCE = {
    "required_outputs",
    "output_exists",
}


_DTYPE_EVIDENCE = {
    "dtype",
    "dtypes",
    "dtype_mismatch",
    "required_dtypes",
    "column_dtypes",
}


_SCHEMA_EVIDENCE = {
    "output_readable",
    "required_columns",
    "row_count",
    "null_values",
    "identifier_column",
    "duplicate_identifiers",
    "identifier_set",
    "identifier_order",
    "finite_numeric_values",
}


def classify_failure(
    attempt: CandidateAttempt,
    *,
    validation_error: str | None = None,
) -> FailureLabel | None:
    """
    Classify the primary repairable root cause.

    Priority matters.

    An execution crash can also cause a missing
    output file. In that case we repair the crash,
    not the downstream missing-file symptom.
    """

    stderr = attempt.stderr or ""

    diagnostic_text = "\n".join(
        part
        for part in (
            validation_error or "",
            stderr,
        )
        if part
    )


    if _text_matches_any(
        diagnostic_text,
        _SYNTAX_ERROR_PATTERNS,
    ):
        return FailureLabel.SYNTAX

    if _text_matches_any(
        diagnostic_text,
        _IMPORT_ERROR_PATTERNS,
    ):
        return FailureLabel.IMPORT

    structural_failures = _failed_evidence(
        attempt.structural_evidence
    )

    quant_failures = _failed_evidence(
        attempt.quant_evidence
    )

    structural_names = {
        item.name
        for item in structural_failures
    }

    if attempt.timed_out:
        return FailureLabel.RUNTIME

    if (
        attempt.return_code is not None
        and attempt.return_code != 0
    ):
        return FailureLabel.RUNTIME

    if (
        "execution_timeout" in structural_names
        or "execution_return_code"
        in structural_names
    ):
        return FailureLabel.RUNTIME

    if structural_names.intersection(
        _MISSING_FILE_EVIDENCE
    ):
        return FailureLabel.MISSING_FILE


    if structural_names.intersection(
        _DTYPE_EVIDENCE
    ):
        return FailureLabel.DTYPE


    if structural_names.intersection(
        _SCHEMA_EVIDENCE
    ):
        return FailureLabel.SCHEMA

    if quant_failures:
        return (
            FailureLabel
            .NUMERICAL_RECONCILIATION
        )
    if structural_failures:
        return FailureLabel.RUNTIME
    return None
_SAFE_STRUCTURAL_DETAIL_KEYS = {
    "required_outputs": {
        "missing",
    },
    "required_columns": {
        "missing_columns",
        "actual_columns",
    },
    "row_count": {
        "expected",
        "actual",
    },
    "null_values": {
        "columns",
    },
    "finite_numeric_values": {
        "columns",
    },
    "duplicate_identifiers": {
        "duplicate_count",
    },
    "dtype": {
        "column",
        "expected_dtype",
        "actual_dtype",
        "mismatches",
    },
    "dtypes": {
        "mismatches",
    },
    "dtype_mismatch": {
        "column",
        "expected_dtype",
        "actual_dtype",
        "mismatches",
    },
    "required_dtypes": {
        "mismatches",
    },
    "column_dtypes": {
        "mismatches",
    },
}


_SAFE_QUANT_DETAIL_KEYS = {
    "violation_count",
    "sample_row_indices",
    "tolerance",
    "maximum_absolute_residual",
    "maximum_relative_residual",
    "columns",
    "violations",
    "unit",
    "convention",
}


def _safe_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        (bool, int, float),
    ):
        return value

    if isinstance(value, str):
        if _contains_forbidden_evidence(
            value
        ):
            raise UnsafeRepairEvidenceError(
                "Prohibited material found "
                "inside repair evidence."
            )

        cleaned = _ANSI_ESCAPE_RE.sub(
            "",
            value,
        )

        cleaned = _redact_traceback_paths(
            cleaned
        )

        return cleaned[:500]

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _safe_value(item)
            for item in value[:50]
        ]

    if isinstance(value, dict):
        result: dict[str, Any] = {}

        for index, (key, item) in enumerate(
            value.items()
        ):
            if index >= 50:
                break

            safe_key = str(key)

            if _contains_forbidden_evidence(
                safe_key
            ):
                raise UnsafeRepairEvidenceError(
                    "Prohibited evidence key "
                    "detected."
                )

            result[safe_key] = (
                _safe_value(item)
            )

        return result

    return str(value)[:500]


def _safe_structural_details(
    item: VerificationEvidence,
) -> dict[str, Any]:
    allowed = (
        _SAFE_STRUCTURAL_DETAIL_KEYS.get(
            item.name,
            set(),
        )
    )

    result: dict[str, Any] = {}

    for key in sorted(allowed):
        if key not in item.details:
            continue

        result[key] = _safe_value(
            item.details[key]
        )

    return result


def _safe_quant_details(
    item: VerificationEvidence,
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key in sorted(
        _SAFE_QUANT_DETAIL_KEYS
    ):
        if key not in item.details:
            continue

        result[key] = _safe_value(
            item.details[key]
        )

    return result


def _build_violated_contract(
    attempt: CandidateAttempt,
) -> tuple[dict[str, Any], ...]:
    failures = _failed_evidence(
        attempt.structural_evidence
    )

    allowed_names = (
        _MISSING_FILE_EVIDENCE
        | _DTYPE_EVIDENCE
        | _SCHEMA_EVIDENCE
    )

    items: list[dict[str, Any]] = []

    for failure in failures:
        if failure.name not in allowed_names:
            continue

        items.append(
            {
                "name": failure.name,
                "details": (
                    _safe_structural_details(
                        failure
                    )
                ),
            }
        )

    return tuple(items)


def _build_schema_delta(
    attempt: CandidateAttempt,
) -> dict[str, Any]:
    failures = _failed_evidence(
        attempt.structural_evidence
    )

    delta: dict[str, Any] = {}

    for item in failures:
        details = _safe_structural_details(
            item
        )

        if item.name == "required_outputs":
            delta["missing_files"] = (
                details.get(
                    "missing",
                    [],
                )
            )

        elif item.name == "required_columns":
            delta["missing_columns"] = (
                details.get(
                    "missing_columns",
                    [],
                )
            )

            delta["actual_columns"] = (
                details.get(
                    "actual_columns",
                    [],
                )
            )

        elif item.name == "row_count":
            delta["expected_rows"] = (
                details.get("expected")
            )

            delta["actual_rows"] = (
                details.get("actual")
            )

        elif item.name == "null_values":
            delta["null_columns"] = (
                details.get(
                    "columns",
                    [],
                )
            )

        elif (
            item.name
            == "finite_numeric_values"
        ):
            delta[
                "nonfinite_columns"
            ] = details.get(
                "columns",
                [],
            )

        elif (
            item.name
            == "duplicate_identifiers"
        ):
            delta[
                "duplicate_identifier_count"
            ] = details.get(
                "duplicate_count"
            )

        elif item.name == "identifier_set":
            delta[
                "identifier_set_mismatch"
            ] = True

        elif item.name == "identifier_order":
            delta[
                "identifier_order_mismatch"
            ] = True

        elif item.name in _DTYPE_EVIDENCE:
            delta[
                "dtype_mismatch"
            ] = details

    return delta


def _build_relevant_invariants(
    attempt: CandidateAttempt,
) -> tuple[dict[str, Any], ...]:
    failures = _failed_evidence(
        attempt.quant_evidence
    )

    items: list[dict[str, Any]] = []

    for failure in failures:
        items.append(
            {
                "name": _safe_value(
                    failure.name
                ),
                "message": _safe_value(
                    failure.message
                ),
                "details": (
                    _safe_quant_details(
                        failure
                    )
                ),
            }
        )

    return tuple(items)


def build_repair_brief(
    *,
    attempt: CandidateAttempt,
    budget: RepairBudget,
    validation_error: str | None = None,
    estimated_tokens: int = 0,
    estimated_wall_seconds: float = 0.0,
) -> RepairBrief | None:
    failure_label = classify_failure(
        attempt,
        validation_error=validation_error,
    )

    if failure_label is None:
        return None

    strategy = _STRATEGIES[
        failure_label
    ]

    safe_traceback = ""

    if failure_label in {
        FailureLabel.SYNTAX,
        FailureLabel.IMPORT,
        FailureLabel.RUNTIME,
    }:
        traceback_source = (
            validation_error
            or attempt.stderr
            or ""
        )

        safe_traceback = (
            sanitize_public_traceback(
                traceback_source
            )
        )

    violated_contract = (
        _build_violated_contract(
            attempt
        )
    )

    schema_delta = (
        _build_schema_delta(
            attempt
        )
    )

    relevant_invariants = (
        _build_relevant_invariants(
            attempt
        )
    )

    repair_attempt_number = (
        budget.reserve_attempt(
            estimated_tokens=(
                estimated_tokens
            ),
            estimated_wall_seconds=(
                estimated_wall_seconds
            ),
        )
    )

    return RepairBrief(
        failure_label=(
            failure_label.value
        ),
        strategy=strategy.name,
        objective=strategy.objective,
        safe_traceback_excerpt=(
            safe_traceback
        ),
        violated_contract=(
            violated_contract
        ),
        schema_delta=schema_delta,
        relevant_invariants=(
            relevant_invariants
        ),
        instructions=(
            strategy.instructions
        ),
        source_attempt_number=(
            attempt.attempt_number
        ),
        repair_attempt_number=(
            repair_attempt_number
        ),
        budget=budget.snapshot(),
    )