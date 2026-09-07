from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent.candidate_contract import VerificationEvidence


def _json_to_frame(
    payload: Any,
) -> pd.DataFrame:
    if isinstance(payload, list):
        return pd.DataFrame(payload)

    if isinstance(payload, dict):
        values = list(
            payload.values()
        )

        if (
            values
            and all(
                isinstance(value, dict)
                for value in values
            )
        ):
            key_sets = [
                set(value)
                for value in values
            ]

            if all(
                keys == key_sets[0]
                for keys in key_sets[1:]
            ):
                frame = pd.DataFrame(
                    payload
                )

                frame.insert(
                    0,
                    "key",
                    frame.index.astype(str),
                )

                return frame.reset_index(
                    drop=True
                )

        if (
            values
            and all(
                isinstance(value, list)
                for value in values
            )
        ):
            lengths = {
                len(value)
                for value in values
            }

            if len(lengths) == 1:
                return pd.DataFrame(
                    payload
                )

        return pd.DataFrame(
            [payload]
        )

    return pd.DataFrame(
        {
            "value": [payload]
        }
    )


def _load_output(
    path: Path,
) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".jsonl":
        return pd.read_json(
            path,
            lines=True,
        )

    if suffix == ".json":
        payload = json.loads(
            path.read_text(
                encoding="utf-8",
                errors="strict",
            )
        )

        return _json_to_frame(
            payload
        )

    raise ValueError(
        "Unsupported output format for "
        f"structural verification: {suffix}"
    )


def _finite_numeric_check(
    df: pd.DataFrame,
) -> list[str]:
    bad_columns: list[str] = []

    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        values = df[column].to_numpy()

        if not np.isfinite(values).all():
            bad_columns.append(str(column))

    return bad_columns


def _dtype_matches(
    series: pd.Series,
    expected_dtype: str,
) -> bool:
    expected = (
        str(expected_dtype)
        .strip()
        .lower()
        .replace(" ", "")
    )

    actual = str(series.dtype).strip().lower()

    if expected in {"str", "string", "text"}:
        if pd.api.types.is_object_dtype(
            series.dtype
        ):
            non_null = series.dropna()

            return bool(
                non_null.map(
                    lambda value: isinstance(
                        value,
                        str,
                    )
                ).all()
            )

        return pd.api.types.is_string_dtype(
            series.dtype
        )

    if expected in {"int", "integer"}:
        return pd.api.types.is_integer_dtype(
            series.dtype
        )

    if expected in {"float", "floating"}:
        return pd.api.types.is_float_dtype(
            series.dtype
        )

    if expected in {"number", "numeric"}:
        return pd.api.types.is_numeric_dtype(
            series.dtype
        )

    if expected in {"bool", "boolean"}:
        return pd.api.types.is_bool_dtype(
            series.dtype
        )

    if expected in {"datetime", "datetime64"}:
        return pd.api.types.is_datetime64_any_dtype(
            series.dtype
        )

    if expected == "object":
        return pd.api.types.is_object_dtype(
            series.dtype
        )

    return actual == expected

def _dtype_mismatches(
    df: pd.DataFrame,
    expected_dtypes: dict[str, str],
) -> list[dict[str, str]]:
    mismatches: list[dict[str, str]] = []

    for column, expected_dtype in expected_dtypes.items():
        if column not in df.columns:
            continue

        series = df[column]

        if _dtype_matches(
            series,
            expected_dtype,
        ):
            continue

        mismatches.append(
            {
                "column": str(column),
                "expected_dtype": str(expected_dtype),
                "actual_dtype": str(series.dtype),
            }
        )

    return mismatches


def verify_structural_output(
    *,
    output_path: Path,
    required_columns: list[str],
    expected_rows: int | None = None,
    id_column: str | None = None,
    expected_ids: list[Any] | None = None,
    require_id_order: bool = False,
    expected_dtypes: dict[str, str] | None = None,
) -> list[VerificationEvidence]:

    evidence: list[VerificationEvidence] = []

    if not output_path.exists():
        return [
            VerificationEvidence(
                name="output_exists",
                status="fail",
                severity="hard_fail",
                message=f"Output file does not exist: {output_path}",
            )
        ]

    try:
        df = _load_output(output_path)

    except Exception as exc:
        return [
            VerificationEvidence(
                name="output_readable",
                status="fail",
                severity="hard_fail",
                message="Output file could not be read.",
                details={
                    "error": str(exc),
                },
            )
        ]

    evidence.append(
        VerificationEvidence(
            name="output_readable",
            status="pass",
            severity="info",
            message="Output file loaded successfully.",
        )
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        evidence.append(
            VerificationEvidence(
                name="required_columns",
                status="fail",
                severity="hard_fail",
                message="Required columns are missing.",
                details={
                    "missing_columns": missing_columns,
                    "actual_columns": [
                        str(column)
                        for column in df.columns
                    ],
                },
            )
        )
    else:
        evidence.append(
            VerificationEvidence(
                name="required_columns",
                status="pass",
                severity="info",
                message="All required columns are present.",
            )
        )

    dtype_expectations = dict(
        expected_dtypes or {}
    )

    if dtype_expectations:
        mismatches = _dtype_mismatches(
            df,
            dtype_expectations,
        )

        if mismatches:
            evidence.append(
                VerificationEvidence(
                    name="dtype_mismatch",
                    status="fail",
                    severity="hard_fail",
                    message=(
                        "Output column dtypes do not match "
                        "the explicit schema."
                    ),
                    details={
                        "mismatches": mismatches,
                    },
                )
            )
        else:
            evidence.append(
                VerificationEvidence(
                    name="required_dtypes",
                    status="pass",
                    severity="info",
                    message=(
                        "Output column dtypes match "
                        "the explicit schema."
                    ),
                )
            )

    if expected_rows is not None:
        if len(df) != expected_rows:
            evidence.append(
                VerificationEvidence(
                    name="row_count",
                    status="fail",
                    severity="hard_fail",
                    message="Output row count does not match expected row count.",
                    details={
                        "expected": expected_rows,
                        "actual": len(df),
                    },
                )
            )
        else:
            evidence.append(
                VerificationEvidence(
                    name="row_count",
                    status="pass",
                    severity="info",
                    message="Output row count matches expected row count.",
                )
            )

    null_counts = df.isna().sum()
    bad_null_columns = [
        str(column)
        for column, count in null_counts.items()
        if int(count) > 0
    ]

    if bad_null_columns:
        evidence.append(
            VerificationEvidence(
                name="null_values",
                status="fail",
                severity="hard_fail",
                message="Output contains null values.",
                details={
                    "columns": bad_null_columns,
                },
            )
        )
    else:
        evidence.append(
            VerificationEvidence(
                name="null_values",
                status="pass",
                severity="info",
                message="Output contains no null values.",
            )
        )

    bad_numeric_columns = _finite_numeric_check(df)

    if bad_numeric_columns:
        evidence.append(
            VerificationEvidence(
                name="finite_numeric_values",
                status="fail",
                severity="hard_fail",
                message="Numeric output contains NaN or Inf.",
                details={
                    "columns": bad_numeric_columns,
                },
            )
        )
    else:
        evidence.append(
            VerificationEvidence(
                name="finite_numeric_values",
                status="pass",
                severity="info",
                message="All numeric values are finite.",
            )
        )

    if id_column is not None:
        if id_column not in df.columns:
            evidence.append(
                VerificationEvidence(
                    name="identifier_column",
                    status="fail",
                    severity="hard_fail",
                    message=f"Identifier column is missing: {id_column}",
                )
            )

        else:
            duplicate_count = int(
                df[id_column].duplicated().sum()
            )

            if duplicate_count > 0:
                evidence.append(
                    VerificationEvidence(
                        name="duplicate_identifiers",
                        status="fail",
                        severity="hard_fail",
                        message="Duplicate identifiers found.",
                        details={
                            "duplicate_count": duplicate_count,
                        },
                    )
                )
            else:
                evidence.append(
                    VerificationEvidence(
                        name="duplicate_identifiers",
                        status="pass",
                        severity="info",
                        message="Identifiers are unique.",
                    )
                )

            if expected_ids is not None:
                actual_ids = df[id_column].tolist()

                if set(actual_ids) != set(expected_ids):
                    evidence.append(
                        VerificationEvidence(
                            name="identifier_set",
                            status="fail",
                            severity="hard_fail",
                            message="Output identifiers do not match expected identifiers.",
                        )
                    )
                else:
                    evidence.append(
                        VerificationEvidence(
                            name="identifier_set",
                            status="pass",
                            severity="info",
                            message="Output identifier set matches expected identifiers.",
                        )
                    )

                if (
                    require_id_order
                    and actual_ids != expected_ids
                ):
                    evidence.append(
                        VerificationEvidence(
                            name="identifier_order",
                            status="fail",
                            severity="hard_fail",
                            message="Output identifier ordering does not match expected ordering.",
                        )
                    )

                elif require_id_order:
                    evidence.append(
                        VerificationEvidence(
                            name="identifier_order",
                            status="pass",
                            severity="info",
                            message="Output identifier ordering is correct.",
                        )
                    )

    return evidence
