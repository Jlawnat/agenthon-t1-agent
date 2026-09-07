from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class SchemaExpectations:
    expected_rows: int | None
    id_column: str | None
    expected_ids: list[Any] | None
    require_id_order: bool
    expected_dtypes: dict[str, str] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(
        f"Unsupported input table format: {suffix}"
    )


def infer_schema_expectations(
    *,
    task_dir: Path,
    candidate_schema_fields: list[str],
    required_dtypes: dict[str, str] | None = None,
) -> SchemaExpectations:
    """
    Infer structural expectations from the task input.

    This is intentionally conservative:
    if we cannot determine something reliably, we return None
    instead of guessing.
    """

    expected_dtypes = dict(
        required_dtypes or {}
    )

    data_dir = (
        task_dir
        / "environment"
        / "data"
    )

    if not data_dir.exists():
        return SchemaExpectations(
            expected_rows=None,
            id_column=None,
            expected_ids=None,
            require_id_order=False,
            expected_dtypes=expected_dtypes,
        )

    table_files = sorted(
        [
            path
            for path in data_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower()
            in {".parquet", ".csv"}
        ]
    )

    if len(table_files) != 1:
        return SchemaExpectations(
            expected_rows=None,
            id_column=None,
            expected_ids=None,
            require_id_order=False,
            expected_dtypes=expected_dtypes,
        )

    input_path = table_files[0]

    try:
        df = _load_table(input_path)
    except Exception:
        return SchemaExpectations(
            expected_rows=None,
            id_column=None,
            expected_ids=None,
            require_id_order=False,
            expected_dtypes=expected_dtypes,
        )

    expected_rows = len(df)

    id_candidates = [
        "option_id",
        "trade_id",
        "security_id",
        "instrument_id",
        "asset_id",
        "contract_id",
        "row_id",
        "id",
    ]

    id_column = None

    for name in id_candidates:
        if (
            name in df.columns
            and name in candidate_schema_fields
        ):
            id_column = name
            break

    if id_column is None:
        return SchemaExpectations(
            expected_rows=expected_rows,
            id_column=None,
            expected_ids=None,
            require_id_order=False,
            expected_dtypes=expected_dtypes,
        )

    expected_ids = df[id_column].tolist()

    return SchemaExpectations(
        expected_rows=expected_rows,
        id_column=id_column,
        expected_ids=expected_ids,
        require_id_order=True,
        expected_dtypes=expected_dtypes,
    )
