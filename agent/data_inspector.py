from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


MAX_SAMPLE_ROWS = 5000
MAX_TEXT_CHARS = 20_000


def _numeric_summary(
    series: pd.Series,
) -> dict[str, Any]:
    clean = series.dropna()

    if clean.empty:
        return {
            "min": None,
            "max": None,
            "mean": None,
        }

    return {
        "min": clean.min(),
        "max": clean.max(),
        "mean": clean.mean(),
    }


def _categorical_summary(
    series: pd.Series,
) -> dict[str, Any]:
    clean = series.dropna()

    counts = clean.value_counts().head(10)

    return {
        "n_unique": int(clean.nunique()),
        "top_values": counts.to_dict(),
    }


def inspect_dataframe(
    df: pd.DataFrame,
    *,
    total_rows: int | None = None,
    sampled: bool = False,
) -> dict[str, Any]:

    numeric: dict[str, Any] = {}
    categorical: dict[str, Any] = {}

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            numeric[str(column)] = (
                _numeric_summary(series)
            )
        else:
            categorical[str(column)] = (
                _categorical_summary(series)
            )

    return {
        "rows": (
            total_rows
            if total_rows is not None
            else len(df)
        ),
        "sample_rows_used": len(df),
        "sampled": sampled,
        "columns": [
            str(column)
            for column in df.columns
        ],
        "dtypes": {
            str(column): str(dtype)
            for column, dtype
            in df.dtypes.items()
        },
        "missing": {
            str(column): int(value)
            for column, value
            in df.isna().sum().items()
        },
        "numeric": numeric,
        "categorical": categorical,
        "sample_rows": (
            df.head(3)
            .to_dict(orient="records")
        ),
    }


def inspect_parquet(
    path: Path,
) -> dict[str, Any]:
    """
    Read Parquet metadata first, then inspect only a bounded
    sample of rows when possible.
    """

    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)

        total_rows = (
            parquet_file.metadata.num_rows
        )

        table = parquet_file.read_row_group(0)

        df = table.to_pandas()

        if len(df) > MAX_SAMPLE_ROWS:
            df = df.head(MAX_SAMPLE_ROWS)

        return inspect_dataframe(
            df,
            total_rows=total_rows,
            sampled=(
                total_rows > len(df)
            ),
        )

    except Exception:
        # Fallback for unusual Parquet files.
        df = pd.read_parquet(path)

        total_rows = len(df)

        if total_rows > MAX_SAMPLE_ROWS:
            df = df.head(MAX_SAMPLE_ROWS)

        return inspect_dataframe(
            df,
            total_rows=total_rows,
            sampled=(
                total_rows > len(df)
            ),
        )


def inspect_csv(
    path: Path,
) -> dict[str, Any]:
    """
    Inspect only a bounded sample instead of loading the
    entire CSV into memory.
    """

    df = pd.read_csv(
        path,
        nrows=MAX_SAMPLE_ROWS,
    )

    return inspect_dataframe(
        df,
        total_rows=None,
        sampled=True,
    )


def inspect_json(
    path: Path,
) -> dict[str, Any]:
    try:
        df = pd.read_json(path)

        total_rows = len(df)

        if total_rows > MAX_SAMPLE_ROWS:
            df = df.head(MAX_SAMPLE_ROWS)

        return inspect_dataframe(
            df,
            total_rows=total_rows,
            sampled=(
                total_rows > len(df)
            ),
        )

    except Exception:
        data = json.loads(
            path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )

        if isinstance(data, dict):
            return {
                "type": "json_object",
                "keys": list(data.keys())[:100],
            }

        if isinstance(data, list):
            return {
                "type": "json_list",
                "length": len(data),
                "sample": data[:3],
            }

        return {
            "type": type(data).__name__,
        }


def inspect_jsonl(
    path: Path,
) -> dict[str, Any]:
    """
    JSONL needs lines=True; the old implementation did not
    handle this correctly.
    """

    df = pd.read_json(
        path,
        lines=True,
        nrows=MAX_SAMPLE_ROWS,
    )

    return inspect_dataframe(
        df,
        total_rows=None,
        sampled=True,
    )


def inspect_text(
    path: Path,
) -> dict[str, Any]:
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    preview = text[:MAX_TEXT_CHARS]

    return {
        "chars": len(text),
        "lines": text.count("\n") + 1,
        "truncated": (
            len(text) > MAX_TEXT_CHARS
        ),
        "preview": preview,
    }


def inspect_data_file(
    path: Path,
) -> dict[str, Any]:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return inspect_parquet(path)

    if suffix == ".csv":
        return inspect_csv(path)

    if suffix == ".json":
        return inspect_json(path)

    if suffix == ".jsonl":
        return inspect_jsonl(path)

    if suffix in {
        ".txt",
        ".md",
    }:
        return inspect_text(path)

    return {
        "type": "unsupported",
        "suffix": suffix,
    }