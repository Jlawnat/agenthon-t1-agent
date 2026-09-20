from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import zipfile

import pandas as pd


MAX_SAMPLE_ROWS = 5000
MAX_TEXT_CHARS = 20_000
MAX_JSON_SAMPLE_KEYS = 50
MAX_JSON_STRING_CHARS = 500
MAX_ARCHIVE_MEMBERS = 200
MAX_ARCHIVE_PREVIEW_MEMBERS = 6
MAX_ARCHIVE_MEMBER_PREVIEW_CHARS = 4000
MAX_ARCHIVE_TOTAL_PREVIEW_CHARS = 12_000
MAX_ARCHIVE_PREVIEW_MEMBER_BYTES = 512_000

_TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".xml",
    ".xbrl",
    ".xsd",
    ".html",
    ".htm",
    ".py",
}


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
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        total_rows = parquet_file.metadata.num_rows
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


def inspect_delimited(
    path: Path,
    *,
    sep: str,
) -> dict[str, Any]:
    df = pd.read_csv(
        path,
        sep=sep,
        nrows=MAX_SAMPLE_ROWS,
        low_memory=False,
    )

    result = inspect_dataframe(
        df,
        total_rows=None,
        sampled=True,
    )
    result["delimiter"] = sep
    return result


def inspect_csv(
    path: Path,
) -> dict[str, Any]:
    return inspect_delimited(
        path,
        sep=",",
    )


def inspect_tsv(
    path: Path,
) -> dict[str, Any]:
    return inspect_delimited(
        path,
        sep="\t",
    )


def _json_sample_value(
    value: Any,
) -> Any:
    if value is None or isinstance(
        value,
        (bool, int, float),
    ):
        return value

    if isinstance(value, str):
        if len(value) <= MAX_JSON_STRING_CHARS:
            return value

        return (
            value[:MAX_JSON_STRING_CHARS]
            + "...[truncated]"
        )

    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "sample": [
                _json_sample_value(item)
                for item in value[:3]
            ],
        }

    if isinstance(value, dict):
        return {
            "type": "object",
            "keys": [
                str(key)
                for key in list(value)[:20]
            ],
        }

    return {
        "type": type(value).__name__,
    }


def inspect_json(
    path: Path,
) -> dict[str, Any]:
    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )
    except Exception:
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

    if isinstance(data, dict):
        keys = list(data.keys())
        sample_keys = keys[:MAX_JSON_SAMPLE_KEYS]

        return {
            "type": "json_object",
            "keys": sample_keys,
            "sample_values": {
                str(key): _json_sample_value(
                    data[key]
                )
                for key in sample_keys
            },
            "truncated_keys": (
                len(keys)
                > MAX_JSON_SAMPLE_KEYS
            ),
        }

    if isinstance(data, list):
        return {
            "type": "json_list",
            "length": len(data),
            "sample": [
                _json_sample_value(item)
                for item in data[:3]
            ],
        }

    return {
        "type": type(data).__name__,
        "value": _json_sample_value(data),
    }


def inspect_jsonl(
    path: Path,
) -> dict[str, Any]:
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
    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        preview = handle.read(
            MAX_TEXT_CHARS + 1
        )

    truncated = (
        len(preview)
        > MAX_TEXT_CHARS
    )

    if truncated:
        preview = preview[
            :MAX_TEXT_CHARS
        ]

    return {
        "chars_at_least": len(preview),
        "truncated": truncated,
        "preview": preview,
    }


def inspect_zip(
    path: Path,
) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    previews: dict[str, str] = {}
    extension_counts: dict[str, int] = {}
    total_uncompressed = 0
    total_compressed = 0
    preview_chars = 0

    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()

        for info in infos:
            if info.is_dir():
                continue

            suffix = Path(info.filename).suffix.lower()
            extension_counts[suffix or "<none>"] = (
                extension_counts.get(
                    suffix or "<none>",
                    0,
                )
                + 1
            )

            total_uncompressed += int(
                info.file_size
            )
            total_compressed += int(
                info.compress_size
            )

            if len(members) < MAX_ARCHIVE_MEMBERS:
                members.append(
                    {
                        "name": info.filename,
                        "size": int(
                            info.file_size
                        ),
                        "compressed_size": int(
                            info.compress_size
                        ),
                    }
                )

            if (
                len(previews)
                >= MAX_ARCHIVE_PREVIEW_MEMBERS
                or preview_chars
                >= MAX_ARCHIVE_TOTAL_PREVIEW_CHARS
                or suffix not in _TEXT_SUFFIXES
                or info.file_size
                > MAX_ARCHIVE_PREVIEW_MEMBER_BYTES
                or info.flag_bits & 0x1
            ):
                continue

            remaining = (
                MAX_ARCHIVE_TOTAL_PREVIEW_CHARS
                - preview_chars
            )
            limit = min(
                MAX_ARCHIVE_MEMBER_PREVIEW_CHARS,
                remaining,
            )

            if limit <= 0:
                continue

            try:
                with archive.open(
                    info,
                    "r",
                ) as member:
                    raw = member.read(
                        limit * 4 + 4
                    )
            except Exception:
                continue

            text = raw.decode(
                "utf-8",
                errors="replace",
            )[:limit]

            previews[
                info.filename
            ] = text
            preview_chars += len(text)

    return {
        "type": "zip_archive",
        "member_count": len(infos),
        "members": members,
        "members_truncated": (
            len(infos)
            > MAX_ARCHIVE_MEMBERS
        ),
        "extension_counts": (
            extension_counts
        ),
        "total_uncompressed_bytes": (
            total_uncompressed
        ),
        "total_compressed_bytes": (
            total_compressed
        ),
        "text_previews": previews,
    }


def inspect_data_file(
    path: Path,
) -> dict[str, Any]:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return inspect_parquet(path)

    if suffix == ".csv":
        return inspect_csv(path)

    if suffix == ".tsv":
        return inspect_tsv(path)

    if suffix == ".json":
        return inspect_json(path)

    if suffix == ".jsonl":
        return inspect_jsonl(path)

    if suffix == ".zip":
        return inspect_zip(path)

    if suffix in {
        ".txt",
        ".md",
        ".html",
        ".htm",
        ".xml",
        ".xbrl",
        ".xsd",
        ".py",
    }:
        return inspect_text(path)

    return {
        "type": "unsupported",
        "suffix": suffix,
    }
