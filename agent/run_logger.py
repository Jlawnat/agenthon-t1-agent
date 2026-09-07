from __future__ import annotations

import json
import os
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def utc_now_iso() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def create_run_dir(
    base_dir: Path,
    task_id: str | None,
) -> Path:
    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    safe_task_id = (
        task_id
        or "unknown-task"
    )

    run_dir = (
        base_dir
        / (
            f"{timestamp}-"
            f"{safe_task_id}"
        )
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return run_dir


def _json_text(
    payload: Any,
    *,
    indent: int | None,
) -> str:
    return json.dumps(
        payload,
        indent=indent,
        ensure_ascii=False,
        default=str,
        sort_keys=True,
    )


def save_json(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    """
    Atomically write JSON.

    The old checkpoint remains intact until the new
    complete temporary file is fsynced and renamed.
    """

    destination = (
        destination.resolve()
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=(
                f".{destination.name}."
            ),
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(
                handle.name
            )

            handle.write(
                _json_text(
                    payload,
                    indent=2,
                )
            )

            handle.write(
                "\n"
            )

            handle.flush()

            os.fsync(
                handle.fileno()
            )

        os.replace(
            temporary_path,
            destination,
        )

    finally:
        if (
            temporary_path
            is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def load_json(
    source: Path,
) -> dict[str, Any]:
    source = (
        source.resolve()
    )

    text = source.read_text(
        encoding="utf-8"
    )

    value = json.loads(
        text
    )

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            "Expected JSON object "
            f"in {source}."
        )

    return value


def append_jsonl(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    """
    Append one structured event to a JSONL journal.

    The file is flushed and fsynced after each event so
    completed events survive an abrupt process exit.
    """

    destination = (
        destination.resolve()
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    line = (
        _json_text(
            payload,
            indent=None,
        )
        + "\n"
    )

    with destination.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            line
        )

        handle.flush()

        os.fsync(
            handle.fileno()
        )