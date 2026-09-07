from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.task_snapshot import TaskSnapshot


def build_snapshot_payload(
    snapshot: TaskSnapshot,
    data_inspections: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "task_dir": str(snapshot.task_dir),
        "instruction_path": (
            str(snapshot.instruction_path)
            if snapshot.instruction_path
            else None
        ),
        "card_path": (
            str(snapshot.card_path)
            if snapshot.card_path
            else None
        ),
        "data_files": [str(path) for path in snapshot.data_files],
        "other_files": [str(path) for path in snapshot.other_files],
        "data_inspections": data_inspections,
    }


def save_snapshot_json(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )