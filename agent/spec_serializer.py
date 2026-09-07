from __future__ import annotations

import json
from pathlib import Path

from agent.specification import TaskSpecification


def save_specification(
    spec: TaskSpecification,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(
        json.dumps(
            spec.to_dict(),
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )