from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class DataArtifact:
    path: Path
    kind: str
    columns: frozenset[str] = frozenset()
    json_keys: frozenset[str] = frozenset()
    json_paths: frozenset[str] = frozenset()


def _collect_json_paths(value, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            part = str(key).strip().lower()
            path = f"{prefix}.{part}" if prefix else part
            paths.add(path)
            paths.update(_collect_json_paths(child, path))
    elif isinstance(value, list):
        for child in value[:8]:
            paths.update(_collect_json_paths(child, prefix + "[]"))
    return paths


@dataclass(frozen=True)
class TaskDataCatalog:
    artifacts: tuple[DataArtifact, ...]

    @classmethod
    def discover(cls, task_dir: Path) -> "TaskDataCatalog":
        artifacts: list[DataArtifact] = []
        for path in sorted(task_dir.rglob("*")):
            if not path.is_file() or "checks" in path.parts or "output" in path.parts:
                continue
            suffix = path.suffix.lower()
            if suffix == ".csv":
                try:
                    frame = pd.read_csv(path, nrows=5)
                except Exception:
                    continue
                artifacts.append(
                    DataArtifact(
                        path=path,
                        kind="csv",
                        columns=frozenset(str(c).strip().lower() for c in frame.columns),
                    )
                )
            elif suffix == ".json":
                try:
                    value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    continue
                keys = (
                    frozenset(str(k).strip().lower() for k in value.keys())
                    if isinstance(value, dict)
                    else frozenset()
                )
                artifacts.append(
                    DataArtifact(
                        path=path,
                        kind="json",
                        json_keys=keys,
                        json_paths=frozenset(_collect_json_paths(value)),
                    )
                )
        return cls(tuple(artifacts))

    def csv_with_columns(
        self,
        required: Iterable[str],
        *,
        min_extra_columns: int = 0,
    ) -> Path:
        req = {str(x).lower() for x in required}
        matches = [
            a
            for a in self.artifacts
            if a.kind == "csv"
            and req.issubset(a.columns)
            and len(a.columns - req) >= min_extra_columns
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one CSV matching columns {sorted(req)}; found "
                f"{[str(a.path) for a in matches]}."
            )
        return matches[0].path

    def json_with_keys(self, required: Iterable[str]) -> Path:
        req = {str(x).lower() for x in required}
        matches = [
            a
            for a in self.artifacts
            if a.kind == "json" and req.issubset(a.json_keys)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one JSON matching keys {sorted(req)}; found "
                f"{[str(a.path) for a in matches]}."
            )
        return matches[0].path

    def has_csv(self, required: Iterable[str], *, min_extra_columns: int = 0) -> bool:
        req = {str(x).lower() for x in required}
        return any(
            a.kind == "csv"
            and req.issubset(a.columns)
            and len(a.columns - req) >= min_extra_columns
            for a in self.artifacts
        )

    def has_json(self, required: Iterable[str]) -> bool:
        req = {str(x).lower() for x in required}
        return any(a.kind == "json" and req.issubset(a.json_keys) for a in self.artifacts)

    def json_with_paths(self, required: Iterable[str]) -> Path:
        req = {str(x).strip().lower() for x in required}
        matches = [
            a
            for a in self.artifacts
            if a.kind == "json" and req.issubset(a.json_paths)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one JSON matching paths {sorted(req)}; found "
                f"{[str(a.path) for a in matches]}."
            )
        return matches[0].path

    def has_json_paths(self, required: Iterable[str]) -> bool:
        req = {str(x).strip().lower() for x in required}
        return any(
            a.kind == "json" and req.issubset(a.json_paths)
            for a in self.artifacts
        )
