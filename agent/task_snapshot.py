from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.data_inspector import inspect_data_file


@dataclass
class TaskSnapshot:
    task_dir: Path
    instruction_path: Path | None
    card_path: Path | None
    data_files: list[Path]
    other_files: list[Path]

    @classmethod
    def build(cls, task_dir: Path) -> "TaskSnapshot":
        task_dir = task_dir.resolve()

        instruction_path = task_dir / "instruction.md"
        if not instruction_path.exists():
            instruction_path = None

        card_path = task_dir / "card.toml"
        if not card_path.exists():
            card_path = None

        data_files: list[Path] = []
        other_files: list[Path] = []

        for path in task_dir.rglob("*"):
            if not path.is_file():
                continue

            rel = path.relative_to(task_dir)

            metadata_names = {
                "manifest.json",
                "instruction.md",
                "card.toml",
            }
            if rel.name in metadata_names:
                other_files.append(rel)
            elif "checks" in rel.parts:
                other_files.append(rel)
            elif (
                "environment" in rel.parts
                and "data" in rel.parts
            ):
                data_files.append(rel)
            elif path.suffix.lower() in {
                ".parquet",
                ".csv",
                ".json",
                ".jsonl",
                ".txt",
                ".md",
            }:
                data_files.append(rel)
            else:
                other_files.append(rel)
        return cls(
            task_dir=task_dir,
            instruction_path=instruction_path,
            card_path=card_path,
            data_files=sorted(data_files),
            other_files=sorted(other_files),
        )

    def print_summary(self) -> None:
        print("[QuantAgent] Task snapshot")

        print(
            f"  instruction: "
            f"{self.instruction_path.name if self.instruction_path else 'NOT FOUND'}"
        )

        print(
            f"  card.toml: "
            f"{'FOUND' if self.card_path else 'NOT FOUND'}"
        )

        print(f"  data files: {len(self.data_files)}")

        for path in self.data_files:
            print(f"    - {path}")

    def inspect_data(self) -> dict[str, dict[str, Any]]:
        inspections: dict[str, dict[str, Any]] = {}

        for rel_path in self.data_files:
            full_path = self.task_dir / rel_path

            try:
                inspections[str(rel_path)] = inspect_data_file(full_path)

            except Exception as exc:
                inspections[str(rel_path)] = {
                    "error": str(exc),
                }

        return inspections