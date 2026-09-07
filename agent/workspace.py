from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile


@dataclass
class Workspace:
    task_dir: Path
    out_dir: Path
    work_dir: Path

    @classmethod
    def create(cls, task_dir: Path, out_dir: Path) -> "Workspace":
        task_dir = task_dir.resolve()
        out_dir = out_dir.resolve()

        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory does not exist: {task_dir}")

        if not task_dir.is_dir():
            raise NotADirectoryError(f"Task path is not a directory: {task_dir}")

        out_dir.mkdir(parents=True, exist_ok=True)

        work_dir = Path(
            tempfile.mkdtemp(
                prefix="quantagent-",
                dir="/tmp",
            )
        )

        return cls(
            task_dir=task_dir,
            out_dir=out_dir,
            work_dir=work_dir,
        )

    def discover_task_files(self) -> list[Path]:
        files: list[Path] = []

        for path in self.task_dir.rglob("*"):
            if path.is_file():
                files.append(path.relative_to(self.task_dir))

        return sorted(files)

    def cleanup(self) -> None:
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)