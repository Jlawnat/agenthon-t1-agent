from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import stat


class CandidateWorkspaceError(
    RuntimeError
):
    """Candidate workspace cannot be created safely."""


def _reject_symlinks(
    root: Path,
) -> None:
    """
    Reject symlinks anywhere in copied task data.

    Following a task-data symlink could copy files from outside
    the declared task data boundary into a candidate workspace.
    """

    if root.is_symlink():
        raise CandidateWorkspaceError(
            "Task data directory must not be a symlink."
        )

    for current_root, dirs, files in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(
            current_root
        )

        for name in (
            list(dirs)
            + list(files)
        ):
            path = current / name

            if path.is_symlink():
                raise CandidateWorkspaceError(
                    "Task data contains a symlink: "
                    f"{path.relative_to(root)}"
                )


def _make_input_read_only(
    root: Path,
) -> None:
    """
    Remove write bits from the copied input tree.

    The candidate executor additionally blocks Python-level
    writes outside output/temp/home. The official container
    read-only task mount remains the outer security boundary.
    """

    if not root.exists():
        return

    for path in sorted(
        root.rglob("*"),
        reverse=True,
    ):
        if path.is_symlink():
            raise CandidateWorkspaceError(
                "Copied input unexpectedly contains a symlink."
            )

        mode = path.stat().st_mode

        if path.is_dir():
            path.chmod(
                mode
                & ~(
                    stat.S_IWUSR
                    | stat.S_IWGRP
                    | stat.S_IWOTH
                )
                | stat.S_IXUSR
            )
        else:
            path.chmod(
                mode
                & ~(
                    stat.S_IWUSR
                    | stat.S_IWGRP
                    | stat.S_IWOTH
                )
            )

    mode = root.stat().st_mode
    root.chmod(
        mode
        & ~(
            stat.S_IWUSR
            | stat.S_IWGRP
            | stat.S_IWOTH
        )
        | stat.S_IXUSR
    )


def _is_within(
    path: Path,
    root: Path,
) -> bool:
    try:
        path.relative_to(
            root
        )
    except ValueError:
        return False

    return True


@dataclass
class CandidateWorkspace:
    candidate_id: int
    root_dir: Path
    input_dir: Path
    output_dir: Path
    source_dir: Path

    @classmethod
    def create(
        cls,
        *,
        base_dir: Path,
        candidate_id: int,
        task_dir: Path,
    ) -> "CandidateWorkspace":
        if (
            isinstance(
                candidate_id,
                bool,
            )
            or not isinstance(
                candidate_id,
                int,
            )
            or candidate_id <= 0
        ):
            raise ValueError(
                "candidate_id must be a positive integer."
            )

        base_dir = Path(
            base_dir
        ).resolve()

        task_dir = Path(
            task_dir
        ).resolve()

        if not task_dir.exists():
            raise FileNotFoundError(
                f"Task directory not found: {task_dir}"
            )

        if not task_dir.is_dir():
            raise NotADirectoryError(
                f"Task path is not a directory: {task_dir}"
            )

        base_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        root_dir = (
            base_dir
            / f"candidate_{candidate_id}"
        )

        if root_dir.is_symlink():
            raise CandidateWorkspaceError(
                "Candidate workspace root must not be a symlink."
            )

        if root_dir.exists():
            shutil.rmtree(
                root_dir
            )

        root_dir.mkdir(
            parents=False,
            exist_ok=False,
        )

        input_dir = (
            root_dir
            / "input"
        )
        output_dir = (
            root_dir
            / "output"
        )
        source_dir = (
            root_dir
            / "src"
        )

        input_dir.mkdir()
        output_dir.mkdir()
        source_dir.mkdir()

        environment_data = (
            task_dir
            / "environment"
            / "data"
        )

        if environment_data.exists():
            if not environment_data.is_dir():
                raise CandidateWorkspaceError(
                    "Task environment/data must be a directory."
                )

            _reject_symlinks(
                environment_data
            )

            copied_data = (
                input_dir
                / "data"
            )

            shutil.copytree(
                environment_data,
                copied_data,
                dirs_exist_ok=False,
                symlinks=False,
            )

            _make_input_read_only(
                copied_data
            )

        return cls(
            candidate_id=candidate_id,
            root_dir=(
                root_dir.resolve()
            ),
            input_dir=(
                input_dir.resolve()
            ),
            output_dir=(
                output_dir.resolve()
            ),
            source_dir=(
                source_dir.resolve()
            ),
        )

    def validate_script_path(
        self,
        script_path: Path,
    ) -> Path:
        raw = Path(
            script_path
        )

        if raw.is_symlink():
            raise CandidateWorkspaceError(
                "Candidate script must not be a symlink."
            )

        resolved = raw.resolve()

        if not _is_within(
            resolved,
            self.source_dir.resolve(),
        ):
            raise CandidateWorkspaceError(
                "Candidate script must remain inside source_dir."
            )

        if not resolved.exists():
            raise FileNotFoundError(
                f"Candidate script not found: {resolved}"
            )

        if not resolved.is_file():
            raise CandidateWorkspaceError(
                "Candidate script must be a regular file."
            )

        return resolved