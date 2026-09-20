from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shlex
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
    """Remove write bits from a copied task-data tree."""
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


def _task_data_is_effectively_read_only(
    root: Path,
) -> bool:
    """
    Return True when the task-data tree is safe to expose zero-copy.

    The official Agenthon runtime mounts /input read-only. We also
    accept a tree that is not writable by the current process.
    Otherwise we fall back to the original copy-and-chmod behavior,
    preserving the candidate-workspace security invariant in local
    development and tests.
    """
    try:
        flags = os.statvfs(root).f_flag
        if flags & getattr(os, "ST_RDONLY", 1):
            return True
    except OSError:
        pass

    try:
        if os.access(root, os.W_OK):
            return False

        for current_root, dirs, files in os.walk(
            root,
            topdown=True,
            followlinks=False,
        ):
            current = Path(current_root)

            if os.access(current, os.W_OK):
                return False

            for name in list(dirs) + list(files):
                path = current / name

                if os.access(path, os.W_OK):
                    return False

    except OSError:
        return False

    return True


def _normalise_copy_source(
    value: str,
) -> Path | None:
    raw = value.strip()

    while raw.startswith("./"):
        raw = raw[2:]

    if raw in {"data", "data/"}:
        return None

    if not raw.startswith("data/"):
        return None

    rel = Path(
        raw[len("data/"):]
    )

    if (
        not rel.parts
        or rel.is_absolute()
        or ".." in rel.parts
    ):
        return None

    return rel


def _normalise_copy_destination(
    value: str,
    *,
    source_rel: Path,
) -> Path | None:
    raw = value.strip()

    if not raw.startswith("/app"):
        return None

    is_directory = raw.endswith("/")

    if raw == "/app":
        is_directory = True
        relative = ""
    elif raw == "/app/data":
        is_directory = True
        relative = ""
    elif raw.startswith("/app/data/"):
        relative = raw[len("/app/data/"):]
    elif raw.startswith("/app/"):
        relative = raw[len("/app/"):]
    else:
        return None

    if is_directory:
        prefix = (
            relative.rstrip("/") + "/"
            if relative
            else ""
        )
        relative = (
            prefix
            + source_rel.name
        )

    rel = Path(relative)

    if (
        not rel.parts
        or rel.is_absolute()
        or ".." in rel.parts
    ):
        return None

    return rel


def _legacy_copy_aliases(
    *,
    task_dir: Path,
    environment_data: Path,
) -> dict[Path, Path]:
    dockerfile = (
        task_dir
        / "environment"
        / "Dockerfile"
    )

    if not dockerfile.exists():
        return {}

    aliases: dict[Path, Path] = {}

    try:
        lines = dockerfile.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        return {}

    for raw_line in lines:
        stripped = raw_line.strip()

        if (
            not stripped
            or not stripped.upper().startswith("COPY ")
        ):
            continue

        parts = stripped.split()

        if (
            len(parts) != 3
            or parts[0].upper() != "COPY"
        ):
            continue

        source_rel = (
            _normalise_copy_source(
                parts[1]
            )
        )

        if source_rel is None:
            continue

        source = (
            environment_data
            / source_rel
        )

        if (
            not source.exists()
            or not source.is_file()
        ):
            continue

        alias_rel = (
            _normalise_copy_destination(
                parts[2],
                source_rel=source_rel,
            )
        )

        if alias_rel is None:
            continue

        aliases[alias_rel] = (
            source_rel
        )

    return aliases


def _stage_zero_copy_tree(
    *,
    environment_data: Path,
    candidate_data: Path,
) -> None:
    candidate_data.mkdir(
        parents=False,
        exist_ok=False,
    )

    for source in sorted(
        environment_data.rglob("*")
    ):
        rel = (
            source.relative_to(
                environment_data
            )
        )
        destination = (
            candidate_data
            / rel
        )

        if source.is_dir():
            destination.mkdir(
                parents=True,
                exist_ok=True,
            )
            continue

        if not source.is_file():
            raise CandidateWorkspaceError(
                "Task data contains an unsupported filesystem entry: "
                f"{rel}"
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        destination.symlink_to(
            source.resolve()
        )


def _apply_legacy_copy_aliases(
    *,
    task_dir: Path,
    environment_data: Path,
    candidate_data: Path,
    source_root: Path,
) -> None:
    aliases = (
        _legacy_copy_aliases(
            task_dir=task_dir,
            environment_data=environment_data,
        )
    )

    for alias_rel, source_rel in aliases.items():
        if alias_rel == source_rel:
            continue

        alias = (
            candidate_data
            / alias_rel
        )
        source = (
            source_root
            / source_rel
        )

        if not source.exists():
            raise CandidateWorkspaceError(
                "Dockerfile data alias source is missing: "
                f"{source_rel}"
            )

        if (
            alias.exists()
            or alias.is_symlink()
        ):
            try:
                if (
                    alias.is_symlink()
                    and alias.resolve()
                    == source.resolve()
                ):
                    continue
            except OSError:
                pass

            raise CandidateWorkspaceError(
                "Dockerfile data alias conflicts with an existing "
                f"candidate-data path: {alias_rel}"
            )

        alias.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        alias.symlink_to(
            source.resolve()
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
        library_dir = (
            root_dir
            / "lib"
        )

        input_dir.mkdir()
        output_dir.mkdir()
        source_dir.mkdir()
        library_dir.mkdir()

        primitive_source = (
            Path(__file__)
            .with_name(
                "qf_primitives.py"
            )
        )

        if not primitive_source.is_file():
            raise CandidateWorkspaceError(
                "Curated candidate primitive library is missing."
            )

        primitive_destination = (
            library_dir
            / "qf_primitives.py"
        )

        shutil.copyfile(
            primitive_source,
            primitive_destination,
        )

        primitive_destination.chmod(
            stat.S_IRUSR
            | stat.S_IRGRP
            | stat.S_IROTH
        )

        environment_data = (
            task_dir
            / "environment"
            / "data"
        )

        candidate_data = (
            input_dir
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

            if _task_data_is_effectively_read_only(
                environment_data
            ):
                _stage_zero_copy_tree(
                    environment_data=environment_data,
                    candidate_data=candidate_data,
                )

                alias_source_root = (
                    environment_data
                )
            else:
                shutil.copytree(
                    environment_data,
                    candidate_data,
                    dirs_exist_ok=False,
                    symlinks=False,
                )

                _make_input_read_only(
                    candidate_data
                )

                alias_source_root = (
                    candidate_data
                )

            _apply_legacy_copy_aliases(
                task_dir=task_dir,
                environment_data=environment_data,
                candidate_data=candidate_data,
                source_root=alias_source_root,
            )

        else:
            # Parameter-only tasks are valid Track-1 units. Keep the
            # candidate input contract stable even when no data files
            # are shipped by exposing an empty INPUT_DIR/data directory.
            candidate_data.mkdir(
                parents=False,
                exist_ok=False,
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