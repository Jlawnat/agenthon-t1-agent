from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import shutil
import stat

from agent.audit_pipeline import (
    CleanRoomAuditResult,
)
from agent.orchestrator_state import (
    OrchestratorState,
    PipelineStage,
)


class PublishError(RuntimeError):
    """Raised when audited output cannot be safely published."""


@dataclass(frozen=True)
class PublishResult:
    final_output_dir: Path
    published_files: tuple[str, ...]
    replaced_existing_output: bool


def _run_token(
    run_id: str,
) -> str:
    return sha256(
        run_id.encode("utf-8")
    ).hexdigest()[:12]


_FORBIDDEN_PUBLISH_BASENAMES = {
    "reward.json",
    "reward.txt",
    "pytest_report.json",
    "test_outputs.py",
    "test.sh",
}


def _safe_relative_path(
    raw_path: str,
) -> Path:
    text = str(raw_path).replace("\\", "/").strip()

    if not text or "\x00" in text:
        raise PublishError(
            "Publish inventory contains an empty or invalid path."
        )

    pure = PurePosixPath(text)

    if pure.is_absolute():
        raise PublishError(
            "Publish inventory contains an absolute path."
        )

    parts = pure.parts

    if (
        not parts
        or any(
            part in {"", ".", ".."}
            for part in parts
        )
    ):
        raise PublishError(
            "Publish inventory contains path traversal "
            "or an invalid component."
        )

    if pure.name.lower() in _FORBIDDEN_PUBLISH_BASENAMES:
        raise PublishError(
            "Publish inventory contains a verifier-owned artifact."
        )

    return Path(*parts)


def _lstat_regular_file(
    path: Path,
) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False

    return stat.S_ISREG(mode)


def _assert_no_symlink_ancestors(
    *,
    root: Path,
    relative: Path,
) -> None:
    current = root

    for part in relative.parts:
        current = current / part

        if current.is_symlink():
            raise PublishError(
                "Audited output contains a symbolic-link "
                f"path component: {relative.as_posix()}"
            )


def _inventory(
    root: Path,
) -> tuple[str, ...]:
    if not root.exists():
        return ()

    if root.is_symlink():
        raise PublishError(
            "Publish inventory root must not be a symbolic link."
        )

    if not root.is_dir():
        raise PublishError(
            "Publish inventory root must be a directory."
        )

    files: list[str] = []

    for current_root, dir_names, file_names in os.walk(
        root,
        followlinks=False,
    ):
        current = Path(current_root)

        for name in list(dir_names):
            child = current / name

            if child.is_symlink():
                raise PublishError(
                    "Publish inventory contains "
                    "a symbolic-link directory."
                )

        for name in file_names:
            path = current / name

            if path.is_symlink():
                raise PublishError(
                    "Publish inventory contains "
                    "a symbolic-link file."
                )

            if not _lstat_regular_file(path):
                raise PublishError(
                    "Publish inventory contains a non-regular file."
                )

            relative = path.relative_to(root).as_posix()
            _safe_relative_path(relative)
            files.append(relative)

    return tuple(sorted(files))


def _remove_directory(
    path: Path,
) -> None:
    if not path.exists():
        return

    if path.is_symlink():
        path.unlink()
        return

    shutil.rmtree(path)


def _copy_audited_files(
    *,
    source_dir: Path,
    staging_dir: Path,
    required_files: tuple[str, ...],
) -> None:
    if source_dir.is_symlink():
        raise PublishError(
            "Audited output directory must not be a symbolic link."
        )

    source_root = source_dir.resolve()

    staging_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    for raw_relative in required_files:
        relative = _safe_relative_path(raw_relative)

        _assert_no_symlink_ancestors(
            root=source_dir,
            relative=relative,
        )

        source = source_dir / relative

        if (
            not source.exists()
            or not _lstat_regular_file(source)
        ):
            raise PublishError(
                "Audited output is missing a required regular file: "
                f"{relative.as_posix()}"
            )

        try:
            resolved_source = source.resolve(strict=True)
            resolved_source.relative_to(source_root)
        except (OSError, ValueError) as exc:
            raise PublishError(
                "Audited output file escapes the audited directory."
            ) from exc

        destination = staging_dir / relative

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
            follow_symlinks=False,
        )

        if not _lstat_regular_file(destination):
            raise PublishError(
                "Staged deliverable is not a regular file."
            )


def run_publish_stage(
    *,
    state: OrchestratorState,
    audit_result: CleanRoomAuditResult,
    final_output_dir: Path,
    allow_quant_only_failures: bool = False,
) -> PublishResult:
    """
    Atomically promote only the successfully audited
    clean-room output.

    Staging and backup directories are siblings of the
    final output directory so the final rename remains
    on the same filesystem.

    Ordinary failures attempt rollback immediately.
    Process-crash recovery is handled by Phase 4.10.
    """

    stage = PipelineStage.PUBLISH

    state.start_stage(
        stage
    )

    raw_final_output_dir = Path(final_output_dir)

    if raw_final_output_dir.is_symlink():
        reason = (
            "Final output directory must not be a symbolic link."
        )
        state.fail_stage(stage, reason=reason)
        raise PublishError(reason)

    if (
        raw_final_output_dir.exists()
        and not raw_final_output_dir.is_dir()
    ):
        reason = (
            "Final output path exists but is not a directory."
        )
        state.fail_stage(stage, reason=reason)
        raise PublishError(reason)

    final_output_dir = raw_final_output_dir.resolve()

    raw_source_dir = Path(
        audit_result.audited_output_dir
    )

    if raw_source_dir.is_symlink():
        reason = (
            "Audited clean-room output directory "
            "must not be a symbolic link."
        )
        state.fail_stage(stage, reason=reason)
        raise PublishError(reason)

    source_dir = raw_source_dir.resolve()

    if (
        not audit_result.audit.passed
        and not (
            allow_quant_only_failures
            and audit_result.audit.publishable
        )
    ):
        reason = (
            "Refusing to publish output "
            "that did not pass final audit."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise PublishError(
            reason
        )

    if (
        not source_dir.exists()
        or not source_dir.is_dir()
    ):
        reason = (
            "Audited clean-room output "
            "directory does not exist."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise PublishError(
            reason
        )

    required_files = tuple(
        sorted(
            audit_result
            .audit
            .required_files
        )
    )

    produced_files = tuple(
        sorted(
            audit_result
            .audit
            .produced_files
        )
    )

    if not required_files:
        reason = (
            "Final audit contains no "
            "publishable deliverables."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise PublishError(
            reason
        )

    if produced_files != required_files:
        reason = (
            "Audited output inventory "
            "does not exactly match the "
            "required deliverables."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise PublishError(
            reason
        )

    # The official Agenthon runtime mounts /output and /app/output
    # as writable directories while the parent filesystem (for example
    # /app) is read-only.  Therefore staging cannot be created as a
    # sibling of final_output_dir.  Keep all transient publication state
    # inside the writable output mount itself.
    #
    # This preserves atomic replacement at the individual-file level and
    # supports rollback, while avoiding writes to final_output_dir.parent.
    final_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    token = _run_token(
        state.run_id
    )

    staging_dir = (
        final_output_dir
        / f".publish-staging.{token}"
    )

    backup_dir = (
        final_output_dir
        / f".publish-backup.{token}"
    )

    # Phase 4.10 and the existing regression contract may leave a
    # sibling backup from an interrupted run created by an older
    # publisher.  We must detect it, but we never create new sibling
    # state because the parent (for example /app) is read-only in the
    # official runtime.
    legacy_backup_dir = (
        final_output_dir.parent
        / (
            f".{final_output_dir.name}"
            f".backup.{token}"
        )
    )

    if (
        backup_dir.exists()
        or legacy_backup_dir.exists()
    ):
        reason = (
            "Publish backup already exists; "
            "crash recovery is required "
            "before promotion can continue."
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        raise PublishError(
            reason
        )

    if staging_dir.exists():
        _remove_directory(
            staging_dir
        )

    # Treat publication as replacement of the complete output tree.
    # Existing deliverables (including stale files from a prior run) are
    # moved into an internal backup first, so the committed inventory can
    # exactly match required_files while rollback can still restore the
    # previous output.
    existing_entries = tuple(
        path
        for path in final_output_dir.iterdir()
        if path not in {
            staging_dir,
            backup_dir,
        }
    )

    replaced_existing = bool(
        existing_entries
    )

    moved_backups: list[tuple[Path, Path]] = []
    committed_paths: list[Path] = []

    try:
        # -------------------------------------------------
        # 1. Build a complete staging tree inside the writable mount.
        # -------------------------------------------------
        _copy_audited_files(
            source_dir=source_dir,
            staging_dir=staging_dir,
            required_files=(
                required_files
            ),
        )

        staging_inventory = (
            _inventory(
                staging_dir
            )
        )

        if (
            staging_inventory
            != required_files
        ):
            raise PublishError(
                "Staging output inventory "
                "does not exactly match "
                "the audited deliverables."
            )

        backup_dir.mkdir(
            parents=False,
            exist_ok=False,
        )

        # -------------------------------------------------
        # 2. Move the complete previous output aside.
        #    Keep the staging and backup directories in place.
        # -------------------------------------------------
        for existing in existing_entries:
            backup = (
                backup_dir
                / existing.name
            )

            os.replace(
                existing,
                backup,
            )

            moved_backups.append(
                (backup, existing)
            )

        # -------------------------------------------------
        # 3. Promote each required staged file atomically.
        # -------------------------------------------------
        for raw_relative in required_files:
            relative = _safe_relative_path(
                raw_relative
            )

            staged = staging_dir / relative
            destination = final_output_dir / relative

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            os.replace(
                staged,
                destination,
            )

            committed_paths.append(
                destination
            )

        # -------------------------------------------------
        # 4. Remove transient trees before validating the
        #    committed output inventory.
        # -------------------------------------------------
        if staging_dir.exists():
            _remove_directory(
                staging_dir
            )

        if backup_dir.exists():
            _remove_directory(
                backup_dir
            )

        committed_inventory = (
            _inventory(
                final_output_dir
            )
        )

        if (
            committed_inventory
            != required_files
        ):
            raise PublishError(
                "Committed output inventory "
                "does not exactly match "
                "the audited deliverables."
            )

    except Exception as exc:
        # -------------------------------------------------
        # Best-effort synchronous rollback.
        # -------------------------------------------------
        try:
            # Remove newly committed files first.  Clean up empty
            # directories afterwards so restored prior directories can
            # be moved back into their original locations.
            for destination in reversed(
                committed_paths
            ):
                if (
                    destination.exists()
                    and _lstat_regular_file(
                        destination
                    )
                ):
                    destination.unlink()

            # Remove empty parent directories created for nested
            # deliverables, without ever removing final_output_dir.
            for destination in reversed(
                committed_paths
            ):
                parent = destination.parent

                while (
                    parent != final_output_dir
                    and parent.exists()
                ):
                    try:
                        parent.rmdir()
                    except OSError:
                        break

                    parent = parent.parent

            for backup, original in reversed(
                moved_backups
            ):
                if backup.exists():
                    os.replace(
                        backup,
                        original,
                    )

        finally:
            if staging_dir.exists():
                _remove_directory(
                    staging_dir
                )

            if backup_dir.exists():
                _remove_directory(
                    backup_dir
                )

        reason = (
            "Atomic output publication "
            "failed: "
            f"{type(exc).__name__}: {exc}"
        )

        state.fail_stage(
            stage,
            reason=reason,
        )

        if isinstance(
            exc,
            PublishError,
        ):
            raise PublishError(
                reason
            ) from exc

        raise PublishError(
            reason
        ) from exc

    state.complete_stage(
        stage,
        detail=(
            f"published "
            f"{len(required_files)} "
            "audited deliverable(s)"
        ),
    )

    return PublishResult(
        final_output_dir=(
            final_output_dir
        ),
        published_files=(
            required_files
        ),
        replaced_existing_output=(
            replaced_existing
        ),
    )