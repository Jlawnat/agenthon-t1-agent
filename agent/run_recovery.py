from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Any

from agent.orchestrator_state import (
    OrchestratorState,
)
from agent.publish_pipeline import (
    _run_token,
)
from agent.run_context import (
    RunContext,
)
from agent.run_logger import (
    append_jsonl,
    load_json,
    save_json,
    utc_now_iso,
)


class RecoveryError(RuntimeError):
    """Raised when recovery cannot make a safe decision."""


@dataclass(frozen=True)
class PublishRecoveryResult:
    action: str

    final_output_dir: Path
    staging_dir: Path
    backup_dir: Path

    restored_backup: bool
    removed_staging: bool
    removed_backup: bool

    final_output_exists: bool


def _inventory(
    root: Path,
) -> tuple[str, ...]:
    if (
        not root.exists()
        or not root.is_dir()
    ):
        return ()

    files: list[str] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        files.append(
            path
            .relative_to(root)
            .as_posix()
        )

    return tuple(
        sorted(files)
    )


def _remove_path(
    path: Path,
) -> None:
    if not path.exists():
        return

    if path.is_symlink():
        path.unlink()
        return

    if path.is_dir():
        shutil.rmtree(
            path
        )

        return

    path.unlink()


def _current_publish_artifact_paths(
    *,
    final_output_dir: Path,
    run_id: str,
) -> tuple[Path, Path]:
    final_output_dir = final_output_dir.resolve()
    token = _run_token(run_id)

    return (
        final_output_dir / f".publish-staging.{token}",
        final_output_dir / f".publish-backup.{token}",
    )


def _legacy_publish_artifact_paths(
    *,
    final_output_dir: Path,
    run_id: str,
) -> tuple[Path, Path]:
    final_output_dir = final_output_dir.resolve()
    token = _run_token(run_id)
    parent = final_output_dir.parent

    return (
        parent / f".{final_output_dir.name}.staging.{token}",
        parent / f".{final_output_dir.name}.backup.{token}",
    )


def publish_recovery_needed(
    *,
    final_output_dir: Path,
    run_id: str,
) -> bool:
    current = _current_publish_artifact_paths(
        final_output_dir=final_output_dir,
        run_id=run_id,
    )
    legacy = _legacy_publish_artifact_paths(
        final_output_dir=final_output_dir,
        run_id=run_id,
    )

    return any(
        path.exists()
        for path in (
            *current,
            *legacy,
        )
    )


def _inventory_excluding(
    root: Path,
    *,
    excluded_top_level: set[str],
) -> tuple[str, ...]:
    if not root.exists() or not root.is_dir():
        return ()

    files: list[str] = []

    for path in root.rglob("*"):
        relative = path.relative_to(root)

        if (
            relative.parts
            and relative.parts[0] in excluded_top_level
        ):
            continue

        if path.is_file():
            files.append(relative.as_posix())

    return tuple(sorted(files))


def _clear_directory_contents(
    root: Path,
    *,
    keep_names: set[str] | None = None,
) -> None:
    if not root.exists():
        return

    keep = set() if keep_names is None else set(keep_names)

    for child in list(root.iterdir()):
        if child.name in keep:
            continue
        _remove_path(child)


def _restore_internal_backup(
    *,
    final_output_dir: Path,
    backup_dir: Path,
) -> None:
    final_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _clear_directory_contents(
        final_output_dir,
        keep_names={backup_dir.name},
    )

    for child in list(backup_dir.iterdir()):
        destination = final_output_dir / child.name

        if destination.exists():
            _remove_path(destination)

        os.replace(child, destination)

    backup_dir.rmdir()


def recover_publish_state(
    *,
    final_output_dir: Path,
    run_id: str,
    expected_files: tuple[str, ...] = (),
) -> PublishRecoveryResult:
    """Recover interrupted current or legacy publication state."""

    final_output_dir = final_output_dir.resolve()
    expected = tuple(sorted(expected_files))

    (
        current_staging,
        current_backup,
    ) = _current_publish_artifact_paths(
        final_output_dir=final_output_dir,
        run_id=run_id,
    )

    (
        legacy_staging,
        legacy_backup,
    ) = _legacy_publish_artifact_paths(
        final_output_dir=final_output_dir,
        run_id=run_id,
    )

    # Current publisher: transient state lives inside the writable output
    # mount, so never replace/remove the output mount itself during rollback.
    if current_staging.exists() or current_backup.exists():
        removed_staging = False

        if current_backup.exists():
            current_inventory = _inventory_excluding(
                final_output_dir,
                excluded_top_level={
                    current_staging.name,
                    current_backup.name,
                },
            )

            if expected and current_inventory == expected:
                if current_staging.exists():
                    _remove_path(current_staging)
                    removed_staging = True

                _remove_path(current_backup)

                return PublishRecoveryResult(
                    action="kept_verified_committed_output",
                    final_output_dir=final_output_dir,
                    staging_dir=current_staging,
                    backup_dir=current_backup,
                    restored_backup=False,
                    removed_staging=removed_staging,
                    removed_backup=True,
                    final_output_exists=True,
                )

            if current_staging.exists():
                _remove_path(current_staging)
                removed_staging = True

            _restore_internal_backup(
                final_output_dir=final_output_dir,
                backup_dir=current_backup,
            )

            return PublishRecoveryResult(
                action="rolled_back_to_previous_output",
                final_output_dir=final_output_dir,
                staging_dir=current_staging,
                backup_dir=current_backup,
                restored_backup=True,
                removed_staging=removed_staging,
                removed_backup=False,
                final_output_exists=final_output_dir.exists(),
            )

        # Staging without backup means the prior committed output was never
        # moved aside. Discard only uncommitted staging.
        _remove_path(current_staging)

        return PublishRecoveryResult(
            action="discarded_uncommitted_staging",
            final_output_dir=final_output_dir,
            staging_dir=current_staging,
            backup_dir=current_backup,
            restored_backup=False,
            removed_staging=True,
            removed_backup=False,
            final_output_exists=final_output_dir.exists(),
        )

    # Legacy sibling publication scheme from older releases.
    staging_dir = legacy_staging
    backup_dir = legacy_backup
    removed_staging = False
    removed_backup = False

    final_exists = final_output_dir.exists()
    staging_exists = staging_dir.exists()
    backup_exists = backup_dir.exists()

    if backup_exists and not final_exists:
        if staging_exists:
            _remove_path(staging_dir)
            removed_staging = True

        os.replace(backup_dir, final_output_dir)

        return PublishRecoveryResult(
            action="restored_previous_output",
            final_output_dir=final_output_dir,
            staging_dir=staging_dir,
            backup_dir=backup_dir,
            restored_backup=True,
            removed_staging=removed_staging,
            removed_backup=False,
            final_output_exists=True,
        )

    if backup_exists and final_exists:
        current_inventory = _inventory(final_output_dir)

        if expected and current_inventory == expected:
            _remove_path(backup_dir)
            removed_backup = True

            if staging_exists:
                _remove_path(staging_dir)
                removed_staging = True

            return PublishRecoveryResult(
                action="kept_verified_committed_output",
                final_output_dir=final_output_dir,
                staging_dir=staging_dir,
                backup_dir=backup_dir,
                restored_backup=False,
                removed_staging=removed_staging,
                removed_backup=True,
                final_output_exists=True,
            )

        _remove_path(final_output_dir)
        os.replace(backup_dir, final_output_dir)

        if staging_exists:
            _remove_path(staging_dir)
            removed_staging = True

        return PublishRecoveryResult(
            action="rolled_back_to_previous_output",
            final_output_dir=final_output_dir,
            staging_dir=staging_dir,
            backup_dir=backup_dir,
            restored_backup=True,
            removed_staging=removed_staging,
            removed_backup=False,
            final_output_exists=True,
        )

    if not backup_exists and staging_exists:
        _remove_path(staging_dir)
        removed_staging = True

    if final_output_dir.exists() and expected:
        current_inventory = _inventory(final_output_dir)

        if current_inventory != expected:
            _remove_path(final_output_dir)

            return PublishRecoveryResult(
                action="removed_unverified_output",
                final_output_dir=final_output_dir,
                staging_dir=staging_dir,
                backup_dir=backup_dir,
                restored_backup=False,
                removed_staging=removed_staging,
                removed_backup=False,
                final_output_exists=False,
            )

    action = (
        "discarded_uncommitted_staging"
        if removed_staging
        else "no_recovery_needed"
    )

    return PublishRecoveryResult(
        action=action,
        final_output_dir=final_output_dir,
        staging_dir=staging_dir,
        backup_dir=backup_dir,
        restored_backup=False,
        removed_staging=removed_staging,
        removed_backup=removed_backup,
        final_output_exists=final_output_dir.exists(),
    )


@dataclass
class RunJournal:
    """
    Durable structured record for one orchestrator run.

    No model credentials, endpoint secrets, checker
    content, or environment variables are written here.
    """

    run_dir: Path

    @property
    def events_path(
        self,
    ) -> Path:
        return (
            self.run_dir
            / "events.jsonl"
        )

    @property
    def checkpoint_path(
        self,
    ) -> Path:
        return (
            self.run_dir
            / "checkpoint.json"
        )

    def event(
        self,
        event_type: str,
        *,
        state: OrchestratorState | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[
            str,
            Any,
        ] = {
            "timestamp_utc": (
                utc_now_iso()
            ),
            "event_type": (
                event_type
            ),
        }

        if state is not None:
            payload["run_id"] = (
                state.run_id
            )

            payload["task_id"] = (
                state.task_id
            )

            payload["run_status"] = (
                state.status.value
            )

            payload["current_stage"] = (
                state.current_stage.value
                if state.current_stage
                is not None
                else None
            )

            payload[
                "expected_stage"
            ] = (
                state.expected_stage().value
                if state.expected_stage()
                is not None
                else None
            )

        if details:
            payload["details"] = (
                details
            )

        append_jsonl(
            payload,
            self.events_path,
        )

    def checkpoint(
        self,
        *,
        state: OrchestratorState,
        run_context: RunContext | None = None,
        selection: Any | None = None,
        publish: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[
            str,
            Any,
        ] = {
            "schema_version": 1,
            "updated_at_utc": (
                utc_now_iso()
            ),
            "run_id": (
                state.run_id
            ),
            "task_id": (
                state.task_id
            ),
            "state": (
                state.to_dict()
            ),
        }

        if run_context is not None:
            payload[
                "run_context"
            ] = (
                run_context.snapshot()
            )

        if selection is not None:
            selection_object = getattr(
                selection,
                "selection",
                None,
            )

            if (
                selection_object
                is not None
                and hasattr(
                    selection_object,
                    "to_dict",
                )
            ):
                payload[
                    "selection"
                ] = (
                    selection_object
                    .to_dict()
                )

            selected_item = getattr(
                selection,
                "selected_item",
                None,
            )

            if selected_item is not None:
                payload[
                    "selected_candidate"
                ] = {
                    "candidate_id": (
                        selected_item
                        .candidate
                        .candidate_id
                    ),
                    "candidate_seed": (
                        selected_item
                        .candidate_seed
                    ),
                    "status": (
                        selected_item
                        .candidate
                        .current_status
                    ),
                }

        if publish is not None:
            payload[
                "publish"
            ] = {
                "final_output_dir": str(
                    publish
                    .final_output_dir
                ),
                "published_files": list(
                    publish
                    .published_files
                ),
                "replaced_existing_output": (
                    publish
                    .replaced_existing_output
                ),
            }

        if extra:
            payload["extra"] = (
                extra
            )

        save_json(
            payload,
            self.checkpoint_path,
        )

        return payload

    def load_checkpoint(
        self,
    ) -> dict[str, Any] | None:
        if not self.checkpoint_path.exists():
            return None

        return load_json(
            self.checkpoint_path
        )