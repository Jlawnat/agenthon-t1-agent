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


def recover_publish_state(
    *,
    final_output_dir: Path,
    run_id: str,
    expected_files: tuple[str, ...] = (),
) -> PublishRecoveryResult:
    """
    Recover an interrupted Phase 4.9 publication.

    Safety policy:

    - an existing backup is treated as the previous
      trusted output;
    - incomplete staging is disposable;
    - a committed final directory is trusted only when
      its inventory matches the recorded expected files;
    - ambiguous states roll back to the previous backup.
    """

    final_output_dir = (
        final_output_dir.resolve()
    )

    parent = (
        final_output_dir.parent
    )

    token = _run_token(
        run_id
    )

    staging_dir = (
        parent
        / (
            f".{final_output_dir.name}"
            f".staging.{token}"
        )
    )

    backup_dir = (
        parent
        / (
            f".{final_output_dir.name}"
            f".backup.{token}"
        )
    )

    expected = tuple(
        sorted(
            expected_files
        )
    )

    removed_staging = False
    removed_backup = False
    restored_backup = False

    final_exists = (
        final_output_dir.exists()
    )

    staging_exists = (
        staging_dir.exists()
    )

    backup_exists = (
        backup_dir.exists()
    )

    if (
        backup_exists
        and not final_exists
    ):
        if staging_exists:
            _remove_path(
                staging_dir
            )

            removed_staging = True

        os.replace(
            backup_dir,
            final_output_dir,
        )

        restored_backup = True

        return PublishRecoveryResult(
            action=(
                "restored_previous_output"
            ),
            final_output_dir=(
                final_output_dir
            ),
            staging_dir=(
                staging_dir
            ),
            backup_dir=(
                backup_dir
            ),
            restored_backup=True,
            removed_staging=(
                removed_staging
            ),
            removed_backup=False,
            final_output_exists=True,
        )

    if (
        backup_exists
        and final_exists
    ):
        current_inventory = (
            _inventory(
                final_output_dir
            )
        )

        if (
            expected
            and current_inventory
            == expected
        ):
            _remove_path(
                backup_dir
            )

            removed_backup = True

            if staging_exists:
                _remove_path(
                    staging_dir
                )

                removed_staging = True

            return PublishRecoveryResult(
                action=(
                    "kept_verified_committed_output"
                ),
                final_output_dir=(
                    final_output_dir
                ),
                staging_dir=(
                    staging_dir
                ),
                backup_dir=(
                    backup_dir
                ),
                restored_backup=False,
                removed_staging=(
                    removed_staging
                ),
                removed_backup=True,
                final_output_exists=True,
            )

        _remove_path(
            final_output_dir
        )

        os.replace(
            backup_dir,
            final_output_dir,
        )

        restored_backup = True

        if staging_exists:
            _remove_path(
                staging_dir
            )

            removed_staging = True

        return PublishRecoveryResult(
            action=(
                "rolled_back_to_previous_output"
            ),
            final_output_dir=(
                final_output_dir
            ),
            staging_dir=(
                staging_dir
            ),
            backup_dir=(
                backup_dir
            ),
            restored_backup=True,
            removed_staging=(
                removed_staging
            ),
            removed_backup=False,
            final_output_exists=True,
        )

    if (
        not backup_exists
        and staging_exists
    ):
        _remove_path(
            staging_dir
        )

        removed_staging = True

    if (
        final_output_dir.exists()
        and expected
    ):
        current_inventory = (
            _inventory(
                final_output_dir
            )
        )

        if (
            current_inventory
            != expected
        ):
            _remove_path(
                final_output_dir
            )

            return PublishRecoveryResult(
                action=(
                    "removed_unverified_output"
                ),
                final_output_dir=(
                    final_output_dir
                ),
                staging_dir=(
                    staging_dir
                ),
                backup_dir=(
                    backup_dir
                ),
                restored_backup=False,
                removed_staging=(
                    removed_staging
                ),
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
        final_output_dir=(
            final_output_dir
        ),
        staging_dir=(
            staging_dir
        ),
        backup_dir=(
            backup_dir
        ),
        restored_backup=False,
        removed_staging=(
            removed_staging
        ),
        removed_backup=False,
        final_output_exists=(
            final_output_dir.exists()
        ),
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