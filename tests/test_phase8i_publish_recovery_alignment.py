from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from agent.orchestrator import _publish_artifact_paths
from agent.publish_pipeline import _run_token
from agent.run_recovery import (
    publish_recovery_needed,
    recover_publish_state,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_orchestrator_paths_match_current_publisher_scheme() -> None:
    with TemporaryDirectory() as directory:
        final_output = Path(directory) / "output"

        staging, backup = _publish_artifact_paths(
            final_output_dir=final_output,
            run_id="run-1",
        )

        token = _run_token("run-1")

        assert staging == (
            final_output.resolve()
            / f".publish-staging.{token}"
        )
        assert backup == (
            final_output.resolve()
            / f".publish-backup.{token}"
        )


def test_recovery_detects_current_internal_artifacts() -> None:
    with TemporaryDirectory() as directory:
        final_output = Path(directory) / "output"
        final_output.mkdir()

        staging, backup = _publish_artifact_paths(
            final_output_dir=final_output,
            run_id="run-2",
        )
        _write(staging / "results.csv", "staged\n")

        assert publish_recovery_needed(
            final_output_dir=final_output,
            run_id="run-2",
        )
        assert not backup.exists()


def test_internal_backup_rolls_back_partial_commit() -> None:
    with TemporaryDirectory() as directory:
        final_output = Path(directory) / "output"
        final_output.mkdir()

        staging, backup = _publish_artifact_paths(
            final_output_dir=final_output,
            run_id="run-3",
        )

        _write(final_output / "results.csv", "partial-new\n")
        _write(staging / "second.csv", "not-promoted\n")
        _write(backup / "previous.csv", "previous-valid\n")

        result = recover_publish_state(
            final_output_dir=final_output,
            run_id="run-3",
            expected_files=("results.csv", "second.csv"),
        )

        assert result.action == "rolled_back_to_previous_output"
        assert (
            final_output / "previous.csv"
        ).read_text(encoding="utf-8") == "previous-valid\n"
        assert not (final_output / "results.csv").exists()
        assert not staging.exists()
        assert not backup.exists()


def test_internal_backup_removed_after_complete_commit() -> None:
    with TemporaryDirectory() as directory:
        final_output = Path(directory) / "output"
        final_output.mkdir()

        staging, backup = _publish_artifact_paths(
            final_output_dir=final_output,
            run_id="run-4",
        )

        _write(final_output / "results.csv", "new-valid\n")
        _write(backup / "previous.csv", "previous-valid\n")
        _write(staging / "stale.tmp", "stale\n")

        result = recover_publish_state(
            final_output_dir=final_output,
            run_id="run-4",
            expected_files=("results.csv",),
        )

        assert result.action == "kept_verified_committed_output"
        assert (
            final_output / "results.csv"
        ).read_text(encoding="utf-8") == "new-valid\n"
        assert not staging.exists()
        assert not backup.exists()


def test_internal_staging_without_backup_is_discarded_only() -> None:
    with TemporaryDirectory() as directory:
        final_output = Path(directory) / "output"
        final_output.mkdir()

        _write(final_output / "previous.csv", "previous-valid\n")

        staging, backup = _publish_artifact_paths(
            final_output_dir=final_output,
            run_id="run-5",
        )
        _write(staging / "results.csv", "uncommitted\n")

        result = recover_publish_state(
            final_output_dir=final_output,
            run_id="run-5",
            expected_files=("results.csv",),
        )

        assert result.action == "discarded_uncommitted_staging"
        assert (
            final_output / "previous.csv"
        ).read_text(encoding="utf-8") == "previous-valid\n"
        assert not staging.exists()
        assert not backup.exists()
