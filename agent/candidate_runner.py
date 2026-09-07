from __future__ import annotations

from pathlib import Path

from agent.candidate_workspace import CandidateWorkspace
from agent.executor import (
    ExecutionResult,
    run_python_candidate,
)


def write_smoke_candidate(
    workspace: CandidateWorkspace,
) -> Path:
    script_path = (
        workspace.source_dir
        / "solver.py"
    )

    script_path.write_text(
        """
from pathlib import Path
import os


def main() -> None:
    input_root = Path(
        os.environ.get(
            "INPUT_DIR",
            "/input",
        )
    )
    output_dir = Path(
        os.environ.get(
            "OUTPUT_DIR",
            "/output",
        )
    )

    input_file = (
        input_root
        / "data"
        / "options.parquet"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"Missing input file: {input_file}"
        )

    marker = (
        output_dir
        / "candidate_smoke.txt"
    )

    marker.write_text(
        "candidate workspace execution works\\n",
        encoding="utf-8",
    )

    print(
        "candidate workspace execution successful"
    )


if __name__ == "__main__":
    main()
""".strip()
        + "\n",
        encoding="utf-8",
    )

    return script_path


def run_candidate(
    workspace: CandidateWorkspace,
    script_path: Path,
    *,
    timeout_seconds: float = 120,
    env_overrides:
        dict[str, str]
        | None = None,
) -> ExecutionResult:
    validated_script = (
        workspace.validate_script_path(
            script_path
        )
    )

    return run_python_candidate(
        script_path=(
            validated_script
        ),
        cwd=workspace.root_dir,
        timeout_seconds=(
            timeout_seconds
        ),
        env_overrides=(
            env_overrides
        ),
    )