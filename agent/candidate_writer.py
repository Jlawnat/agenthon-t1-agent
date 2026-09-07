from __future__ import annotations

from pathlib import Path

from agent.candidate_workspace import CandidateWorkspace
from agent.code_validator import validate_python_code


def write_candidate_solver(
    *,
    workspace: CandidateWorkspace,
    model_text: str,
) -> Path:

    code = validate_python_code(
        model_text
    )

    solver_path = (
        workspace.source_dir
        / "solver.py"
    )

    solver_path.write_text(
        code + "\n",
        encoding="utf-8",
    )

    return solver_path