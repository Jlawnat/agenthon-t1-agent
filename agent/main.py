from __future__ import annotations

import argparse
import os
from pathlib import Path

from agent.model_client import ModelClient
from agent.orchestrator import solve_task
from agent.runtime_adapters import (
    build_runtime_components,
)


_REQUIRED_RUNTIME_ENV = (
    "MODEL_ENDPOINT",
    "MODEL_NAME",
    "MODEL_TOKEN",
    "QFBENCH_SEED",
)


def _require_runtime_environment() -> None:
    missing = [
        name
        for name in _REQUIRED_RUNTIME_ENV
        if not os.getenv(name)
    ]

    if missing:
        raise RuntimeError(
            "Missing required runtime environment "
            "variable(s): "
            + ", ".join(missing)
        )

    try:
        seed = int(
            os.environ["QFBENCH_SEED"]
        )
    except ValueError as exc:
        raise RuntimeError(
            "QFBENCH_SEED must be a non-negative integer."
        ) from exc

    if seed < 0:
        raise RuntimeError(
            "QFBENCH_SEED must be a non-negative integer."
        )


def _validate_paths(
    *,
    task_dir: Path,
    out_dir: Path,
    work_root: Path,
) -> tuple[Path, Path, Path]:
    task_dir = task_dir.resolve()
    out_dir = out_dir.resolve()
    work_root = work_root.resolve()

    if not task_dir.exists():
        raise FileNotFoundError(
            f"Task directory does not exist: {task_dir}"
        )

    if not task_dir.is_dir():
        raise NotADirectoryError(
            f"Task path is not a directory: {task_dir}"
        )

    if out_dir == task_dir:
        raise RuntimeError(
            "Output directory must be separate from the task directory."
        )

    if work_root == task_dir:
        raise RuntimeError(
            "Work directory must be separate from the task directory."
        )

    out_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    work_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        task_dir,
        out_dir,
        work_root,
    )


def solve(
    task_dir: Path,
    out_dir: Path,
) -> None:
    _require_runtime_environment()

    work_root = Path(
        os.getenv(
            "AGENT_WORK_ROOT",
            "/tmp/agenthon-t1",
        )
    )

    (
        task_dir,
        out_dir,
        work_root,
    ) = _validate_paths(
        task_dir=task_dir,
        out_dir=out_dir,
        work_root=work_root,
    )

    model_client = ModelClient()

    (
        front_half_dependencies,
        generator,
        repairer,
    ) = build_runtime_components(
        model_client
    )

    solve_task(
        task_dir=task_dir,
        final_output_dir=out_dir,
        work_root=work_root,
        front_half_dependencies=(
            front_half_dependencies
        ),
        generator=generator,
        repairer=repairer,
        environ=os.environ,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantagent",
        description=(
            "Agenthon 2026 Track 1 "
            "quantitative-finance coding agent"
        ),
    )

    parser.add_argument(
        "verb",
        choices=["solve"],
        help="Agent action to perform",
    )

    parser.add_argument(
        "--task-dir",
        required=True,
        type=Path,
        help="Read-only task directory",
    )

    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Writable output directory",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verb == "solve":
        solve(
            task_dir=args.task_dir,
            out_dir=args.out,
        )


if __name__ == "__main__":
    main()
