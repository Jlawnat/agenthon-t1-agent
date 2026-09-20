from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from agent.model_client import ModelClient
from agent.offline_runtime import solve_offline
from agent.orchestrator import solve_task
from agent.runtime_adapters import (
    build_runtime_components,
)


_MODEL_RUNTIME_ENV = (
    "MODEL_ENDPOINT",
    "MODEL_NAME",
    "MODEL_TOKEN",
)


def _require_seed() -> int:
    raw = os.getenv("QFBENCH_SEED")
    if not raw:
        raise RuntimeError(
            "Missing required runtime environment variable: QFBENCH_SEED"
        )

    try:
        seed = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            "QFBENCH_SEED must be a non-negative integer."
        ) from exc

    if seed < 0:
        raise RuntimeError(
            "QFBENCH_SEED must be a non-negative integer."
        )

    return seed


def _runtime_mode() -> str:
    configured = [
        bool(os.getenv(name))
        for name in _MODEL_RUNTIME_ENV
    ]

    if all(configured):
        return "model"

    if any(configured):
        missing = [
            name
            for name, present
            in zip(_MODEL_RUNTIME_ENV, configured)
            if not present
        ]
        raise RuntimeError(
            "Partial model runtime configuration. Missing: "
            + ", ".join(missing)
        )

    return "offline"


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
    seed = _require_seed()

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

    mode = _runtime_mode()

    if mode == "offline":
        solve_offline(
            task_dir=task_dir,
            out_dir=out_dir,
            seed=seed,
        )
        return

    # Hybrid generalization path:
    #
    # 1. Prefer the deterministic library when it can satisfy the complete
    #    output contract.
    # 2. If no deterministic skill can complete the task, fall through to
    #    the generic model-driven planner/generator/validator/repair loop.
    #
    # The deterministic probe writes into an isolated temporary directory so
    # a failed or incomplete attempt cannot contaminate final output.
    with tempfile.TemporaryDirectory(
        prefix="offline-probe-",
        dir=str(work_root),
    ) as temporary:
        probe_output = Path(temporary) / "output"

        try:
            solve_offline(
                task_dir=task_dir,
                out_dir=probe_output,
                seed=seed,
            )
        except Exception:
            # An unfamiliar domain is expected to reach this path.
            pass
        else:
            if not probe_output.is_dir():
                raise RuntimeError(
                    "Offline solver reported success without an output directory."
                )

            if out_dir.exists():
                if out_dir.is_dir():
                    shutil.rmtree(out_dir)
                else:
                    out_dir.unlink()

            shutil.copytree(
                probe_output,
                out_dir,
            )
            return

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
