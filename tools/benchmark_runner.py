from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from typing import Any


DEFAULT_IMAGE = "agenthon-t1:phase5"
DEFAULT_REPO = (
    Path.home()
    / "track1-coding-public"
)
DEFAULT_INVENTORY = (
    Path("benchmark")
    / "public_units_inventory.json"
)
DEFAULT_RUNS_DIR = (
    Path("benchmark")
    / "runs"
)


def read_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def write_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def read_card(
    unit_dir: Path,
) -> dict[str, Any]:
    with (
        unit_dir
        / "card.toml"
    ).open("rb") as handle:
        payload = tomllib.load(
            handle
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"Invalid card: {unit_dir}"
        )

    return payload


def safe_get(
    payload: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    current: Any = payload

    for key in keys:
        if not isinstance(
            current,
            dict,
        ):
            return default

        if key not in current:
            return default

        current = current[
            key
        ]

    return current


def git_commit(
    repo: Path,
) -> str | None:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

    except Exception:
        return None

    value = (
        completed
        .stdout
        .strip()
    )

    return value or None


def docker_image_id(
    image: str,
) -> str | None:
    try:
        completed = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                image,
                "--format",
                "{{.Id}}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )

    except Exception:
        return None

    value = (
        completed
        .stdout
        .strip()
    )

    return value or None


def docker_network_exists(
    network: str,
) -> bool:
    try:
        subprocess.run(
            [
                "docker",
                "network",
                "inspect",
                network,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )

    except Exception:
        return False

    return True


def check_command(
    name: str,
) -> None:
    if shutil.which(
        name
    ) is None:
        raise SystemExit(
            f"Required command not found: {name}"
        )


def load_inventory(
    path: Path,
) -> list[dict[str, Any]]:
    payload = read_json(
        path
    )

    units = payload.get(
        "units"
    )

    if not isinstance(
        units,
        list,
    ):
        raise SystemExit(
            "Inventory JSON does not contain a units list."
        )

    result: list[
        dict[str, Any]
    ] = []

    for item in units:
        if isinstance(
            item,
            dict,
        ):
            result.append(
                item
            )

    return result


def select_units(
    inventory: list[dict[str, Any]],
    requested: list[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    if requested:
        by_name = {
            str(
                item.get(
                    "unit_dir"
                )
            ): item
            for item
            in inventory
        }

        missing = [
            name
            for name
            in requested
            if name not in by_name
        ]

        if missing:
            raise SystemExit(
                "Unknown unit(s): "
                + ", ".join(
                    missing
                )
            )

        selected = [
            by_name[
                name
            ]
            for name
            in requested
        ]

    else:
        selected = list(
            inventory
        )

    if limit is not None:
        selected = selected[
            :limit
        ]

    return selected


def parse_reward(
    output_dir: Path,
) -> dict[str, Any] | None:
    direct = (
        output_dir
        / "reward.json"
    )

    candidates: list[Path] = []

    if direct.is_file():
        candidates.append(
            direct
        )

    if output_dir.exists():
        for path in sorted(
            output_dir.rglob(
                "reward.json"
            )
        ):
            if (
                path.is_file()
                and path
                not in candidates
            ):
                candidates.append(
                    path
                )

    for path in candidates:
        try:
            payload = read_json(
                path
            )
        except Exception:
            continue

        if isinstance(
            payload,
            dict,
        ):
            result = dict(
                payload
            )
            result[
                "_reward_path"
            ] = str(
                path
            )
            return result

    return None


def parse_pytest_report(
    output_dir: Path,
) -> dict[str, Any]:
    direct = (
        output_dir
        / "pytest_report.json"
    )

    if not direct.is_file():
        return {
            "pytest_total": None,
            "pytest_passed": None,
            "pytest_failed": None,
            "failed_tests": [],
        }

    try:
        report = read_json(
            direct
        )
    except Exception:
        return {
            "pytest_total": None,
            "pytest_passed": None,
            "pytest_failed": None,
            "failed_tests": [],
        }

    tests = report.get(
        "tests",
        []
    )

    if not isinstance(
        tests,
        list,
    ):
        tests = []

    failed_tests: list[str] = []

    passed = 0
    failed = 0

    for test in tests:
        if not isinstance(
            test,
            dict,
        ):
            continue

        outcome = test.get(
            "outcome"
        )

        if outcome == "passed":
            passed += 1

        if outcome in {
            "failed",
            "error",
        }:
            failed += 1

            nodeid = test.get(
                "nodeid"
            )

            if nodeid:
                failed_tests.append(
                    str(
                        nodeid
                    )
                )

    return {
        "pytest_total": len(
            tests
        ),
        "pytest_passed": passed,
        "pytest_failed": failed,
        "failed_tests": (
            failed_tests
        ),
    }


def list_deliverables(
    output_dir: Path,
) -> list[str]:
    if not output_dir.exists():
        return []

    ignored = {
        "reward.json",
        "pytest_report.json",
    }

    result: list[str] = []

    for path in sorted(
        output_dir.rglob("*")
    ):
        if not path.is_file():
            continue

        relative = path.relative_to(
            output_dir
        ).as_posix()

        if (
            Path(relative).name
            in ignored
        ):
            continue

        result.append(
            relative
        )

    return result


def classify_status(
    *,
    return_code: int | None,
    timed_out: bool,
    reward: float | None,
) -> str:
    if timed_out:
        return "runner_timeout"

    if reward == 1.0:
        return "pass"

    if reward == 0.0:
        return "fail"

    if (
        return_code is not None
        and return_code != 0
    ):
        return "harness_error"

    return "no_reward"


def terminate_process_group(
    process: subprocess.Popen[Any],
) -> None:
    if os.name == "posix":
        try:
            os.killpg(
                process.pid,
                signal.SIGTERM,
            )
        except ProcessLookupError:
            return
        except OSError:
            pass

        try:
            process.wait(
                timeout=3,
            )
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            os.killpg(
                process.pid,
                signal.SIGKILL,
            )
        except ProcessLookupError:
            return
        except OSError:
            pass

    else:
        try:
            process.terminate()
        except ProcessLookupError:
            return

    try:
        process.wait(
            timeout=3,
        )
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass


def run_smoke(
    *,
    repo: Path,
    unit_dir: Path,
    output_dir: Path,
    image: str,
    timeout_seconds: float,
    log_path: Path,
) -> tuple[
    int | None,
    bool,
    float,
]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "qfbench2",
        "smoke",
        str(
            unit_dir.resolve()
        ),
        str(
            output_dir.resolve()
        ),
        "--track",
        "coding",
        "--agent-image",
        image,
    ]

    environment = os.environ.copy()

    current_pythonpath = (
        environment.get(
            "PYTHONPATH",
            ""
        )
    )

    repo_value = str(
        repo.resolve()
    )

    if current_pythonpath:
        environment[
            "PYTHONPATH"
        ] = (
            repo_value
            + os.pathsep
            + current_pythonpath
        )
    else:
        environment[
            "PYTHONPATH"
        ] = repo_value

    started = (
        time.perf_counter()
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_handle:
        log_handle.write(
            "$ "
            + " ".join(
                command
            )
            + "\n\n"
        )
        log_handle.flush()

        process = subprocess.Popen(
            command,
            cwd=repo,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
            start_new_session=(
                os.name
                == "posix"
            ),
        )

        timed_out = False

        try:
            return_code = (
                process.wait(
                    timeout=(
                        timeout_seconds
                    )
                )
            )

        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(
                process
            )
            return_code = None

            log_handle.write(
                "\n[BENCHMARK RUNNER] "
                "Outer timeout reached.\n"
            )

    elapsed = (
        time.perf_counter()
        - started
    )

    return (
        return_code,
        timed_out,
        elapsed,
    )


def existing_result(
    path: Path,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    try:
        payload = read_json(
            path
        )
    except Exception:
        return None

    if isinstance(
        payload,
        dict,
    ):
        return payload

    return None


def write_results_csv(
    rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    fields = [
        "unit",
        "category",
        "difficulty",
        "status",
        "reward",
        "elapsed_seconds",
        "agent_timeout_sec",
        "qfbench_exit_code",
        "runner_timed_out",
        "pytest_exit_code",
        "pytest_total",
        "pytest_passed",
        "pytest_failed",
        "deliverable_count",
        "failed_test_count",
        "output_dir",
        "log_path",
    ]

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(
                        key,
                        "",
                    )
                    for key
                    in fields
                }
            )


def print_summary(
    rows: list[dict[str, Any]],
) -> None:
    total = len(
        rows
    )

    passes = sum(
        1
        for row
        in rows
        if row.get(
            "status"
        )
        == "pass"
    )

    fails = sum(
        1
        for row
        in rows
        if row.get(
            "status"
        )
        == "fail"
    )

    errors = (
        total
        - passes
        - fails
    )

    pass_rate = (
        passes
        / total
        if total
        else 0.0
    )

    print()
    print(
        "=" * 72
    )
    print(
        "AGENTHON T1 — PUBLIC BENCHMARK SUMMARY"
    )
    print(
        "=" * 72
    )
    print(
        f"Units completed: {total}"
    )
    print(
        f"Passed: {passes}"
    )
    print(
        f"Failed: {fails}"
    )
    print(
        f"Harness/other errors: {errors}"
    )
    print(
        f"Observed pass@1: "
        f"{pass_rate:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Agenthon T1 image "
            "against official public-dev units "
            "using the official qfbench2 smoke runner."
        )
    )

    parser.add_argument(
        "--repo",
        type=Path,
        default=DEFAULT_REPO,
    )

    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY,
    )

    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
    )

    parser.add_argument(
        "--unit",
        action="append",
        default=[],
        help=(
            "Run only this unit. "
            "Repeat --unit for several units."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--run-name",
        default=None,
    )

    parser.add_argument(
        "--timeout-buffer",
        type=float,
        default=600.0,
        help=(
            "Extra outer-runner seconds added "
            "to [agent].timeout_sec."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--allow-offline",
        action="store_true",
        help=(
            "Allow execution without qfb2-eval/model access. "
            "This is only a harness smoke test and is not a "
            "meaningful performance benchmark for a model-dependent agent."
        ),
    )

    args = parser.parse_args()

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        raise SystemExit(
            "--limit must be positive."
        )

    if (
        args.timeout_buffer
        < 0
    ):
        raise SystemExit(
            "--timeout-buffer must be non-negative."
        )

    repo = (
        args.repo
        .expanduser()
        .resolve()
    )

    inventory_path = (
        args.inventory
        .expanduser()
        .resolve()
    )

    runs_dir = (
        args.runs_dir
        .expanduser()
        .resolve()
    )

    if not repo.is_dir():
        raise SystemExit(
            f"Official repo not found: {repo}"
        )

    if not inventory_path.is_file():
        raise SystemExit(
            "Inventory not found: "
            f"{inventory_path}"
        )

    check_command(
        "docker"
    )

    check_command(
        "qfbench2"
    )

    image_id = docker_image_id(
        args.image
    )

    if image_id is None:
        raise SystemExit(
            "Docker image not found: "
            f"{args.image}"
        )

    eval_network = (
        docker_network_exists(
            "qfb2-eval"
        )
    )

    model_endpoint = (
        os.environ.get(
            "MODEL_ENDPOINT"
        )
    )

    model_name = (
        os.environ.get(
            "MODEL_NAME"
        )
    )

    if not args.allow_offline:
        missing: list[str] = []

        if not eval_network:
            missing.append(
                "Docker network qfb2-eval"
            )

        if not model_endpoint:
            missing.append(
                "MODEL_ENDPOINT"
            )

        if not model_name:
            missing.append(
                "MODEL_NAME"
            )

        if missing:
            raise SystemExit(
                "A meaningful model-backed benchmark "
                "cannot start because these are missing: "
                + ", ".join(
                    missing
                )
                + ".\n"
                "Use --allow-offline only for a "
                "harness/packaging smoke test."
            )

    inventory = load_inventory(
        inventory_path
    )

    selected = select_units(
        inventory,
        requested=args.unit,
        limit=args.limit,
    )

    run_name = (
        args.run_name
        or datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    run_root = (
        runs_dir
        / run_name
    )

    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "run_name": run_name,
        "started_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "official_repo": str(
            repo
        ),
        "official_repo_commit": (
            git_commit(
                repo
            )
        ),
        "inventory": str(
            inventory_path
        ),
        "agent_image": args.image,
        "agent_image_id": (
            image_id
        ),
        "qfb2_eval_network_present": (
            eval_network
        ),
        "model_endpoint_configured": (
            bool(
                model_endpoint
            )
        ),
        "model_name": (
            model_name
        ),
        "allow_offline": (
            args.allow_offline
        ),
        "selected_units": [
            item.get(
                "unit_dir"
            )
            for item
            in selected
        ],
    }

    write_json(
        run_root
        / "run_metadata.json",
        metadata,
    )

    print(
        "=" * 72
    )
    print(
        "AGENTHON T1 — PUBLIC BENCHMARK RUNNER"
    )
    print(
        "=" * 72
    )
    print(
        f"Image: {args.image}"
    )
    print(
        f"Image ID: {image_id}"
    )
    print(
        f"Units selected: {len(selected)}"
    )
    print(
        f"Run directory: {run_root}"
    )
    print(
        f"qfb2-eval present: {eval_network}"
    )
    print(
        f"Model configured: "
        f"{bool(model_endpoint and model_name)}"
    )
    print()

    results: list[
        dict[str, Any]
    ] = []

    for index, inventory_item in enumerate(
        selected,
        start=1,
    ):
        unit_name = str(
            inventory_item[
                "unit_dir"
            ]
        )

        unit_dir = (
            repo
            / "units"
            / unit_name
        )

        unit_run_dir = (
            run_root
            / unit_name
        )

        output_dir = (
            unit_run_dir
            / "output"
        )

        log_path = (
            unit_run_dir
            / "smoke.log"
        )

        result_path = (
            unit_run_dir
            / "result.json"
        )

        if args.resume:
            cached = existing_result(
                result_path
            )

            if cached is not None:
                print(
                    f"[{index}/{len(selected)}] "
                    f"{unit_name}: resumed "
                    f"({cached.get('status')})"
                )

                results.append(
                    cached
                )
                continue

        card = read_card(
            unit_dir
        )

        agent_timeout = float(
            safe_get(
                card,
                "agent",
                "timeout_sec",
                default=1800.0,
            )
        )

        outer_timeout = (
            agent_timeout
            + float(
                args.timeout_buffer
            )
        )

        print(
            f"[{index}/{len(selected)}] "
            f"{unit_name}"
        )
        print(
            f"  agent timeout: "
            f"{agent_timeout:.0f}s"
        )

        return_code, timed_out, elapsed = (
            run_smoke(
                repo=repo,
                unit_dir=unit_dir,
                output_dir=output_dir,
                image=args.image,
                timeout_seconds=(
                    outer_timeout
                ),
                log_path=log_path,
            )
        )

        reward_payload = (
            parse_reward(
                output_dir
            )
        )

        reward: float | None = None
        pytest_exit_code: int | None = None
        reward_details: str | None = None

        if reward_payload is not None:
            raw_reward = (
                reward_payload.get(
                    "reward"
                )
            )

            if isinstance(
                raw_reward,
                (int, float),
            ):
                reward = float(
                    raw_reward
                )

            raw_pytest_exit = (
                reward_payload.get(
                    "pytest_exit_code"
                )
            )

            if isinstance(
                raw_pytest_exit,
                int,
            ):
                pytest_exit_code = (
                    raw_pytest_exit
                )

            raw_details = (
                reward_payload.get(
                    "details"
                )
            )

            if raw_details is not None:
                reward_details = str(
                    raw_details
                )

        pytest_summary = (
            parse_pytest_report(
                output_dir
            )
        )

        deliverables = (
            list_deliverables(
                output_dir
            )
        )

        status = classify_status(
            return_code=return_code,
            timed_out=timed_out,
            reward=reward,
        )

        result = {
            "unit": unit_name,
            "category": inventory_item.get(
                "category"
            ),
            "difficulty": inventory_item.get(
                "difficulty"
            ),
            "status": status,
            "reward": reward,
            "reward_details": (
                reward_details
            ),
            "elapsed_seconds": round(
                elapsed,
                6,
            ),
            "agent_timeout_sec": (
                agent_timeout
            ),
            "outer_timeout_sec": (
                outer_timeout
            ),
            "qfbench_exit_code": (
                return_code
            ),
            "runner_timed_out": (
                timed_out
            ),
            "pytest_exit_code": (
                pytest_exit_code
            ),
            **pytest_summary,
            "deliverables": (
                deliverables
            ),
            "deliverable_count": len(
                deliverables
            ),
            "failed_test_count": len(
                pytest_summary[
                    "failed_tests"
                ]
            ),
            "output_dir": str(
                output_dir
            ),
            "log_path": str(
                log_path
            ),
        }

        write_json(
            result_path,
            result,
        )

        results.append(
            result
        )

        print(
            f"  status: {status}"
        )
        print(
            f"  reward: {reward}"
        )
        print(
            f"  elapsed: "
            f"{elapsed:.1f}s"
        )
        print(
            f"  deliverables: "
            f"{len(deliverables)}"
        )

        if (
            pytest_summary[
                "failed_tests"
            ]
        ):
            print(
                "  first failed test: "
                + pytest_summary[
                    "failed_tests"
                ][0]
            )

        print()

        write_results_csv(
            results,
            run_root
            / "benchmark_results.csv",
        )

        write_json(
            run_root
            / "benchmark_results.json",
            {
                "run_metadata": (
                    metadata
                ),
                "results": (
                    results
                ),
            },
        )

    write_results_csv(
        results,
        run_root
        / "benchmark_results.csv",
    )

    write_json(
        run_root
        / "benchmark_results.json",
        {
            "run_metadata": (
                metadata
            ),
            "results": (
                results
            ),
        },
    )

    print_summary(
        results
    )

    print()
    print(
        "Saved:"
    )
    print(
        "  "
        + str(
            run_root
            / "benchmark_results.csv"
        )
    )
    print(
        "  "
        + str(
            run_root
            / "benchmark_results.json"
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )