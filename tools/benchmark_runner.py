from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
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


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


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

    if not isinstance(
        payload,
        dict,
    ):
        raise SystemExit(
            "Inventory JSON must contain an object."
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

    for index, item in enumerate(units):
        if not isinstance(
            item,
            dict,
        ):
            raise SystemExit(
                f"Inventory unit at index {index} must be an object."
            )

        unit_dir = item.get(
            "unit_dir"
        )

        if not isinstance(
            unit_dir,
            str,
        ) or not unit_dir.strip():
            raise SystemExit(
                f"Inventory unit at index {index} has an invalid unit_dir."
            )

        result.append(
            item
        )

    return result


def select_units(
    inventory: list[dict[str, Any]],
    requested: list[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    if (
        limit is not None
        and limit <= 0
    ):
        raise SystemExit(
            "--limit must be positive."
        )

    if requested:
        if len(set(requested)) != len(requested):
            raise SystemExit(
                "Duplicate --unit values are not allowed."
            )

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


def validate_unit_name(
    value: str,
) -> str:
    name = value.strip()

    if (
        not name
        or Path(name).is_absolute()
        or ".." in Path(name).parts
        or Path(name).name != name
    ):
        raise SystemExit(
            f"Invalid unit directory name: {value}"
        )

    return name


def unique_run_name(
    runs_dir: Path,
    requested: str | None,
) -> str:
    if requested:
        if (
            Path(requested).is_absolute()
            or ".." in Path(requested).parts
            or Path(requested).name != requested
            or not requested.strip()
        ):
            raise SystemExit(
                "--run-name must be a non-empty directory name."
            )

        return requested

    base = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    candidate = base
    suffix = 1

    while (runs_dir / candidate).exists():
        candidate = f"{base}-{suffix}"
        suffix += 1

    return candidate


def resolve_timeouts(
    card: dict[str, Any],
    unit_name: str,
    timeout_buffer: float,
) -> tuple[float, float | None, float]:
    agent_timeout = float(
        safe_get(
            card,
            "agent",
            "timeout_sec",
            default=1800.0,
        )
    )

    if (
        not math.isfinite(agent_timeout)
        or agent_timeout <= 0
    ):
        raise ValueError(
            f"Invalid agent timeout for {unit_name}: "
            f"{agent_timeout}"
        )

    raw_verifier_timeout = safe_get(
        card,
        "verifier",
        "timeout_sec",
        default=None,
    )

    verifier_timeout = None

    if raw_verifier_timeout is not None:
        verifier_timeout = float(
            raw_verifier_timeout
        )

        if (
            not math.isfinite(verifier_timeout)
            or verifier_timeout <= 0
        ):
            raise ValueError(
                f"Invalid verifier timeout for {unit_name}: "
                f"{verifier_timeout}"
            )

    outer_timeout = (
        max(
            agent_timeout,
            verifier_timeout or 0.0,
        )
        + timeout_buffer
    )

    return (
        agent_timeout,
        verifier_timeout,
        outer_timeout,
    )


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

    if not isinstance(
        report,
        dict,
    ):
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
    pytest_failed: int | None = None,
) -> str:
    if timed_out:
        return "runner_timeout"

    if (
        return_code is not None
        and return_code != 0
    ):
        return "harness_error"

    if (
        pytest_failed is not None
        and pytest_failed > 0
    ):
        return "fail"

    if reward == 1.0:
        return "pass"

    if reward == 0.0:
        return "fail"

    return "no_reward"


def benchmark_exit_code(
    rows: list[dict[str, Any]],
) -> int:
    if any(
        row.get("status")
        in {
            "harness_error",
            "runner_timeout",
        }
        for row in rows
    ):
        return 1

    return 0


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


@dataclass(frozen=True)
class SmokeExecution:
    agent_return_code: int | None
    agent_timed_out: bool
    checker_return_code: int | None
    checker_timed_out: bool
    verifier_return_code: int | None
    verifier_timed_out: bool
    elapsed_seconds: float
    launch_error: str | None = None

    @property
    def return_code(self) -> int | None:
        if self.launch_error is not None:
            return 127

        if (
            self.agent_return_code is not None
            and self.agent_return_code != 0
        ):
            return self.agent_return_code

        if (
            self.checker_return_code is not None
            and self.checker_return_code != 0
        ):
            return self.checker_return_code

        if (
            self.verifier_return_code is not None
            and self.verifier_return_code not in {0, 2}
        ):
            return self.verifier_return_code

        return 0

    @property
    def timed_out(self) -> bool:
        return (
            self.agent_timed_out
            or self.checker_timed_out
            or self.verifier_timed_out
        )


def _docker_environment_args(
    environment: dict[str, str],
) -> list[str]:
    keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "MODEL_ENDPOINT",
        "MODEL_NAME",
        "MODEL_TOKEN",
        "QFBENCH_SEED",
        "QFBENCH_NETWORK",
        "AGENT_WORK_ROOT",
        "MPLCONFIGDIR",
    )

    result: list[str] = []

    for key in keys:
        if key in environment:
            result.extend(
                [
                    "-e",
                    key,
                ]
            )

    return result


def build_agent_command(
    *,
    unit_dir: Path,
    output_dir: Path,
    image: str,
    network: str,
    environment: dict[str, str],
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        network,
        "--read-only",
        "--user",
        "65534:65534",
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777",
        "--pids-limit",
        "256",
        "--ulimit",
        "nofile=1024:1024",
        "--ulimit",
        "nproc=256:256",
        *_docker_environment_args(environment),
        "-v",
        f"{unit_dir.resolve()}:/input:ro",
        "-v",
        f"{output_dir.resolve()}:/output",
        "-v",
        f"{output_dir.resolve()}:/app/output",
        image,
        "solve",
        "--task-dir",
        "/input",
        "--out",
        "/app/output",
    ]


def build_verifier_command(
    *,
    unit_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        "qfbench2",
        "smoke",
        str(unit_dir.resolve()),
        str(output_dir.resolve()),
        "--track",
        "coding",
    ]


def _checker_input_mount_args(
    unit_dir: Path,
) -> list[str]:
    """Reproduce the task environment's data layout for public checkers.

    QFBench units commonly use:
      COPY data/ /app/data/
    or:
      COPY data/ /app/

    Some units also rename individual files, for example:
      COPY data/stock_chars.pqt /app/data/stock_data.parquet

    Preserve the broad compatibility mounts and recreate simple file-level
    COPY aliases from the unit Dockerfile as read-only bind mounts.
    """
    data_dir = (
        unit_dir
        / "environment"
        / "data"
    )

    if not data_dir.is_dir():
        return []

    args = []

    for child in sorted(
        data_dir.iterdir(),
        key=lambda x: x.name,
    ):
        # Reproduce both common COPY layouts without bind-mounting
        # /app/data itself.  Keeping the parent path unmounted allows
        # Dockerfile-derived file aliases to coexist beneath /app/data.
        args.extend(
            [
                "-v",
                (
                    f"{child.resolve()}:"
                    f"/app/data/{child.name}:ro"
                ),
                "-v",
                f"{child.resolve()}:/app/{child.name}:ro",
            ]
        )

    dockerfile = (
        unit_dir
        / "environment"
        / "Dockerfile"
    )

    if dockerfile.is_file():
        for raw_line in dockerfile.read_text().splitlines():
            parts = raw_line.strip().split()

            if (
                len(parts) == 3
                and parts[0].upper() == "COPY"
                and parts[1].startswith("data/")
                and parts[2].startswith("/app/")
            ):
                source_rel = parts[1][len("data/"):]
                source_path = data_dir / source_rel
                container_path = parts[2]

                if source_path.is_file():
                    args.extend(
                        [
                            "-v",
                            (
                                f"{source_path.resolve()}:"
                                f"{container_path}:ro"
                            ),
                        ]
                    )

    return args


def build_checker_command(
    *,
    unit_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "-e",
        "OUTPUT_DIR=/app/output",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        *_checker_input_mount_args(unit_dir),
        "-v",
        f"{unit_dir.resolve()}:/input:ro",
        "-v",
        f"{(unit_dir.resolve() / 'checks')}:/tests:ro",
        "-v",
        f"{(output_dir.parent / 'checker_logs').resolve()}:/logs",
        "-v",
        f"{output_dir.resolve()}:/app/output",
        "-v",
        f"{output_dir.resolve()}:/output",
        "finance-bench-sandbox:latest",
        "bash",
        "/input/checks/test.sh",
    ]


def _run_logged_command(
    *,
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: float,
    log_handle: Any,
    label: str,
) -> tuple[int | None, bool, str | None]:
    log_handle.write(
        f"[{label}] $ "
        + " ".join(command)
        + "\n\n"
    )
    log_handle.flush()

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
            start_new_session=(
                os.name == "posix"
            ),
        )
    except OSError:
        log_handle.write(
            f"[BENCHMARK RUNNER] {label} subprocess could not be started.\n"
        )
        log_handle.flush()
        return (
            None,
            False,
            label,
        )

    try:
        return (
            process.wait(
                timeout=timeout_seconds
            ),
            False,
                None,
        )
    except subprocess.TimeoutExpired:
        terminate_process_group(
            process
        )
        log_handle.write(
            f"\n[BENCHMARK RUNNER] {label} timeout reached.\n"
        )
        log_handle.flush()
        return (
            None,
            True,
            None,
        )


def run_smoke(
    *,
    repo: Path,
    unit_dir: Path,
    output_dir: Path,
    image: str,
    network: str,
    agent_timeout_seconds: float,
    verifier_timeout_seconds: float,
    outer_timeout_seconds: float | None = None,
    log_path: Path,
) -> SmokeExecution:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_dir.chmod(0o777)

    checker_logs_dir = (
        output_dir.parent
        / "checker_logs"
    )
    checker_logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    checker_logs_dir.chmod(0o777)

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    docker_environment = dict(
        environment
    )
    docker_environment.setdefault(
        "QFBENCH_NETWORK",
        "restricted",
    )

    agent_command = build_agent_command(
        unit_dir=unit_dir,
        output_dir=output_dir,
        image=image,
        network=network,
        environment=docker_environment,
    )

    checker_command = build_checker_command(
        unit_dir=unit_dir,
        output_dir=output_dir,
    )

    verifier_command = (
        build_verifier_command(
            unit_dir=unit_dir,
            output_dir=output_dir,
        )
        if shutil.which("qfbench2") is not None
        else None
    )

    started = (
        time.perf_counter()
    )
    deadline = (
        started + outer_timeout_seconds
        if outer_timeout_seconds is not None
        else None
    )

    def phase_timeout(
        configured: float,
    ) -> float:
        if deadline is None:
            return configured

        return max(
            0.001,
            min(
                configured,
                deadline - time.perf_counter(),
            ),
        )

    def deadline_expired() -> bool:
        return (
            deadline is not None
            and time.perf_counter() >= deadline
        )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_handle:
        agent_return_code, agent_timed_out, launch_error = (
            _run_logged_command(
                command=agent_command,
                cwd=repo,
                environment=docker_environment,
                timeout_seconds=phase_timeout(
                    agent_timeout_seconds
                ),
                log_handle=log_handle,
                label="agent",
            )
        )

        checker_return_code: int | None = None
        checker_timed_out = False
        verifier_return_code: int | None = None
        verifier_timed_out = False

        if (
            agent_return_code == 0
            and not agent_timed_out
            and launch_error is None
            and not deadline_expired()
        ):
            checker_return_code, checker_timed_out, checker_error = (
                _run_logged_command(
                    command=checker_command,
                    cwd=repo,
                    environment=environment,
                    timeout_seconds=phase_timeout(
                        verifier_timeout_seconds
                    ),
                    log_handle=log_handle,
                    label="checker",
                )
            )
            launch_error = checker_error
        elif (
            agent_return_code == 0
            and not agent_timed_out
            and launch_error is None
        ):
            checker_timed_out = True

        if verifier_command is not None:
            if (
                not agent_timed_out
                and launch_error is None
                and not checker_timed_out
                and not deadline_expired()
            ):
                verifier_return_code, verifier_timed_out, verifier_error = (
                    _run_logged_command(
                        command=verifier_command,
                        cwd=repo,
                        environment=environment,
                        timeout_seconds=phase_timeout(
                            verifier_timeout_seconds
                        ),
                        log_handle=log_handle,
                        label="verifier",
                    )
                )
                launch_error = verifier_error
            elif (
                not agent_timed_out
                and launch_error is None
                and not checker_timed_out
            ):
                verifier_timed_out = True

    elapsed = (
        time.perf_counter()
        - started
    )

    return SmokeExecution(
        agent_return_code=agent_return_code,
        agent_timed_out=agent_timed_out,
        checker_return_code=checker_return_code,
        checker_timed_out=checker_timed_out,
        verifier_return_code=verifier_return_code,
        verifier_timed_out=verifier_timed_out,
        elapsed_seconds=elapsed,
        launch_error=launch_error,
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
        "verifier_timeout_sec",
        "outer_timeout_sec",
        "qfbench_exit_code",
        "checker_exit_code",
        "runner_timed_out",
        "pytest_exit_code",
        "agent_exit_code",
        "agent_timed_out",
        "checker_timed_out",
        "verifier_timed_out",
        "subprocess_launch_error",
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
        not math.isfinite(args.timeout_buffer)
        or args.timeout_buffer < 0
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

    # qfbench2 is required for a model-backed official smoke run,
    # but the offline crash-recovery sweep can use each unit's
    # public checks/test.sh directly.
    if not args.allow_offline:
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

        if not os.environ.get(
            "MODEL_TOKEN"
        ):
            missing.append(
                "MODEL_TOKEN"
            )

        seed_text = os.environ.get(
            "QFBENCH_SEED"
        )

        if not seed_text:
            missing.append(
                "QFBENCH_SEED"
            )
        else:
            try:
                seed = int(
                    seed_text
                )
            except ValueError:
                raise SystemExit(
                    "QFBENCH_SEED must be a non-negative integer."
                )

            if seed < 0:
                raise SystemExit(
                    "QFBENCH_SEED must be a non-negative integer."
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

    run_name = unique_run_name(
        runs_dir,
        args.run_name,
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
        "inventory_sha256": sha256_file(
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
        "qfbench_seed": os.environ.get(
            "QFBENCH_SEED"
        ),
        "timeout_buffer_sec": (
            args.timeout_buffer
        ),
        "runner_options": {
            "units": list(args.unit),
            "limit": args.limit,
            "resume": args.resume,
            "allow_offline": args.allow_offline,
        },
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
        unit_name = validate_unit_name(
            str(
            inventory_item[
                "unit_dir"
            ]
            )
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

        try:
            card = read_card(
                unit_dir
            )
            agent_timeout, verifier_timeout, outer_timeout = (
                resolve_timeouts(
                    card,
                    unit_name,
                    float(args.timeout_buffer),
                )
            )
        except (OSError, TypeError, ValueError) as exc:
            result = {
                "unit": unit_name,
                "category": inventory_item.get("category"),
                "difficulty": inventory_item.get("difficulty"),
                "status": "harness_error",
                "reward": None,
                "reward_details": str(exc),
                "elapsed_seconds": 0.0,
                "agent_timeout_sec": None,
                "verifier_timeout_sec": None,
                "outer_timeout_sec": None,
                "qfbench_exit_code": None,
                "runner_timed_out": False,
                "pytest_exit_code": None,
                "pytest_total": None,
                "pytest_passed": None,
                "pytest_failed": None,
                "failed_tests": [],
                "deliverables": [],
                "deliverable_count": 0,
                "failed_test_count": 0,
                "output_dir": str(output_dir),
                "log_path": str(log_path),
            }
            write_json(
                result_path,
                result,
            )
            results.append(result)
            print(
                f"  status: {result['status']} ({exc})"
            )
            print()
            continue

        print(
            f"[{index}/{len(selected)}] "
            f"{unit_name}"
        )
        print(
            f"  agent timeout: "
            f"{agent_timeout:.0f}s"
        )

        execution = run_smoke(
            repo=repo,
            unit_dir=unit_dir,
            output_dir=output_dir,
            image=args.image,
            network=(
                "none"
                if args.allow_offline
                else (
                    "qfb2-eval"
                    if eval_network
                    else "none"
                )
            ),
            agent_timeout_seconds=agent_timeout,
            verifier_timeout_seconds=(
                verifier_timeout or 300.0
            ),
            outer_timeout_seconds=outer_timeout,
            log_path=log_path,
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
              return_code=execution.return_code,
              timed_out=execution.timed_out,
            reward=reward,
              pytest_failed=pytest_summary[
                  "pytest_failed"
              ],
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
                                    execution.elapsed_seconds,
                                    6,
            ),
            "agent_timeout_sec": (
                agent_timeout
            ),
            "verifier_timeout_sec": (
                verifier_timeout
            ),
            "outer_timeout_sec": (
                outer_timeout
            ),
            "qfbench_exit_code": (
                  execution.verifier_return_code
              ),
              "checker_exit_code": (
                  execution.checker_return_code
              ),
              "agent_exit_code": (
                  execution.agent_return_code
              ),
              "agent_timed_out": (
                  execution.agent_timed_out
              ),
              "checker_timed_out": (
                  execution.checker_timed_out
              ),
              "verifier_timed_out": (
                  execution.verifier_timed_out
            ),
              "subprocess_launch_error": (
                  execution.launch_error
              ),
            "runner_timed_out": (
                  execution.timed_out
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
              f"{execution.elapsed_seconds:.1f}s"
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

    return benchmark_exit_code(results)


if __name__ == "__main__":
    raise SystemExit(
        main()
    )