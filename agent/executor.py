from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import math
import os
import selectors
import signal
import subprocess
import sys
import time


MAX_CAPTURE_CHARS = 50_000
MAX_CAPTURE_BYTES = 50_000
_CAPTURE_CHUNK_BYTES = 8_192
_TRUNCATION_MARKER = "\n...[truncated]..."
_POST_EXIT_DRAIN_SECONDS = 0.5

SAFE_PARENT_ENV_KEYS = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
)

SAFE_CANDIDATE_OVERRIDE_KEYS = frozenset(
    {
        "QFBENCH_SEED",
        "PYTHONHASHSEED",
    }
)

_GUARD_DIR_NAME = ".candidate_guard"
_HOME_DIR_NAME = ".candidate_home"
_TMP_DIR_NAME = ".candidate_tmp"

_GUARD_SOURCE = r"""
from __future__ import annotations

import os
import sys


def _normalise(value):
    if isinstance(value, int):
        return None

    try:
        raw = os.fspath(value)
    except TypeError:
        return None

    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)

    return os.path.realpath(
        os.path.abspath(raw)
    )


def _within(path, root):
    return (
        path == root
        or path.startswith(
            root + os.sep
        )
    )


_ROOT = _normalise(
    os.environ[
        "AGENT_CANDIDATE_ROOT"
    ]
)

_OUTPUT = _normalise(
    os.environ[
        "AGENT_CANDIDATE_OUTPUT"
    ]
)

_TMP = _normalise(
    os.environ[
        "AGENT_CANDIDATE_TMP"
    ]
)

_HOME = _normalise(
    os.environ[
        "AGENT_CANDIDATE_HOME"
    ]
)

_ALLOWED_WRITE_ROOTS = tuple(
    root
    for root in (
        _OUTPUT,
        _TMP,
        _HOME,
    )
    if root
)


def _write_allowed(value):
    path = _normalise(value)

    if path is None:
        return True

    return any(
        _within(
            path,
            root,
        )
        for root in _ALLOWED_WRITE_ROOTS
    )


def _deny_write(value):
    if not _write_allowed(value):
        raise PermissionError(
            "candidate filesystem write "
            "outside the allowed workspace "
            "is disabled"
        )


def _open_is_write(mode, flags):
    if isinstance(mode, str):
        if any(
            token in mode
            for token in (
                "w",
                "a",
                "x",
                "+",
            )
        ):
            return True

    if isinstance(flags, int):
        write_mask = (
            os.O_WRONLY
            | os.O_RDWR
            | os.O_CREAT
            | os.O_TRUNC
            | os.O_APPEND
        )

        if flags & write_mask:
            return True

    return False


_BLOCKED_PROCESS_EVENTS = {
    "subprocess.Popen",
    "os.system",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.fork",
    "os.forkpty",
    "os.exec",
}

_BLOCKED_NETWORK_PREFIXES = (
    "socket.",
)

_SINGLE_PATH_MUTATIONS = {
    "os.remove",
    "os.unlink",
    "os.mkdir",
    "os.rmdir",
    "os.chmod",
    "os.chown",
    "os.truncate",
    "os.utime",
}

_TWO_PATH_MUTATIONS = {
    "os.rename",
    "os.replace",
    "os.link",
    "os.symlink",
}


def _audit(event, args):
    if event in _BLOCKED_PROCESS_EVENTS:
        raise PermissionError(
            "candidate child-process "
            "creation is disabled"
        )

    if event.startswith(
        _BLOCKED_NETWORK_PREFIXES
    ):
        raise PermissionError(
            "candidate network access "
            "is disabled"
        )

    if event == "open":
        path = (
            args[0]
            if len(args) >= 1
            else None
        )
        mode = (
            args[1]
            if len(args) >= 2
            else None
        )
        flags = (
            args[2]
            if len(args) >= 3
            else None
        )

        if _open_is_write(
            mode,
            flags,
        ):
            _deny_write(path)

        return

    if event in _SINGLE_PATH_MUTATIONS:
        if args:
            _deny_write(
                args[0]
            )
        return

    if event in _TWO_PATH_MUTATIONS:
        if len(args) >= 1:
            _deny_write(
                args[0]
            )

        if len(args) >= 2:
            _deny_write(
                args[1]
            )


sys.addaudithook(
    _audit
)
""".lstrip()


@dataclass
class ExecutionResult:
    command: list[str]
    return_code: int | None
    stdout: str
    stderr: str
    runtime_seconds: float
    timed_out: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _truncate(text: str) -> str:
    if len(text) <= MAX_CAPTURE_CHARS:
        return text

    return (
        text[:MAX_CAPTURE_CHARS]
        + _TRUNCATION_MARKER
    )


class _BoundedCaptureBuffer:
    """Retain only a fixed prefix while excess pipe data is discarded."""

    def __init__(
        self,
        max_bytes: int = MAX_CAPTURE_BYTES,
    ) -> None:
        self.max_bytes = max_bytes
        self._buffer = bytearray()
        self.truncated = False

    def feed(
        self,
        chunk: bytes,
    ) -> None:
        if not chunk:
            return

        remaining = (
            self.max_bytes
            - len(self._buffer)
        )

        if remaining > 0:
            self._buffer.extend(
                chunk[:remaining]
            )

        if len(chunk) > remaining:
            self.truncated = True

    @property
    def stored_bytes(self) -> int:
        return len(self._buffer)

    def text(self) -> str:
        text = bytes(
            self._buffer
        ).decode(
            "utf-8",
            errors="replace",
        )

        if len(text) > MAX_CAPTURE_CHARS:
            text = text[:MAX_CAPTURE_CHARS]
            self.truncated = True

        if self.truncated:
            return text + _TRUNCATION_MARKER

        return text


def _kill_process_group(
    process: subprocess.Popen[bytes],
) -> None:
    """Kill the whole candidate process group when possible."""

    if os.name == "posix":
        try:
            os.killpg(
                process.pid,
                signal.SIGKILL,
            )
            return
        except ProcessLookupError:
            return
        except OSError:
            pass

    try:
        process.kill()
    except ProcessLookupError:
        pass


def _close_registered_streams(
    selector: selectors.BaseSelector,
) -> None:
    for key in list(
        selector.get_map().values()
    ):
        try:
            selector.unregister(
                key.fileobj
            )
        except Exception:
            pass

        try:
            key.fileobj.close()
        except Exception:
            pass


def _run_with_bounded_capture(
    *,
    command: list[str],
    cwd: Path,
    timeout: float,
    env: Mapping[str, str],
    preexec_fn,
) -> tuple[int | None, str, str, bool]:
    """Execute with bounded in-memory capture and process-group cleanup."""

    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        env=dict(env),
        bufsize=0,
        text=False,
        start_new_session=(
            os.name == "posix"
        ),
        preexec_fn=preexec_fn,
    )

    if (
        process.stdout is None
        or process.stderr is None
    ):
        _kill_process_group(process)
        raise RuntimeError(
            "Candidate capture pipes were not created."
        )

    selector = selectors.DefaultSelector()
    stdout_buffer = _BoundedCaptureBuffer()
    stderr_buffer = _BoundedCaptureBuffer()

    streams = {
        process.stdout: stdout_buffer,
        process.stderr: stderr_buffer,
    }

    for stream in streams:
        if os.name == "posix":
            os.set_blocking(
                stream.fileno(),
                False,
            )

        selector.register(
            stream,
            selectors.EVENT_READ,
        )

    deadline = time.monotonic() + timeout
    timed_out = False
    parent_exit_time: float | None = None
    forced_group_cleanup = False

    try:
        while selector.get_map():
            now = time.monotonic()

            if (
                not timed_out
                and process.poll() is None
                and now >= deadline
            ):
                timed_out = True
                _kill_process_group(process)

            if (
                process.poll() is not None
                and parent_exit_time is None
            ):
                parent_exit_time = now

            if (
                parent_exit_time is not None
                and not forced_group_cleanup
                and (
                    now - parent_exit_time
                ) >= _POST_EXIT_DRAIN_SECONDS
            ):
                forced_group_cleanup = True
                _kill_process_group(process)

            events = selector.select(
                timeout=0.05
            )

            for key, _ in events:
                stream = key.fileobj
                buffer = streams[stream]

                try:
                    chunk = os.read(
                        stream.fileno(),
                        _CAPTURE_CHUNK_BYTES,
                    )
                except BlockingIOError:
                    continue
                except OSError:
                    chunk = b""

                if chunk:
                    buffer.feed(chunk)
                    continue

                try:
                    selector.unregister(stream)
                except Exception:
                    pass

                try:
                    stream.close()
                except Exception:
                    pass

            if (
                process.poll() is not None
                and not selector.get_map()
            ):
                break

        if process.poll() is None:
            _kill_process_group(process)

        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            process.wait(timeout=1.0)

    finally:
        _close_registered_streams(selector)
        selector.close()

        for stream in (
            process.stdout,
            process.stderr,
        ):
            try:
                stream.close()
            except Exception:
                pass

    return (
        None if timed_out else process.returncode,
        stdout_buffer.text(),
        stderr_buffer.text(),
        timed_out,
    )


def _is_within(
    path: Path,
    root: Path,
) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False

    return True


def _validate_timeout(
    timeout_seconds: float,
) -> float:
    if (
        isinstance(
            timeout_seconds,
            bool,
        )
        or not isinstance(
            timeout_seconds,
            (int, float),
        )
    ):
        raise ValueError(
            "timeout_seconds must be a positive number."
        )

    timeout = float(
        timeout_seconds
    )

    if (
        not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError(
            "timeout_seconds must be a positive finite number."
        )

    return timeout


def _prepare_guard(
    cwd: Path,
) -> tuple[
    Path,
    Path,
    Path,
]:
    guard_dir = (
        cwd
        / _GUARD_DIR_NAME
    )
    home_dir = (
        cwd
        / _HOME_DIR_NAME
    )
    tmp_dir = (
        cwd
        / _TMP_DIR_NAME
    )

    guard_dir.mkdir(
        parents=False,
        exist_ok=True,
    )
    home_dir.mkdir(
        parents=False,
        exist_ok=True,
    )
    tmp_dir.mkdir(
        parents=False,
        exist_ok=True,
    )

    sitecustomize_path = (
        guard_dir
        / "sitecustomize.py"
    )

    sitecustomize_path.write_text(
        _GUARD_SOURCE,
        encoding="utf-8",
    )

    return (
        guard_dir.resolve(),
        home_dir.resolve(),
        tmp_dir.resolve(),
    )


def _build_environment(
    *,
    cwd: Path,
    env_overrides:
        Mapping[str, str]
        | None = None,
) -> dict[str, str]:
    """
    Build a deliberately minimal environment for generated
    candidate subprocesses.

    Candidate code does not inherit model configuration,
    credentials, API keys, benchmark secrets, or arbitrary
    parent variables.
    """

    cwd = cwd.resolve()

    (
        guard_dir,
        home_dir,
        tmp_dir,
    ) = _prepare_guard(
        cwd
    )

    env: dict[str, str] = {}

    for key in SAFE_PARENT_ENV_KEYS:
        value = os.environ.get(
            key
        )

        if value is not None:
            env[key] = value

    env.update(
        {
            "HOME": str(
                home_dir
            ),
            "TMPDIR": str(
                tmp_dir
            ),
            "TMP": str(
                tmp_dir
            ),
            "TEMP": str(
                tmp_dir
            ),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join(
                [
                    str(
                        guard_dir
                    ),
                    str(
                        cwd
                        / "lib"
                    ),
                ]
            ),
            # Candidate code should resolve task paths from
            # these roots rather than hardcoding global mounts.
            "INPUT_DIR": str(
                cwd / "input"
            ),
            "OUTPUT_DIR": str(
                cwd / "output"
            ),
            # Internal values consumed by sitecustomize.
            "AGENT_CANDIDATE_ROOT": str(
                cwd
            ),
            "AGENT_CANDIDATE_OUTPUT": str(
                cwd / "output"
            ),
            "AGENT_CANDIDATE_TMP": str(
                tmp_dir
            ),
            "AGENT_CANDIDATE_HOME": str(
                home_dir
            ),
        }
    )

    if env_overrides is not None:
        unsupported = (
            set(env_overrides)
            - SAFE_CANDIDATE_OVERRIDE_KEYS
        )

        if unsupported:
            names = ", ".join(
                sorted(
                    str(name)
                    for name
                    in unsupported
                )
            )

            raise ValueError(
                "Unsafe candidate environment "
                f"override(s): {names}"
            )

        for key, value in (
            env_overrides.items()
        ):
            env[str(key)] = str(
                value
            )

    return env


def _resource_limit_preexec(
    timeout_seconds: float,
):
    """
    Apply only non-disruptive local child-process limits.

    Do NOT impose RLIMIT_NPROC or RLIMIT_CPU here:
    - RLIMIT_NPROC also constrains native-library thread creation
      and can break pandas/pyarrow/numpy workloads.
    - RLIMIT_CPU counts CPU time rather than wall time and can
      prematurely kill legitimate multithreaded quant code.

    The subprocess wall timeout remains enforced here. The
    competition container/harness is authoritative for CPU,
    memory, and PID limits.
    """

    _ = timeout_seconds

    if os.name != "posix":
        return None

    try:
        import resource
    except ImportError:
        return None

    def apply_limits() -> None:
        resource.setrlimit(
            resource.RLIMIT_CORE,
            (0, 0),
        )

        resource_id = getattr(
            resource,
            "RLIMIT_NOFILE",
            None,
        )

        if resource_id is None:
            return

        current_soft, current_hard = (
            resource.getrlimit(
                resource_id
            )
        )

        target = 256

        if (
            current_hard
            != resource.RLIM_INFINITY
        ):
            target = min(
                target,
                int(current_hard),
            )

        # Never increase an already tighter inherited soft limit.
        if (
            current_soft
            != resource.RLIM_INFINITY
        ):
            target = min(
                target,
                int(current_soft),
            )

        resource.setrlimit(
            resource_id,
            (
                target,
                target,
            ),
        )

    return apply_limits


def run_python_candidate(
    script_path: Path,
    *,
    cwd: Path,
    timeout_seconds: float = 120,
    env_overrides:
        Mapping[str, str]
        | None = None,
) -> ExecutionResult:
    timeout = _validate_timeout(
        timeout_seconds
    )

    raw_script_path = Path(script_path)
    raw_cwd = Path(cwd)

    if raw_cwd.is_symlink():
        raise ValueError(
            "Candidate working directory must not be a symlink."
        )

    cwd = raw_cwd.resolve()

    if not cwd.exists():
        raise FileNotFoundError(
            "Candidate working directory not found: "
            f"{cwd}"
        )

    if not cwd.is_dir():
        raise NotADirectoryError(
            "Candidate working directory is not a directory: "
            f"{cwd}"
        )

    if raw_script_path.is_symlink():
        raise ValueError(
            "Candidate script must not be a symlink."
        )

    script_path = raw_script_path.resolve()

    if not script_path.exists():
        raise FileNotFoundError(
            f"Candidate script not found: {script_path}"
        )

    if not script_path.is_file():
        raise ValueError(
            "Candidate script must be a regular file."
        )

    if not _is_within(script_path, cwd):
        raise ValueError(
            "Candidate script escapes the candidate workspace."
        )

    command = [
        sys.executable,
        str(script_path),
    ]

    environment = _build_environment(
        cwd=cwd,
        env_overrides=env_overrides,
    )

    preexec_fn = _resource_limit_preexec(
        timeout
    )

    start = time.perf_counter()

    (
        return_code,
        stdout,
        stderr,
        timed_out,
    ) = _run_with_bounded_capture(
        command=command,
        cwd=cwd,
        timeout=timeout,
        env=environment,
        preexec_fn=preexec_fn,
    )

    runtime = time.perf_counter() - start

    return ExecutionResult(
        command=command,
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        runtime_seconds=runtime,
        timed_out=timed_out,
    )