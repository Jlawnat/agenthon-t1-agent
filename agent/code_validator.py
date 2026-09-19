from __future__ import annotations

import ast
import re


_FORBIDDEN_TEXT_FRAGMENTS = (
    "checks/test_outputs.py",
    "checks/test.sh",
    "checks/reference_data",
    "/logs/verifier",
    "reward.json",
    "reward.txt",
    "pytest_report.json",
)

_FORBIDDEN_IMPORT_ROOTS = {
    "socket",
    "http",
    "urllib",
    "requests",
    "aiohttp",
    "ftplib",
    "smtplib",
    "telnetlib",
    "paramiko",
    "websockets",
    "subprocess",
    "multiprocessing",
    "ctypes",
    "cffi",
    "importlib",
}

_FORBIDDEN_BUILTIN_CALLS = {
    "__import__",
    "eval",
    "exec",
    "compile",
}

_FORBIDDEN_OS_CALLS = {
    "system",
    "popen",
    "fork",
    "forkpty",
    "posix_spawn",
    "posix_spawnp",
    "startfile",
    "chmod",
    "chown",
    "fchmod",
    "fchown",
    "lchmod",
}

_FORBIDDEN_OS_CALL_PREFIXES = (
    "exec",
    "spawn",
)

_FORBIDDEN_ASYNCIO_CALLS = {
    "create_subprocess_exec",
    "create_subprocess_shell",
}


def clean_model_code(text: str) -> str:
    # Recover executable Python from common model wrappers.
    code = str(text).strip()

    code = re.sub(
        r"(?is)<think>.*?</think>",
        "",
        code,
    ).strip()

    fenced = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)```",
        code,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced:
        code = max(
            fenced,
            key=len,
        ).strip()

    elif code.startswith("```"):
        lines = code.splitlines()

        if lines:
            lines = lines[1:]

        code = "\n".join(lines).strip()

    return code


def _root_module(
    module_name: str,
) -> str:
    return module_name.split(".", 1)[0].strip()


class _SecurityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.module_aliases: dict[str, str] = {}
        self.direct_imports: dict[
            str,
            tuple[str, str],
        ] = {}

    def _fail(
        self,
        message: str,
        node: ast.AST,
    ) -> None:
        line = getattr(node, "lineno", None)
        suffix = (
            f" on line {line}"
            if line is not None
            else ""
        )

        raise ValueError(
            "Candidate code violates "
            f"the execution policy{suffix}: {message}"
        )

    def visit_Import(
        self,
        node: ast.Import,
    ) -> None:
        for alias in node.names:
            root = _root_module(alias.name)

            if root in _FORBIDDEN_IMPORT_ROOTS:
                self._fail(
                    f"forbidden import {alias.name!r}",
                    node,
                )

            self.module_aliases[
                alias.asname or root
            ] = root

        self.generic_visit(node)

    def visit_ImportFrom(
        self,
        node: ast.ImportFrom,
    ) -> None:
        module = node.module or ""
        root = _root_module(module)

        if root in _FORBIDDEN_IMPORT_ROOTS:
            self._fail(
                f"forbidden import {module!r}",
                node,
            )

        for alias in node.names:
            local_name = alias.asname or alias.name

            self.direct_imports[
                local_name
            ] = (
                root,
                alias.name,
            )

            if (
                root == "os"
                and (
                    alias.name in _FORBIDDEN_OS_CALLS
                    or alias.name.startswith(
                        _FORBIDDEN_OS_CALL_PREFIXES
                    )
                )
            ):
                self._fail(
                    "forbidden process or filesystem-control "
                    f"import os.{alias.name}",
                    node,
                )

            if (
                root == "asyncio"
                and alias.name in _FORBIDDEN_ASYNCIO_CALLS
            ):
                self._fail(
                    "forbidden child-process import "
                    f"asyncio.{alias.name}",
                    node,
                )

        self.generic_visit(node)

    def visit_Call(
        self,
        node: ast.Call,
    ) -> None:
        func = node.func

        if isinstance(func, ast.Name):
            name = func.id

            if name in _FORBIDDEN_BUILTIN_CALLS:
                self._fail(
                    f"forbidden dynamic-code call {name}()",
                    node,
                )

            imported = self.direct_imports.get(name)

            if imported is not None:
                root, original = imported

                if (
                    root == "os"
                    and (
                        original in _FORBIDDEN_OS_CALLS
                        or original.startswith(
                            _FORBIDDEN_OS_CALL_PREFIXES
                        )
                    )
                ):
                    self._fail(
                        "forbidden process or filesystem-control "
                        f"call os.{original}()",
                        node,
                    )

                if (
                    root == "asyncio"
                    and original in _FORBIDDEN_ASYNCIO_CALLS
                ):
                    self._fail(
                        "forbidden child-process call "
                        f"asyncio.{original}()",
                        node,
                    )

        elif isinstance(func, ast.Attribute):
            attr = func.attr
            value = func.value

            if isinstance(value, ast.Name):
                root = self.module_aliases.get(
                    value.id,
                    value.id,
                )

                if (
                    root == "os"
                    and (
                        attr in _FORBIDDEN_OS_CALLS
                        or attr.startswith(
                            _FORBIDDEN_OS_CALL_PREFIXES
                        )
                    )
                ):
                    self._fail(
                        "forbidden process or filesystem-control "
                        f"call os.{attr}()",
                        node,
                    )

                if (
                    root == "asyncio"
                    and attr in _FORBIDDEN_ASYNCIO_CALLS
                ):
                    self._fail(
                        "forbidden child-process call "
                        f"asyncio.{attr}()",
                        node,
                    )

                if (
                    root == "builtins"
                    and attr in _FORBIDDEN_BUILTIN_CALLS
                ):
                    self._fail(
                        "forbidden dynamic-code call "
                        f"builtins.{attr}()",
                        node,
                    )

        self.generic_visit(node)


def validate_python_code(
    text: str,
) -> str:
    code = clean_model_code(text)

    if not code:
        raise ValueError(
            "Model returned empty candidate code."
        )

    try:
        tree = ast.parse(code)

    except SyntaxError as exc:
        raise ValueError(
            f"Candidate code is not valid Python: {exc}"
        ) from exc

    lowered = code.lower()

    for fragment in _FORBIDDEN_TEXT_FRAGMENTS:
        if fragment.lower() in lowered:
            raise ValueError(
                "Candidate code references forbidden "
                f"benchmark infrastructure: {fragment}"
            )

    _SecurityVisitor().visit(tree)

    return code
