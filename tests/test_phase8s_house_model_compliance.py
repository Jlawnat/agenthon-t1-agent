from __future__ import annotations

from pathlib import Path

import agent.main as main_module


def _make_task(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    task.mkdir()

    (task / "instruction.md").write_text(
        "# Synthetic task\n"
        "Solve this task from the provided inputs.\n",
        encoding="utf-8",
    )

    return task


def _configure_model_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("QFBENCH_SEED", "42")
    monkeypatch.setenv("MODEL_ENDPOINT", "http://mock-model:8000")
    monkeypatch.setenv("MODEL_NAME", "mock-model")
    monkeypatch.setenv("MODEL_TOKEN", "test-token")
    monkeypatch.setenv(
        "AGENT_WORK_ROOT",
        str(tmp_path / "work"),
    )


def test_model_mode_never_short_circuits_through_offline_solver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = _make_task(tmp_path)
    output = tmp_path / "output"

    _configure_model_environment(
        monkeypatch,
        tmp_path,
    )

    offline_calls = {"count": 0}

    def fake_offline(
        *,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> str:
        offline_calls["count"] += 1

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (out_dir / "answer.json").write_text(
            '{"prepared": true}\n',
            encoding="utf-8",
        )

        return "prepared-offline-solution"

    monkeypatch.setattr(
        main_module,
        "solve_offline",
        fake_offline,
    )

    components_built = {}

    def fake_build_runtime_components(model_client):
        components_built["called"] = True

        return (
            "front",
            "generator",
            "repairer",
        )

    monkeypatch.setattr(
        main_module,
        "build_runtime_components",
        fake_build_runtime_components,
    )

    called = {}

    def fake_solve_task(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(
        main_module,
        "solve_task",
        fake_solve_task,
    )

    main_module.solve(
        task,
        output,
    )

    assert offline_calls["count"] == 0
    assert components_built["called"] is True

    assert called["task_dir"] == task.resolve()
    assert called["final_output_dir"] == output.resolve()
    assert called["front_half_dependencies"] == "front"
    assert called["generator"] == "generator"
    assert called["repairer"] == "repairer"


def test_offline_mode_still_uses_offline_solver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = _make_task(tmp_path)
    output = tmp_path / "output"

    monkeypatch.setenv("QFBENCH_SEED", "42")
    monkeypatch.setenv(
        "AGENT_WORK_ROOT",
        str(tmp_path / "work"),
    )

    monkeypatch.delenv(
        "MODEL_ENDPOINT",
        raising=False,
    )
    monkeypatch.delenv(
        "MODEL_NAME",
        raising=False,
    )
    monkeypatch.delenv(
        "MODEL_TOKEN",
        raising=False,
    )

    called = {}

    def fake_offline(
        *,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> str:
        called["task_dir"] = task_dir
        called["out_dir"] = out_dir
        called["seed"] = seed

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (out_dir / "answer.json").write_text(
            '{"ok": true}\n',
            encoding="utf-8",
        )

        return "test-skill"

    monkeypatch.setattr(
        main_module,
        "solve_offline",
        fake_offline,
    )

    def forbidden_model(*args, **kwargs):
        raise AssertionError(
            "model runtime should not be entered in offline mode"
        )

    monkeypatch.setattr(
        main_module,
        "build_runtime_components",
        forbidden_model,
    )

    main_module.solve(
        task,
        output,
    )

    assert called["task_dir"] == task.resolve()
    assert called["out_dir"] == output.resolve()
    assert called["seed"] == 42

    assert (
        output
        / "answer.json"
    ).is_file()
