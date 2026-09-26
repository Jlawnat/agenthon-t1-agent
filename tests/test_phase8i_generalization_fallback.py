from __future__ import annotations

import json
from pathlib import Path
import zipfile

import agent.main as main_module
from agent.data_inspector import inspect_data_file


def test_tsv_inspection_discovers_schema(tmp_path: Path) -> None:
    path = tmp_path / "holdings.tsv"
    path.write_text(
        "CIK\tCUSIP\tVALUE\n"
        "1001\tABC123\t42\n"
        "1002\tXYZ987\t84\n",
        encoding="utf-8",
    )

    result = inspect_data_file(path)

    assert result["delimiter"] == "\t"
    assert result["columns"] == [
        "CIK",
        "CUSIP",
        "VALUE",
    ]
    assert result["sample_rows"][0]["CIK"] == 1001


def test_json_object_exposes_bounded_scalar_values(tmp_path: Path) -> None:
    path = tmp_path / "terms.json"
    path.write_text(
        json.dumps(
            {
                "station_id": "ABC",
                "month": 1,
                "seed": 42,
                "nested": {
                    "x": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    result = inspect_data_file(path)

    assert result["type"] == "json_object"
    assert result["sample_values"]["station_id"] == "ABC"
    assert result["sample_values"]["seed"] == 42
    assert result["sample_values"]["nested"]["type"] == "object"


def test_zip_inspection_lists_members_without_extracting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "filing.zip"

    with zipfile.ZipFile(
        path,
        "w",
    ) as archive:
        archive.writestr(
            "report.htm",
            "<html><body>Annual report</body></html>",
        )
        archive.writestr(
            "facts.xml",
            "<root><fact>123</fact></root>",
        )

    result = inspect_data_file(path)

    assert result["type"] == "zip_archive"
    assert result["member_count"] == 2

    names = {
        item["name"]
        for item in result["members"]
    }

    assert names == {
        "report.htm",
        "facts.xml",
    }

    assert "report.htm" in result["text_previews"]


def _make_task(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text(
        "# Test task\n",
        encoding="utf-8",
    )
    return task


def _configure_model_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "QFBENCH_SEED",
        "42",
    )
    monkeypatch.setenv(
        "MODEL_ENDPOINT",
        "http://mock-model:8000",
    )
    monkeypatch.setenv(
        "MODEL_NAME",
        "mock-model",
    )
    monkeypatch.setenv(
        "MODEL_TOKEN",
        "test-token",
    )
    monkeypatch.setenv(
        "AGENT_WORK_ROOT",
        str(tmp_path / "work"),
    )


def test_model_mode_uses_house_runtime_even_if_offline_skill_exists(
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

    def fake_offline(**kwargs):
        offline_calls["count"] += 1
        raise AssertionError(
            "model mode must not invoke solve_offline"
        )

    monkeypatch.setattr(
        main_module,
        "solve_offline",
        fake_offline,
    )

    monkeypatch.setattr(
        main_module,
        "build_runtime_components",
        lambda model_client: (
            "front",
            "generator",
            "repairer",
        ),
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

    assert called["task_dir"] == task.resolve()
    assert called["final_output_dir"] == output.resolve()
    assert called["front_half_dependencies"] == "front"
    assert called["generator"] == "generator"
    assert called["repairer"] == "repairer"


def test_model_mode_hands_existing_output_root_to_house_runtime_unchanged(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = _make_task(tmp_path)

    output = tmp_path / "output"
    output.mkdir()

    stale_file = output / "stale.txt"
    stale_file.write_text(
        "stale\n",
        encoding="utf-8",
    )

    output_inode = output.stat().st_ino

    _configure_model_environment(
        monkeypatch,
        tmp_path,
    )

    offline_calls = {"count": 0}

    def fake_offline(**kwargs):
        offline_calls["count"] += 1
        raise AssertionError(
            "model mode must not invoke solve_offline"
        )

    monkeypatch.setattr(
        main_module,
        "solve_offline",
        fake_offline,
    )

    monkeypatch.setattr(
        main_module,
        "build_runtime_components",
        lambda model_client: (
            "front",
            "generator",
            "repairer",
        ),
    )

    called = {}

    def fake_solve_task(**kwargs):
        # main.py must not remove, replace, or clean the bind-mount-style
        # output root before handing control to the House runtime.
        assert output.is_dir()
        assert output.stat().st_ino == output_inode
        assert stale_file.read_text(
            encoding="utf-8",
        ) == "stale\n"

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

    assert called["final_output_dir"] == output.resolve()

    assert output.is_dir()
    assert output.stat().st_ino == output_inode
    assert stale_file.is_file()


def test_model_mode_enters_house_runtime_directly(
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

    def fake_offline(**kwargs):
        offline_calls["count"] += 1
        raise AssertionError(
            "model mode must not invoke solve_offline"
        )

    monkeypatch.setattr(
        main_module,
        "solve_offline",
        fake_offline,
    )

    monkeypatch.setattr(
        main_module,
        "build_runtime_components",
        lambda model_client: (
            "front",
            "generator",
            "repairer",
        ),
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
    assert called["task_dir"] == task.resolve()
    assert called["final_output_dir"] == output.resolve()
    assert called["front_half_dependencies"] == "front"
    assert called["generator"] == "generator"
    assert called["repairer"] == "repairer"
