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


def test_model_mode_prefers_complete_offline_solution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = _make_task(tmp_path)
    output = tmp_path / "output"
    _configure_model_environment(
        monkeypatch,
        tmp_path,
    )

    def fake_offline(
        *,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> str:
        assert task_dir == task.resolve()
        assert seed == 42
        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        (out_dir / "answer.json").write_text(
            '{"ok": true}\n',
            encoding="utf-8",
        )
        return "generic-deterministic-skill"

    monkeypatch.setattr(
        main_module,
        "solve_offline",
        fake_offline,
    )

    def should_not_build_model(*args, **kwargs):
        raise AssertionError(
            "model runtime should not be entered"
        )

    monkeypatch.setattr(
        main_module,
        "build_runtime_components",
        should_not_build_model,
    )

    main_module.solve(
        task,
        output,
    )

    assert (
        output
        / "answer.json"
    ).is_file()



def test_model_mode_preserves_existing_output_root(
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
    stale_dir = output / "stale-dir"
    stale_dir.mkdir()
    (stale_dir / "old.txt").write_text(
        "old\n",
        encoding="utf-8",
    )

    output_inode = output.stat().st_ino

    _configure_model_environment(
        monkeypatch,
        tmp_path,
    )

    def fake_offline(
        *,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> str:
        assert task_dir == task.resolve()
        assert seed == 42

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        (out_dir / "answer.json").write_text(
            '{"ok": true}\n',
            encoding="utf-8",
        )
        nested = out_dir / "nested"
        nested.mkdir()
        (nested / "result.txt").write_text(
            "new\n",
            encoding="utf-8",
        )
        return "generic-deterministic-skill"

    monkeypatch.setattr(
        main_module,
        "solve_offline",
        fake_offline,
    )

    def should_not_build_model(*args, **kwargs):
        raise AssertionError(
            "model runtime should not be entered"
        )

    monkeypatch.setattr(
        main_module,
        "build_runtime_components",
        should_not_build_model,
    )

    original_rmtree = main_module.shutil.rmtree
    rmtree_calls = []

    def guarded_rmtree(path, *args, **kwargs):
        resolved = Path(path).resolve()
        rmtree_calls.append(resolved)

        if resolved == output.resolve():
            raise AssertionError(
                "final output root must never be removed"
            )

        return original_rmtree(
            path,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        main_module.shutil,
        "rmtree",
        guarded_rmtree,
    )

    main_module.solve(
        task,
        output,
    )

    assert output.is_dir()
    assert output.stat().st_ino == output_inode

    assert not stale_file.exists()
    assert not stale_dir.exists()

    assert (
        output
        / "answer.json"
    ).read_text(
        encoding="utf-8",
    ) == '{"ok": true}\n'

    assert (
        output
        / "nested"
        / "result.txt"
    ).read_text(
        encoding="utf-8",
    ) == "new\n"

    assert output.resolve() not in rmtree_calls

def test_model_mode_falls_back_after_offline_miss(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = _make_task(tmp_path)
    output = tmp_path / "output"
    _configure_model_environment(
        monkeypatch,
        tmp_path,
    )

    def miss_offline(**kwargs):
        raise RuntimeError(
            "no deterministic skill"
        )

    monkeypatch.setattr(
        main_module,
        "solve_offline",
        miss_offline,
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

    assert called[
        "task_dir"
    ] == task.resolve()
    assert called[
        "final_output_dir"
    ] == output.resolve()
    assert called[
        "front_half_dependencies"
    ] == "front"
    assert called[
        "generator"
    ] == "generator"
    assert called[
        "repairer"
    ] == "repairer"
