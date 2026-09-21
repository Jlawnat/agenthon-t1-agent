from pathlib import Path

from agent.offline_runtime import (
    _offline_temp_root,
)


def test_offline_temp_root_uses_agent_work_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    requested = tmp_path / "agent-work"

    monkeypatch.setenv(
        "AGENT_WORK_ROOT",
        str(requested),
    )

    result = _offline_temp_root()

    assert result == (
        requested.resolve() / "offline"
    )
    assert result.is_dir()
    assert result.parent == requested.resolve()
