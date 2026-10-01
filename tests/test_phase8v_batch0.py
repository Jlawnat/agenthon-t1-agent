from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agent.candidate_prompt import (
    MAX_CANDIDATE_PROMPT_CHARS,
    build_candidate_prompt,
)
from agent.candidate_workspace import CandidateWorkspace
from agent.capability_bridge import (
    CANDIDATE_LIBRARY_FILENAMES,
    compact_planner_data_inspections,
    rank_capabilities,
    relevant_primitive_catalog,
)
from agent.planner import (
    CandidateStrategy,
    MAX_PLANNER_PROMPT_CHARS,
    build_planner_prompt,
)
from agent.planning import build_task_plan
from agent.skill_packs import load_skill_packs
from agent.spec_compiler import compile_specification
from agent.specification import load_task_specification
from agent.task_snapshot import TaskSnapshot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = PROJECT_ROOT / "benchmark" / "public_units_inventory.json"
INVENTORY = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
PUBLIC_REPO = Path(
    os.environ.get(
        "TRACK1_PUBLIC_REPO",
        str(INVENTORY["repository"]),
    )
).expanduser().resolve()
PUBLIC_UNITS_ROOT = PUBLIC_REPO / "units"
PUBLIC_UNIT_NAMES = tuple(
    row["unit_dir"]
    for row in INVENTORY["units"]
)


def test_generated_inventory_matches_current_public_kit() -> None:
    discovered = tuple(
        sorted(
            path.name
            for path in PUBLIC_UNITS_ROOT.iterdir()
            if path.is_dir() and (path / "card.toml").is_file()
        )
    )

    assert INVENTORY["unit_count"] == 86
    assert len(PUBLIC_UNIT_NAMES) == 86
    assert PUBLIC_UNIT_NAMES == discovered
    assert "t1-polars-api-migration" not in PUBLIC_UNIT_NAMES


@pytest.mark.parametrize("unit_name", PUBLIC_UNIT_NAMES)
def test_production_prompts_construct_for_current_public_unit(
    unit_name: str,
) -> None:
    task_dir = PUBLIC_UNITS_ROOT / unit_name
    specification = load_task_specification(task_dir)
    compiled = compile_specification(specification)
    task_plan = build_task_plan(compiled)
    skill_packs = load_skill_packs(task_plan.review_packs)
    inspections = TaskSnapshot.build(task_dir).inspect_data()
    selected = rank_capabilities(
        instruction=specification.instruction_text,
        task_dir=task_dir,
    )
    primitives = relevant_primitive_catalog(selected)

    planner_prompt = build_planner_prompt(
        spec=compiled,
        task_plan=task_plan,
        skill_packs=skill_packs,
        instruction_text=specification.instruction_text,
        data_inspections=compact_planner_data_inspections(inspections),
        selected_capabilities=selected,
        relevant_runtime_primitives=primitives,
    )
    candidate_prompt = build_candidate_prompt(
        instruction_text=specification.instruction_text,
        spec=compiled,
        strategy=CandidateStrategy(
            candidate_id=1,
            approach_name="contract-first",
            implementation_steps=["Map the public contract." * 8] * 8,
            verification_steps=["Validate every required output." * 6] * 8,
            numerical_method=None,
            risks=["Handle the stated numerical edge cases." * 5] * 8,
        ),
        skill_packs=skill_packs,
        data_inspections=inspections,
        selected_capabilities=selected,
        relevant_runtime_primitives=primitives,
    )

    assert planner_prompt
    assert candidate_prompt
    assert len(planner_prompt) <= MAX_PLANNER_PROMPT_CHARS
    assert len(candidate_prompt) <= MAX_CANDIDATE_PROMPT_CHARS


def _verify_candidate_imports(workspace: CandidateWorkspace) -> None:
    script = workspace.source_dir / "import_smoke.py"
    script.write_text(
        "\n".join(
            [
                "import qf_primitives",
                "from offline_common import derivatives",
                "from offline_common import event_study",
                "from offline_common import fixed_income",
                "from offline_common import portfolio",
                "from offline_common import risk",
                "from offline_common import volatility",
                "assert callable(qf_primitives.black_scholes_price)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(workspace.root_dir / "lib")
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=workspace.root_dir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def _verify_workspace_files(
    workspace: CandidateWorkspace,
    task_dir: Path,
) -> None:
    library = workspace.root_dir / "lib"
    assert (library / "qf_primitives.py").is_file()
    assert (library / "qf_primitives.py").stat().st_mode & 0o222 == 0

    common = library / "offline_common"
    assert common.stat().st_mode & 0o222 == 0
    for filename in CANDIDATE_LIBRARY_FILENAMES:
        assert (common / filename).is_file()

    source_data = task_dir / "environment" / "data"
    if source_data.is_dir():
        for source in source_data.rglob("*"):
            if source.is_file():
                assert (
                    workspace.input_dir
                    / "data"
                    / source.relative_to(source_data)
                ).is_file()


@pytest.mark.parametrize("unit_name", PUBLIC_UNIT_NAMES)
def test_candidate_workspace_lifecycle_for_current_public_unit(
    unit_name: str,
    tmp_path: Path,
) -> None:
    task_dir = PUBLIC_UNITS_ROOT / unit_name
    base_dir = tmp_path / "candidate-workspaces"
    workspace = CandidateWorkspace.create(
        base_dir=base_dir,
        candidate_id=1,
        task_dir=task_dir,
    )

    try:
        _verify_workspace_files(workspace, task_dir)
        _verify_candidate_imports(workspace)
    finally:
        workspace.cleanup()

    assert not workspace.root_dir.exists()

    recreated = CandidateWorkspace.create(
        base_dir=base_dir,
        candidate_id=1,
        task_dir=task_dir,
    )

    try:
        _verify_workspace_files(recreated, task_dir)
    finally:
        recreated.cleanup()

    assert not recreated.root_dir.exists()
