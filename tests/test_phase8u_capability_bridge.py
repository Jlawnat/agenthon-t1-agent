from __future__ import annotations

from dataclasses import asdict, replace
import importlib
import inspect
import json
from pathlib import Path
import pickle
import re
import stat

import pandas as pd
import pytest

from agent.candidate_prompt import (
    MAX_CANDIDATE_PROMPT_CHARS,
    build_candidate_prompt,
)
from agent.candidate_runner import run_candidate
from agent.candidate_workspace import CandidateWorkspace
from agent.capability_bridge import (
    CAPABILITY_CATALOG,
    CANDIDATE_LIBRARY_EXPORTS,
    MAX_CAPABILITY_PROMPT_GROWTH_CHARS,
    RankedCapability,
    compact_planner_data_inspections,
    rank_capabilities,
    relevant_primitive_catalog,
)
from agent.planner import (
    MAX_PLANNER_PROMPT_CHARS,
    build_planner_prompt,
)
from agent.runtime_adapters import RuntimePlan


class _Payload:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def to_dict(self) -> dict:
        return dict(self.payload)


class _TaskPlan(_Payload):
    candidate_count = 2


def _task(tmp_path: Path, instruction: str) -> Path:
    task = tmp_path / "task"
    data = task / "environment" / "data"
    data.mkdir(parents=True)
    (task / "instruction.md").write_text(
        instruction,
        encoding="utf-8",
    )
    return task


def _write_ohlc(path: Path) -> None:
    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
        }
    ).to_csv(path, index=False)


def test_ranking_combines_instruction_and_ohlc_fingerprint(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Compare Parkinson and Yang-Zhang OHLC volatility estimators.",
    )
    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
        }
    ).to_csv(
        task / "environment" / "data" / "prices.csv",
        index=False,
    )

    ranked = rank_capabilities(
        instruction=(task / "instruction.md").read_text(),
        task_dir=task,
    )

    assert ranked
    assert ranked[0].descriptor.capability_id == "ohlc-volatility"
    assert ranked[0].semantic_score > 0
    assert ranked[0].data_score > 0


def test_schema_alone_does_not_force_a_capability(tmp_path: Path) -> None:
    task = _task(tmp_path, "Analyse the supplied observations.")
    pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
        }
    ).to_csv(
        task / "environment" / "data" / "observations.csv",
        index=False,
    )

    assert rank_capabilities(
        instruction="Analyse the supplied observations.",
        task_dir=task,
    ) == ()


def test_asian_option_does_not_misroute_to_ohlc_volatility(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Price an Asian option with Levy and Curran approximations.",
    )
    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
        }
    ).to_csv(
        task / "environment" / "data" / "prices.csv",
        index=False,
    )

    ranked = rank_capabilities(
        instruction=(task / "instruction.md").read_text(),
        task_dir=task,
    )

    assert all(
        item.descriptor.capability_id != "ohlc-volatility"
        for item in ranked
    )


def test_capability_descriptors_are_generic_and_compliance_bounded() -> None:
    assert {
        item.import_path
        for item in CAPABILITY_CATALOG
    } == {
        "offline_common.derivatives",
        "offline_common.event_study",
        "offline_common.execution",
        "offline_common.fixed_income",
        "offline_common.invariants",
        "offline_common.ledger",
        "offline_common.microstructure",
        "offline_common.panel",
        "offline_common.volume_scheduling",
        "offline_common.portfolio",
        "offline_common.risk",
        "offline_common.schedules",
        "offline_common.asian_options",
        "offline_common.bonds",
        "offline_common.cap_floor",
        "offline_common.cliquet",
        "offline_common.copula_fitting",
        "offline_common.credit_migration",
        "offline_common.first_passage",
        "offline_common.merton_jump_diffusion",
        "offline_common.ou_jump",
        "offline_common.regime_black_litterman",
        "offline_common.factor_portfolio",
        "offline_common.intraday_variation",
        "offline_common.fx",
        "offline_common.lookback_options",
        "offline_common.finite_difference",
        "offline_common.monte_carlo_greeks",
        "offline_common.rate_curves",
        "offline_common.statistics",
        "offline_common.variance_swap",
        "offline_common.volatility",
    }

    forbidden_fragments = (
        "output_dir",
        "out_dir",
        "required output",
        "output contract",
        "deliverable",
        "skill.solve",
        "skill.matches",
        "_solve",
        "run_",
        "workflow",
        ".csv",
        ".json",
        ".html",
        "/output",
    )
    unit_id_patterns = (
        re.compile(r"\bunit[-_ ]?\d+\b", re.I),
        re.compile(r"\b(?:dev|test|task)[-_]\d+\b", re.I),
        re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{12}\b",
            re.I,
        ),
    )

    for descriptor in CAPABILITY_CATALOG:
        payload = json.dumps(
            descriptor.to_prompt_dict(),
            sort_keys=True,
        ).lower()
        assert not any(term in payload for term in forbidden_fragments)
        assert not any(pattern.search(payload) for pattern in unit_id_patterns)
        assert all("(" in api and ")" in api for api in descriptor.api_signatures)

        module = importlib.import_module(
            "agent." + descriptor.import_path
        )
        filename = descriptor.import_path.rsplit(".", 1)[-1] + ".py"
        approved_exports = CANDIDATE_LIBRARY_EXPORTS[filename]
        for api in descriptor.api_signatures:
            api_name = api.split("(", 1)[0]
            assert api_name in approved_exports
            if " -> " not in api:
                continue
            advertised_return = api.rsplit(" -> ", 1)[-1]
            actual_return = inspect.signature(
                getattr(module, api_name)
            ).return_annotation
            assert isinstance(actual_return, str)

            def normalize(annotation: str) -> str:
                return re.sub(
                    r"\s+",
                    "",
                    annotation.replace("pd.", "").replace("np.", ""),
                )

            assert normalize(advertised_return) == normalize(actual_return)


def test_planner_and_candidate_prompts_prioritize_selected_context(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Estimate historical VaR and expected shortfall.",
    )
    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "symbol": ["AAA", "AAA"],
            "close": [100.0, 99.0],
        }
    ).to_csv(
        task / "environment" / "data" / "market.csv",
        index=False,
    )
    selected = rank_capabilities(
        instruction="Estimate historical VaR and expected shortfall.",
        task_dir=task,
    )
    relevant = relevant_primitive_catalog(selected)
    inspections = {
        "environment/data/market.csv": {
            "rows": 2,
            "columns": ["date", "symbol", "close"],
            "dtypes": {
                "date": "object",
                "symbol": "object",
                "close": "float64",
            },
        }
    }

    planner = build_planner_prompt(
        spec=_Payload({"category": "risk-management"}),
        task_plan=_TaskPlan({"candidate_count": 2}),
        skill_packs=[],
        instruction_text="Estimate historical VaR and expected shortfall.",
        data_inspections=inspections,
        selected_capabilities=selected,
        relevant_runtime_primitives=relevant,
    )
    candidate = build_candidate_prompt(
        instruction_text="Estimate historical VaR and expected shortfall.",
        spec=_Payload({"category": "risk-management"}),
        strategy=_Payload({"approach_name": "historical"}),
        skill_packs=[],
        data_inspections=inspections,
        selected_capabilities=selected,
        relevant_runtime_primitives=relevant,
    )

    for prompt in (planner, candidate):
        assert '"selected_general_capabilities"' in prompt
        assert '"relevant_runtime_primitives"' in prompt
        assert '"data_inspections"' in prompt
        assert "offline_common.risk" in prompt

    assert candidate.index('"selected_general_capabilities"') < candidate.index(
        '"available_runtime_primitives"'
    )
    assert candidate.index('"relevant_runtime_primitives"') < candidate.index(
        '"available_runtime_primitives"'
    )


def test_candidate_prompt_normalizes_nested_non_json_mapping_keys() -> None:
    timestamp = pd.Timestamp("2025-01-02T03:04:05")
    inspections = {
        "environment/data/events.jsonl": {
            "categorical": {
                "event_time": {
                    "top_values": {
                        timestamp: {
                            timestamp: 2,
                        },
                    },
                },
            },
        },
    }

    prompt = build_candidate_prompt(
        instruction_text="Summarize the supplied events.",
        spec=_Payload({"category": "event-study"}),
        strategy=_Payload({"approach_name": "contract-first"}),
        skill_packs=[],
        data_inspections=inspections,
    )

    assert '"2025-01-02 03:04:05"' in prompt


def test_candidate_prompt_compacts_raw_inspections_to_context_ceiling() -> None:
    inspections = {
        f"environment/data/filing_{index:02d}.html": {
            "chars_at_least": 20_000,
            "truncated": True,
            "preview": f"filing-{index}:" + ("x" * 20_000),
        }
        for index in range(20)
    }

    prompt = build_candidate_prompt(
        instruction_text="Classify all supplied filings.",
        spec=_Payload({"category": "nlp-on-finance"}),
        strategy=_Payload(
            {
                "approach_name": "contract-first",
                "implementation_steps": ["x" * 200] * 10,
                "verification_steps": ["y" * 200] * 10,
                "risks": ["z" * 200] * 10,
            }
        ),
        skill_packs=[],
        data_inspections=inspections,
    )

    assert len(prompt) <= MAX_CANDIDATE_PROMPT_CHARS
    assert '"chars_at_least": 20000' in prompt
    assert "[prompt preview truncated]" in prompt


def test_prompt_ceilings_reserve_at_least_twenty_percent_headroom() -> None:
    assumed_context_tokens = 32_768
    reserved_headroom_tokens = int(assumed_context_tokens * 0.20)
    conservative_chars_per_token = 3

    assert (
        MAX_PLANNER_PROMPT_CHARS // conservative_chars_per_token
        + 3_500
        + reserved_headroom_tokens
        <= assumed_context_tokens
    )
    assert (
        MAX_CANDIDATE_PROMPT_CHARS // conservative_chars_per_token
        + 4_000
        + reserved_headroom_tokens
        <= assumed_context_tokens
    )


def test_capability_context_has_a_fixed_prompt_growth_budget(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Estimate historical VaR and expected shortfall.",
    )
    selected = tuple(
        RankedCapability(
            descriptor=descriptor,
            semantic_score=5,
            data_score=4,
            matched_evidence=("generic semantic match",),
        )
        for descriptor in CAPABILITY_CATALOG[:3]
    )
    relevant = relevant_primitive_catalog(selected)
    common = {
        "instruction_text": "Estimate historical VaR and expected shortfall.",
        "spec": _Payload({"category": "risk-management"}),
        "strategy": _Payload({"approach_name": "historical"}),
        "skill_packs": [],
        "data_inspections": {},
    }

    baseline = build_candidate_prompt(**common)
    enriched = build_candidate_prompt(
        **common,
        selected_capabilities=selected,
        relevant_runtime_primitives=relevant,
    )

    assert len(enriched) >= len(baseline)
    assert (
        len(enriched) - len(baseline)
        <= MAX_CAPABILITY_PROMPT_GROWTH_CHARS
    )

    planner_common = {
        "spec": _Payload({"category": "risk-management"}),
        "task_plan": _TaskPlan({"candidate_count": 2}),
        "skill_packs": [],
        "instruction_text": "Estimate historical VaR and expected shortfall.",
        "data_inspections": {},
    }
    planner_baseline = build_planner_prompt(**planner_common)
    planner_enriched = build_planner_prompt(
        **planner_common,
        selected_capabilities=selected,
        relevant_runtime_primitives=relevant,
    )
    assert (
        len(planner_enriched) - len(planner_baseline)
        <= MAX_CAPABILITY_PROMPT_GROWTH_CHARS
    )


def test_planner_data_inspections_keep_structure_without_raw_previews() -> None:
    original = {
        "environment/data/prices.csv": {
            "rows": 100,
            "columns": ["date", "close"],
            "dtypes": {"date": "object", "close": "float64"},
            "missing": {"date": 0, "close": 1},
            "sample_rows": [{"date": "2025-01-01", "close": 100.0}],
            "numeric": {"close": {"min": 90.0, "max": 110.0}},
        },
        "environment/data/source.py": {
            "chars_at_least": 20_000,
            "truncated": True,
            "preview": "prepared implementation text",
        },
    }

    compact = compact_planner_data_inspections(original)

    assert compact["environment/data/prices.csv"] == {
        "rows": 100,
        "columns": ["date", "close"],
        "dtypes": {"date": "object", "close": "float64"},
        "missing": {"date": 0, "close": 1},
    }
    assert compact["environment/data/source.py"] == {
        "chars_at_least": 20_000,
        "truncated": True,
    }
    assert "sample_rows" in original["environment/data/prices.csv"]
    assert "preview" in original["environment/data/source.py"]


def test_runtime_plan_new_context_is_internal_and_backward_compatible() -> None:
    legacy = RuntimePlan(
        planning_specification=None,  # type: ignore[arg-type]
        task_plan=None,
        planner_output=None,
        skill_packs=(),
        data_inspections={},
        strategies=(),
    )

    assert legacy.selected_capabilities == ()
    assert legacy.relevant_runtime_primitives == ()

    descriptor = CAPABILITY_CATALOG[0]
    enriched = replace(
        legacy,
        selected_capabilities=(
            RankedCapability(
                descriptor=descriptor,
                semantic_score=5,
                data_score=0,
                matched_evidence=("semantic: example",),
            ),
        ),
        relevant_runtime_primitives=("example() -> float",),
    )

    assert enriched == legacy
    assert "selected_capabilities" not in repr(enriched)
    assert "relevant_runtime_primitives" not in repr(enriched)
    assert not hasattr(enriched, "to_dict")
    serialized = asdict(enriched)
    assert serialized["selected_capabilities"]
    restored = pickle.loads(pickle.dumps(enriched))
    assert restored == enriched
    assert restored.selected_capabilities == enriched.selected_capabilities
    assert (
        restored.relevant_runtime_primitives
        == enriched.relevant_runtime_primitives
    )


def test_candidate_workspace_exposes_read_only_general_modules(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, "Compute generic risk statistics.")
    workspace = CandidateWorkspace.create(
        base_dir=tmp_path / "work",
        candidate_id=1,
        task_dir=task,
    )
    package = workspace.root_dir / "lib" / "offline_common"

    assert package.is_dir()
    assert not package.stat().st_mode & stat.S_IWUSR

    expected = {"__init__.py", *CANDIDATE_LIBRARY_EXPORTS}
    assert {path.name for path in package.iterdir()} == expected
    assert all(
        not path.stat().st_mode & stat.S_IWUSR
        for path in package.iterdir()
    )

    script = workspace.source_dir / "solver.py"
    script.write_text(
        """
from pathlib import Path
import os
from offline_common.fixed_income import discount_factor_from_zero
from offline_common.risk import historical_var_es

result = historical_var_es(
    [0.01, -0.02, 0.03, -0.01],
    alpha=0.75,
    notional=1.0,
)
discount = discount_factor_from_zero(0.05, 2.0)
Path(os.environ["OUTPUT_DIR"]).joinpath("probe.txt").write_text(
    f"{result}|{discount:.8f}\\n",
    encoding="utf-8",
)
""".strip()
        + "\n",
        encoding="utf-8",
    )

    execution = run_candidate(workspace, script, timeout_seconds=30)

    assert execution.return_code == 0, execution.stderr
    assert (workspace.output_dir / "probe.txt").is_file()


def test_staged_modules_have_no_solution_or_output_entrypoints(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, "Compute generic finance statistics.")
    workspace = CandidateWorkspace.create(
        base_dir=tmp_path / "work",
        candidate_id=1,
        task_dir=task,
    )
    root = workspace.root_dir / "lib" / "offline_common"
    forbidden_definitions = re.compile(
        r"^def\s+(?:solve|matches|_solve|run_[A-Za-z0-9_]*workflow)\s*\(",
        re.MULTILINE,
    )
    forbidden_output_operations = (
        ".write_text(",
        ".write_bytes(",
        ".to_csv(",
        ".to_json(",
    )
    forbidden_prepared_helpers = (
        "parse_risk_spec",
        "parse_cross_sectional_momentum_spec",
        "find_price_panel_csv",
    )

    for descriptor in CAPABILITY_CATALOG:
        filename = descriptor.import_path.rsplit(".", 1)[-1] + ".py"
        source = (root / filename).read_text(encoding="utf-8")
        assert not forbidden_definitions.search(source)
        assert not any(term in source for term in forbidden_output_operations)
        assert not any(term in source for term in forbidden_prepared_helpers)
        assert "task_id" not in source


def test_asian_option_ohlc_calibration_does_not_select_ohlc_estimators(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Price an Asian option using OHLC history only for calibration.",
    )
    _write_ohlc(task / "environment" / "data" / "prices.csv")

    selected = rank_capabilities(
        instruction=(task / "instruction.md").read_text(),
        task_dir=task,
    )

    assert "ohlc-volatility" not in {
        item.descriptor.capability_id for item in selected
    }


def test_kirk_margrabe_avoids_irrelevant_ohlc_guidance(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Calibrate two assets, then price Kirk spread and Margrabe options.",
    )
    for name in ("first.csv", "second.csv"):
        _write_ohlc(task / "environment" / "data" / name)

    selected = rank_capabilities(
        instruction=(task / "instruction.md").read_text(),
        task_dir=task,
    )
    capability_ids = [
        item.descriptor.capability_id for item in selected
    ]

    assert capability_ids[0] == "two-asset-derivatives"
    assert "ohlc-volatility" not in capability_ids


def test_dedicated_ohlc_methods_still_select_ohlc_capability(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        (
            "Estimate Parkinson, Garman-Klass, Rogers-Satchell, and "
            "Yang-Zhang range-based volatility."
        ),
    )
    _write_ohlc(task / "environment" / "data" / "prices.csv")

    selected = rank_capabilities(
        instruction=(task / "instruction.md").read_text(),
        task_dir=task,
    )

    assert selected[0].descriptor.capability_id == "ohlc-volatility"


@pytest.mark.parametrize(
    ("instruction", "data_kind"),
    (
        ("Do not calculate VaR; report the sample mean only.", None),
        (
            "OHLC data is provided but use close-to-close returns only.",
            "ohlc",
        ),
        (
            "Do not bootstrap the supplied yield curve; use it unchanged.",
            "curve",
        ),
        (
            "Estimate realized volatility from intraday returns.",
            None,
        ),
        (
            "Compare this estimator against Parkinson but do not use Parkinson.",
            None,
        ),
    ),
)
def test_excluded_or_comparison_only_methods_do_not_select_capabilities(
    tmp_path: Path,
    instruction: str,
    data_kind: str | None,
) -> None:
    task = _task(tmp_path, instruction)
    if data_kind == "ohlc":
        _write_ohlc(task / "environment" / "data" / "prices.csv")
    elif data_kind == "curve":
        (task / "environment" / "data" / "curve.json").write_text(
            json.dumps(
                {
                    "maturities": [1, 2],
                    "par_rates": [0.03, 0.04],
                }
            ),
            encoding="utf-8",
        )

    assert rank_capabilities(
        instruction=instruction,
        task_dir=task,
    ) == (), instruction


def test_student_t_copula_does_not_select_market_risk(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Fit a Student-t copula to paired equity returns.",
    )
    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "symbol": ["AAA", "AAA"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
        }
    ).to_csv(
        task / "environment" / "data" / "prices.csv",
        index=False,
    )

    assert rank_capabilities(
        instruction=(task / "instruction.md").read_text(),
        task_dir=task,
    ) == ()


def test_generic_event_study_does_not_expose_stock_event_workflow(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        "Run an event study of FOMC text and Treasury yield changes.",
    )

    assert rank_capabilities(
        instruction=(task / "instruction.md").read_text(),
        task_dir=task,
    ) == ()


@pytest.mark.parametrize(
    "instruction",
    (
        "Impose a parametric VaR constraint on portfolio leverage.",
        "Estimate position VaR and ES at 99% confidence.",
        "Compare Hill and Smith tail index estimators.",
        "Fit per-asset GARCH(1,1) conditional volatility for a CTA strategy.",
    ),
)
def test_specific_risk_language_selects_market_risk_capability(
    tmp_path: Path,
    instruction: str,
) -> None:
    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert selected[0].descriptor.capability_id == "market-risk-statistics"
