from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_barrier_garch import _barrier_price, BarrierGarchVarSkill
from agent.offline_cap_floor import InterestRateCapFloorSkill
from agent.offline_credit_spread import CreditSpreadDecompositionSkill
from agent.offline_copula_sampling import CopulaSamplingSkill
from agent.offline_fx_carry_hedge import FxCarryForwardHedgeSkill
from agent.offline_pca_factor import PcaFactorPortfolioSkill
from agent.offline_router import inspect_task_fingerprint, rank_fallback_candidates
from agent.offline_runtime import _expected_output_files


def test_phase8e_matchers_cover_all_six_domains(tmp_path: Path) -> None:
    cases = [
        (
            CreditSpreadDecompositionSkill(),
            "Decompose a credit spread using Newey-West HAC and variance decomposition.",
        ),
        (
            FxCarryForwardHedgeSkill(),
            "Build an FX carry strategy with a forward hedge and option overlay.",
        ),
        (
            PcaFactorPortfolioSkill(),
            "Construct a PCA factor-neutral portfolio using hedge weights.",
        ),
        (
            BarrierGarchVarSkill(),
            "Fit GARCH and compute VaR for a down-and-out barrier option.",
        ),
        (
            CopulaSamplingSkill(),
            "Sample copulas and compare Kendall rank correlation and tail dependence.",
        ),
        (
            InterestRateCapFloorSkill(),
            "Price an interest rate cap with caplets and floorlets using Black's model.",
        ),
    ]

    for skill, instruction in cases:
        assert skill.matches(instruction=instruction, task_dir=tmp_path)


def test_barrier_formula_matches_reference_checkpoint() -> None:
    value = _barrier_price(
        155.09951085246948,
        152.0,
        0.5,
        0.0121,
        0.2586800935963609,
        111.05,
    )
    assert np.isclose(value, 13.276884710501182, atol=0.01)


def test_router_uses_nested_schema_and_semantics(tmp_path: Path) -> None:
    params = {
        "tickers": ["A", "B"],
        "monthly_returns": {"A": [0.01, 0.02], "B": [0.0, 0.01]},
        "n_factors": 1,
        "target_weights": {"A": 0.5, "B": 0.5},
    }
    (tmp_path / "params.json").write_text(json.dumps(params), encoding="utf-8")

    @dataclass(frozen=True)
    class Skill:
        name: str

    skills = [
        Skill("pca-factor-portfolio-domain"),
        Skill("portfolio-strategy-domain"),
    ]

    ranked = rank_fallback_candidates(
        instruction="Neutralize a portfolio's principal-component factor exposures.",
        task_dir=tmp_path,
        skills=skills,
    )
    assert ranked[0].skill_name == "pca-factor-portfolio-domain"
    fingerprint = inspect_task_fingerprint(tmp_path)
    assert any("monthly_returns" in path for path in fingerprint.json_key_paths)


def test_output_contract_extraction() -> None:
    instruction = """
    Save /app/output/results.json and `/output/summary.csv`.
    """
    assert _expected_output_files(instruction) == {"results.json", "summary.csv"}


def test_output_contract_extraction_from_markdown_output_section() -> None:
    instruction = """
    ## Input
    Read `prices.csv` and `params.json`.

    ## Output Files
    Write all results to `/app/output`.

    - **`digital_prices.csv`**
    - `gap_prices.csv`
    - **`barrier_digital_prices.csv`**
    - `summary.json`

    ## Notes
    Use float64 precision.
    """
    assert _expected_output_files(instruction) == {
        "digital_prices.csv",
        "gap_prices.csv",
        "barrier_digital_prices.csv",
        "summary.json",
    }


def test_output_contract_does_not_confuse_inputs_with_outputs() -> None:
    instruction = """
    # Task
    Read `dj30_constituents_daily.csv`.

    ## Output
    - `data_summary.json`
    - `copula_fits.csv`
    - `best_copulas.json`
    - `tail_dependence.csv`
    - `summary.json`
    """
    expected = _expected_output_files(instruction)
    assert "dj30_constituents_daily.csv" not in expected
    assert expected == {
        "data_summary.json",
        "copula_fits.csv",
        "best_copulas.json",
        "tail_dependence.csv",
        "summary.json",
    }
