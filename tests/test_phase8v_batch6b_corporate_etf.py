from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.capability_bridge import rank_capabilities
from agent.finance_multimodal import normalize_cross_section as mature_normalize
from agent.finance_process_selection import select_return_process as mature_process
from agent.offline_common import (
    attribution,
    corporate_actions,
    etf_pressure,
    lead_lag,
    multimodal,
    process_selection,
)
from agent.offline_etf_cross_asset_lead_lag import _lagged_pair_metrics
from agent.offline_etf_overlap_redemption import _parse_money, _parse_percent


PUBLIC_UNITS = Path("/home/elliot/track1-coding-public/units")


def test_corporate_actions_match_backward_adjustment_conventions() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "close": [100.0, 102.0, 51.0, 50.0],
            "volume": [10.0, 20.0, 40.0, 50.0],
        }
    )
    result = corporate_actions.backward_adjust_prices(
        prices,
        [
            {"date": "2024-01-02", "type": "dividend", "amount": 2.0},
            {"date": "2024-01-03", "type": "split", "ratio": "2:1"},
        ],
    )
    assert np.isclose(result.cumulative_price_factor, (100.0 / 102.0) / 2.0)
    assert np.isclose(result.cumulative_volume_factor, 2.0)
    assert np.isclose(result.prices.loc[0, "close_adjusted"], 100.0 * (100.0 / 102.0) / 2.0)
    assert result.prices.loc[0, "volume_adjusted"] == 20.0


def test_corporate_action_errors_are_explicit() -> None:
    assert corporate_actions.parse_split_ratio("3:2") == 1.5
    with pytest.raises(ValueError):
        corporate_actions.parse_split_ratio("0:1")


def test_lead_lag_metrics_match_mature_pair_calculation() -> None:
    rng = np.random.default_rng(42)
    first = rng.normal(size=80)
    second = np.roll(first, 2) + rng.normal(scale=0.05, size=80)
    frame = pd.DataFrame({"A": first, "B": second})
    _metrics, mature = _lagged_pair_metrics(frame, 5)
    generic = lead_lag.pairwise_lead_lag(frame, max_lag=5)[0]
    assert generic.correlations == {int(key): value for key, value in mature[0]["lag_correlations"].items()}
    assert generic.asymmetry == mature[0]["asymmetry"]
    assert generic.leader == mature[0]["leader"]
    assert generic.significant == mature[0]["significant"]


def test_market_model_fit_is_reused_out_of_sample() -> None:
    market = np.linspace(-0.02, 0.03, 20)
    sample = pd.DataFrame({"MKT": market, "A": 0.01 + 1.5 * market})
    models = lead_lag.fit_market_models(sample, "MKT")
    assert np.isclose(models["A"]["alpha"], 0.01)
    assert np.isclose(models["A"]["beta"], 1.5)
    residuals = lead_lag.market_model_residuals(sample, models, "MKT")
    assert np.allclose(residuals["A"], 0.0, atol=1e-15)


def test_etf_parsers_match_mature_workbook_parsers() -> None:
    for value in ("$653,201.93 M", "$1.2 B", "4,000"):
        compact = value.replace(" ", "")
        assert np.isclose(etf_pressure.parse_financial_amount(value), _parse_money(compact))
    for value in ("12.5%", "0.03"):
        assert np.isclose(etf_pressure.parse_percentage(value), _parse_percent(value))


def test_redemption_pressure_reconciles_across_funds() -> None:
    shares = np.array([[100.0, 50.0], [0.0, 40.0]])
    dollars = np.array([[1000.0, 500.0], [0.0, 800.0]])
    result = etf_pressure.redemption_pressure(shares, dollars, np.array([0.1, 0.2]))
    assert np.allclose(result["combined_sell_shares"], [20.0, 8.0])
    assert np.allclose(result["combined_sell_dollars"], [200.0, 160.0])
    assert np.isclose(etf_pressure.reconcile_constituent_price(shares[0], [0.1, 0.05], [10_000.0, 10_000.0]), 10.0)


def test_brinson_effects_reconcile_active_arithmetic_return() -> None:
    wp = np.array([0.6, 0.4])
    wb = np.array([0.5, 0.5])
    rp = np.array([0.10, 0.02])
    rb = np.array([0.08, 0.03])
    result = attribution.brinson_fachler_effects(wp, wb, rp, rb)
    total = result.allocation.sum() + result.selection.sum() + result.interaction.sum()
    assert np.isclose(total, np.dot(wp, rp) - np.dot(wb, rb))
    assert np.isclose(attribution.compound_returns([0.1, -0.05]), 0.045)
    assert np.isclose(attribution.drift_weights([0.5, 0.5], [0.1, 0.0]).sum(), 1.0)


def test_multimodal_normalization_and_fusion_match_mature_formula() -> None:
    values = np.array([1.0, 2.0, np.nan, 4.0])
    assert np.allclose(multimodal.normalize_cross_section(values), mature_normalize(pd.Series(values)))
    scores = np.array([[1.0, -1.0, 0.5, 0.0], [0.2, 0.1, 0.0, -0.1]])
    weights = np.array([0.3, 0.2, 0.25, 0.25])
    result = multimodal.fuse_committee_scores(scores, weights, disagreement_penalty=0.4)
    expected_score = scores @ weights
    expected_dispersion = scores.std(axis=1, ddof=0)
    assert np.allclose(result["committee_score"], expected_score)
    assert np.allclose(result["committee_alpha"], expected_score * np.maximum(0.0, 1.0 - 0.4 * expected_dispersion))


def test_reference_vintage_audit_is_asof_and_staleness_aware() -> None:
    first = pd.DataFrame({"date": ["2024-01-01", "2024-02-01"], "field": ["a", "b"], "value": [1.0, "x"]})
    second = pd.DataFrame({"date": ["2024-01-15", "2024-02-01"], "field": ["a", "b"], "value": [1.5, "y"]})
    result = multimodal.reference_vintage_disagreement(first, second, asof_date="2024-03-01", stale_after_days=40)
    assert result == {"mismatch_count": 2, "mismatch_abs_sum": 1.5, "stale_field_count": 1}


def test_return_process_selection_matches_mature_aic_core() -> None:
    returns = [0.02, 0.04, -0.01, 0.03, 0.08, 0.01, -0.02, 0.05, 0.01, 0.02, -0.01, 0.03]
    generic = process_selection.select_return_process(
        returns,
        minimum_history=6,
        holding_period_days=21,
        conviction_scale=4.0,
        jump_threshold_sigma=1.4,
    )
    mature = mature_process(
        returns,
        minimum_history=6,
        holding_period_days=21,
        conviction_scale=4.0,
        jump_threshold_sigma=1.4,
    )
    assert generic.selected_process == mature.selected_process
    assert np.isclose(generic.aic["martingale"], mature.martingale_aic)
    assert np.isclose(generic.aic["gbm"], mature.gbm_aic)
    assert np.isclose(generic.aic["ou"], mature.ou_aic)
    assert np.isclose(generic.aic["merton_jump_diffusion"], mature.merton_aic)


@pytest.mark.parametrize(
    ("unit_name", "capability_ids"),
    (
        ("t1-corporate-action-adjustment", {"corporate-action-adjustments"}),
        ("t1-etf-cross-asset-lead-lag", {"cross-asset-lead-lag"}),
        ("t1-etf-overlap-redemption-pressure", {"etf-redemption-pressure"}),
        ("t1-brinson-sector-attribution", {"brinson-fachler-attribution"}),
        ("t1-multimodal-alpha-fusion-edgar-cot-gdelt", {"multimodal-committee-fusion", "return-process-selection"}),
    ),
)
def test_batch6b_public_units_route_narrowly(unit_name: str, capability_ids: set[str]) -> None:
    task = PUBLIC_UNITS / unit_name
    selected = rank_capabilities(instruction=(task / "instruction.md").read_text(), task_dir=task)
    assert capability_ids.issubset({item.descriptor.capability_id for item in selected})


@pytest.mark.parametrize(
    "instruction",
    (
        "Compute a contemporaneous correlation matrix.",
        "List holdings in a mutual fund.",
        "Explain allocation decisions in prose.",
        "Apply a generic Gaussian model.",
    ),
)
def test_batch6b_generic_language_does_not_route(tmp_path: Path, instruction: str) -> None:
    task = tmp_path / "task"
    task.mkdir()
    assert rank_capabilities(instruction=instruction, task_dir=task) == ()


def test_dividend_adjusted_backtest_does_not_select_corporate_actions(tmp_path: Path) -> None:
    instruction = "Run a Bollinger backtest on dividend-adjusted prices and account for stock splits."
    task = tmp_path / "task"
    task.mkdir()
    selected = rank_capabilities(instruction=instruction, task_dir=task)
    assert "corporate-action-adjustments" not in {item.descriptor.capability_id for item in selected}
