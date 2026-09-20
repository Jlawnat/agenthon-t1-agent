from __future__ import annotations

import json

import numpy as np
import pandas as pd

from agent.finance_composition import plan_finance_task
from agent.finance_multivariate import (
    conditional_covariance_path,
    coverage_backtest,
    dcc_correlation_path,
    filtered_historical_var_es,
    portfolio_conditional_volatility,
    standardized_residuals,
    stationary_bootstrap_rows,
)


def test_standardized_residuals_and_covariance_shapes():
    returns = np.array([[0.01, -0.02], [0.02, 0.01], [-0.01, 0.03]])
    variances = np.full_like(returns, 0.0004)
    z = standardized_residuals(returns, variances)
    assert z.shape == returns.shape
    corr = dcc_correlation_path(z, 0.05, 0.90)
    assert corr.shape == (3, 2, 2)
    assert np.allclose(np.diagonal(corr, axis1=1, axis2=2), 1.0)
    sigma = np.sqrt(variances)
    cov = conditional_covariance_path(sigma, corr)
    port = portfolio_conditional_volatility(cov, [0.5, 0.5])
    assert cov.shape == (3, 2, 2)
    assert (port > 0.0).all()


def test_stationary_bootstrap_is_deterministic_and_preserves_rows():
    x = np.column_stack((np.arange(20.0), np.arange(20.0) + 100.0))
    a = stationary_bootstrap_rows(x, n_samples=50, restart_probability=1 / 7, seed=42)
    b = stationary_bootstrap_rows(x, n_samples=50, restart_probability=1 / 7, seed=42)
    assert np.array_equal(a, b)
    originals = {tuple(row) for row in x}
    assert all(tuple(row) in originals for row in a)


def test_filtered_historical_var_es_properties():
    z = np.array([
        [-2.0, -1.0],
        [-1.5, -0.5],
        [-1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.5],
        [2.0, 1.0],
    ])
    sigma = np.array([[0.02, 0.03], [0.04, 0.02]])
    var, es = filtered_historical_var_es(
        z, sigma, [0.5, 0.5], tail_probability=0.2
    )
    assert var.shape == (2,)
    assert es.shape == (2,)
    assert (var > 0.0).all()
    assert (es >= var).all()


def test_coverage_backtest_is_self_consistent():
    violations = [0] * 94 + [1, 0, 0, 1, 0, 0]
    result = coverage_backtest(violations, expected_rate=0.01)
    assert result.num_observations == 100
    assert result.num_violations == 2
    assert np.isclose(result.violation_rate, 0.02)
    assert result.kupiec_lr_stat >= 0.0
    assert 0.0 <= result.kupiec_p_value <= 1.0
    assert result.christoffersen_lr_cc >= result.kupiec_lr_stat
    assert 0.0 <= result.christoffersen_p_value <= 1.0


def test_planner_recognizes_schema_composed_dcc_recipe(tmp_path):
    data = tmp_path / "environment" / "data"
    data.mkdir(parents=True)
    pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "asset_a": [100.0, 101.0, 102.0],
        "asset_b": [80.0, 79.0, 81.0],
    }).to_csv(data / "panel.csv", index=False)
    (data / "allocation.json").write_text(
        json.dumps({"asset_a": 0.6, "asset_b": 0.4}),
        encoding="utf-8",
    )

    plan = plan_finance_task(
        "Fit DCC GARCH, compute VaR and expected shortfall using filtered historical "
        "simulation with stationary bootstrap, then run Kupiec and Christoffersen tests.",
        tmp_path,
    )
    assert plan.executable_recipe == "dcc-multivariate-risk"
    assert {
        "garch11",
        "standardized_residuals",
        "dcc_correlation",
        "stationary_bootstrap",
        "kupiec_backtest",
        "christoffersen_backtest",
        "var_es",
    }.issubset(set(plan.capabilities))
