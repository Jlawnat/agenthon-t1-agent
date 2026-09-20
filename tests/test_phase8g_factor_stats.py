from __future__ import annotations

import numpy as np
import pandas as pd

from agent.finance_composition import plan_finance_task
from agent.finance_factor_stats import (
    durbin_watson,
    grs_joint_alpha_test,
    newey_west_covariance,
    ols_with_inference,
    optimal_newey_west_lag,
    rolling_ols_coefficient,
    variance_inflation_factors,
)


def test_ols_and_hac_primitives():
    rng = np.random.default_rng(7)
    n = 300
    factors = rng.normal(size=(n, 3))
    x = np.column_stack([np.ones(n), factors])
    true = np.array([0.002, 1.1, -0.3, 0.4])
    y = x @ true + rng.normal(scale=0.05, size=n)

    fit = ols_with_inference(x, y)
    assert fit.coefficients.shape == (4,)
    assert np.allclose(fit.coefficients, true, atol=0.02)
    assert 0.0 < fit.r_squared < 1.0

    lag = optimal_newey_west_lag(n)
    cov = newey_west_covariance(x, fit.residuals, lag=lag)
    assert cov.shape == (4, 4)
    assert np.allclose(cov, cov.T)
    assert np.diag(cov).min() >= 0.0
    assert 0.0 < durbin_watson(fit.residuals) < 4.0


def test_vif_and_rolling_ols():
    rng = np.random.default_rng(11)
    n = 120
    factors = rng.normal(size=(n, 3))
    vif = variance_inflation_factors(factors)
    assert vif.shape == (3,)
    assert (vif >= 1.0).all()

    x = np.column_stack([np.ones(n), factors])
    y = 0.001 + 0.8 * factors[:, 0] + rng.normal(scale=0.02, size=n)
    rolling = rolling_ols_coefficient(x, y, window=40, coefficient_index=1)
    assert rolling.shape == (81,)
    assert np.isfinite(rolling).all()


def test_grs_joint_alpha_test_returns_valid_f_statistic():
    rng = np.random.default_rng(21)
    t_obs, n_assets, n_factors = 400, 4, 3
    factors = rng.normal(scale=0.01, size=(t_obs, n_factors))
    residuals = rng.normal(scale=0.02, size=(t_obs, n_assets))
    alphas = np.array([0.0001, -0.0001, 0.0002, 0.0])
    result = grs_joint_alpha_test(alphas, residuals, factors)
    assert result.statistic >= 0.0
    assert 0.0 <= result.p_value <= 1.0
    assert result.df1 == n_assets
    assert result.df2 == t_obs - n_assets - n_factors


def test_planner_recognizes_factor_regression_recipe(tmp_path):
    data = tmp_path / "environment" / "data"
    data.mkdir(parents=True)

    pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "asset_a": [100.0, 101.0, 102.0],
        "asset_b": [80.0, 81.0, 82.0],
    }).to_csv(data / "assets.csv", index=False)

    pd.DataFrame({
        "Date": ["2026-01-02", "2026-01-03"],
        "Mkt-RF": [0.01, 0.02],
        "SMB": [0.001, -0.001],
        "HML": [0.002, 0.003],
        "RF": [0.0001, 0.0001],
    }).to_csv(data / "factors.csv", index=False)

    plan = plan_finance_task(
        "Estimate a Fama-French factor model with Newey-West HAC, GRS joint "
        "alpha test, VIF diagnostics and rolling market beta.",
        tmp_path,
    )
    assert plan.executable_recipe == "factor-regression-analysis"
    assert {
        "factor_ols",
        "newey_west_hac",
        "grs_joint_alpha_test",
        "rolling_factor_beta",
        "vif",
    }.issubset(set(plan.capabilities))
