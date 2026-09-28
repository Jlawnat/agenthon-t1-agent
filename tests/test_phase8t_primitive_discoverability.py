from __future__ import annotations

import re

import agent.qf_primitives as qf
from agent.qf_primitives import PRIMITIVE_API_CATALOG


EXPECTED_FUNCTIONS = (
    "black_scholes_price",
    "black_scholes_greeks",
    "historical_log_return_calibration",
    "discount_cashflow",
    "sma_seeded_ema",
    "ewma_annualized_volatility",
    "log_return_performance",
    "down_and_out_call_price",
    "central_price_delta",
    "ols_with_intercept",
    "fit_ou_euler",
    "fit_ou_exact_ar1",
    "ou_exact_step_parameters",
    "simulate_ou_exact",
    "brownian_running_max_hit_probability",
    "brownian_running_min_hit_probability",
    "brownian_joint_terminal_max_cdf",
    "empirical_var_es_from_losses",
    "normalize_nonnegative",
    "ewma_weights",
    "largest_remainder_allocate",
    "fit_garch11_zero_mean",
    "garch11_forecast_variance",
    "implied_volatility_black_scholes",
    "discount_factor_from_continuous_zero_rate",
    "continuous_zero_rate_from_discount_factor",
    "log_linear_discount_factor",
    "bootstrap_annual_par_discount_factors",
    "pseudo_observations",
    "sample_gaussian_copula",
    "sample_student_t_copula",
    "fit_gpd_exceedances",
    "evt_var_es_from_gpd",
    "hill_tail_index",
    "pca_from_observations",
    "pca_from_covariance",
)


EXPECTED_DICT_KEYS = {
    "black_scholes_greeks": {
        "delta",
        "gamma",
        "rho",
        "theta_annual",
        "vega",
    },
    "historical_log_return_calibration": {
        "annualized_vol",
        "n_returns",
        "return_mean",
        "return_std",
        "spot",
    },
    "log_return_performance": {
        "annualized_return",
        "annualized_volatility",
        "calmar_ratio",
        "max_drawdown",
        "sharpe_ratio",
    },
    "ols_with_intercept": {
        "intercept",
        "n_observations",
        "r_squared",
        "residuals",
        "slope",
    },
    "fit_ou_euler": {
        "intercept",
        "kappa",
        "n_observations",
        "r_squared",
        "residual_std",
        "residuals",
        "sigma",
        "slope",
        "theta",
    },
    "fit_ou_exact_ar1": {
        "intercept",
        "kappa",
        "n_observations",
        "phi",
        "r_squared",
        "residual_std",
        "residuals",
        "sigma",
        "theta",
    },
    "ou_exact_step_parameters": {
        "decay",
        "step_std",
    },
    "fit_garch11_zero_mean": {
        "alpha",
        "beta",
        "conditional_variance",
        "convergence_flag",
        "long_run_variance",
        "n_observations",
        "omega",
        "persistence",
    },
    "garch11_forecast_variance": {
        "aggregate_variance",
        "persistence",
        "variance_path",
    },
    "bootstrap_annual_par_discount_factors": {
        "discount_factors",
        "forward_rates",
        "maturities",
        "zero_rates",
    },
    "fit_gpd_exceedances": {
        "mean_excess",
        "n_exceedances",
        "scale",
        "shape",
    },
    "evt_var_es_from_gpd": {
        "expected_shortfall",
        "var",
    },
    "pca_from_observations": {
        "components",
        "explained_variance",
        "explained_variance_ratio",
        "mean",
        "scores",
    },
    "pca_from_covariance": {
        "components",
        "cumulative_explained_variance_ratio",
        "eigenvalues",
        "explained_variance_ratio",
    },
}


def _catalog_by_name() -> dict[str, str]:
    result = {}

    for entry in PRIMITIVE_API_CATALOG:
        name = entry.split("(", 1)[0].strip()

        assert name not in result, (
            f"duplicate primitive catalog entry for {name}"
        )

        result[name] = entry

    return result


def _documented_keys(entry: str) -> set[str]:
    match = re.search(
        r"keys=\[([^\]]*)\]",
        entry,
    )

    assert match is not None, (
        f"dictionary-return primitive lacks key schema: {entry}"
    )

    return {
        item.strip()
        for item in match.group(1).split(",")
        if item.strip()
    }


def test_catalog_exposes_exact_36_runtime_primitives():
    catalog = _catalog_by_name()

    assert tuple(catalog) == EXPECTED_FUNCTIONS
    assert len(PRIMITIVE_API_CATALOG) == 36

    for name in EXPECTED_FUNCTIONS:
        assert callable(getattr(qf, name))


def test_all_14_dictionary_returns_document_exact_keys():
    catalog = _catalog_by_name()

    assert len(EXPECTED_DICT_KEYS) == 14

    dict_entries = {
        name
        for name, entry in catalog.items()
        if "-> dict" in entry
    }

    assert dict_entries == set(EXPECTED_DICT_KEYS)

    for name, expected_keys in EXPECTED_DICT_KEYS.items():
        assert (
            _documented_keys(catalog[name])
            == expected_keys
        )


def test_no_dictionary_api_is_advertised_as_bare_dict():
    for entry in PRIMITIVE_API_CATALOG:
        if "-> dict" in entry:
            assert "keys=[" in entry
