from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agent.finance_composition import plan_finance_task
from agent.finance_derivatives_mc import (
    black_scholes_greeks,
    black_scholes_price,
    cliquet_forward_start_prices,
    forward_start_atm_call_price,
    historical_log_return_calibration,
)


def _task(tmp_path: Path) -> Path:
    data = tmp_path / "task" / "environment" / "data"
    data.mkdir(parents=True)
    return data.parent.parent


def test_black_scholes_put_call_parity() -> None:
    params = dict(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        dividend_yield=0.02,
        volatility=0.30,
        maturity=1.0,
    )
    call = black_scholes_price(option_type="call", **params)
    put = black_scholes_price(option_type="put", **params)
    expected = 100.0 * np.exp(-0.02) - 100.0 * np.exp(-0.05)
    assert abs((call - put) - expected) < 1e-12


def test_black_scholes_greek_parity() -> None:
    params = dict(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        dividend_yield=0.02,
        volatility=0.30,
        maturity=1.0,
    )
    call = black_scholes_greeks(option_type="call", **params)
    put = black_scholes_greeks(option_type="put", **params)
    assert abs(call["delta"] - put["delta"] - np.exp(-0.02)) < 1e-12
    assert abs(call["gamma"] - put["gamma"]) < 1e-12
    assert abs(
        call["rho"] - put["rho"] - 100.0 * np.exp(-0.05)
    ) < 1e-12


def test_forward_start_first_period_equals_same_horizon_atm_call() -> None:
    price = forward_start_atm_call_price(
        spot=100.0,
        rate=0.05,
        dividend_yield=0.013,
        volatility=0.20,
        start=0.0,
        end=0.25,
    )
    vanilla = black_scholes_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        dividend_yield=0.013,
        volatility=0.20,
        maturity=0.25,
        option_type="call",
    )
    assert abs(price - vanilla) < 1e-12


def test_cliquet_contains_requested_number_of_forward_starts() -> None:
    values = cliquet_forward_start_prices(
        spot=100.0,
        rate=0.05,
        dividend_yield=0.013,
        volatility=0.20,
        maturity=2.0,
        resets=12,
    )
    assert values.shape == (12,)
    assert np.all(values > 0.0)


def test_historical_volatility_uses_sample_std() -> None:
    closes = np.array([100.0, 101.0, 99.0, 102.0, 103.0])
    fit = historical_log_return_calibration(closes)
    returns = np.log(closes[1:] / closes[:-1])
    assert fit.n_returns == 4
    assert abs(fit.return_std - np.std(returns, ddof=1)) < 1e-15


def test_planner_recognises_cliquet_from_semantics_and_schema(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    data = task / "environment" / "data"
    pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "close": [100.0, 101.0],
    }).to_csv(data / "market.csv", index=False)

    plan = plan_finance_task(
        "Estimate historical volatility, price a cliquet "
        "(ratchet) as forward-starting ATM calls, and compare "
        "with a Black-Scholes European option.",
        task,
    )
    assert plan.executable_recipe == "cliquet-forward-start-analysis"
    assert {
        "historical_volatility",
        "black_scholes_vanilla",
        "forward_start_option",
        "cliquet_aggregation",
    }.issubset(plan.capabilities)


def test_planner_recognises_mc_greeks_without_input_files(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    plan = plan_finance_task(
        "Use Monte Carlo for European and Asian options. "
        "Compute finite difference Greeks, pathwise Greeks, "
        "likelihood ratio Greeks, delta and vega surfaces, "
        "and a convergence study.",
        task,
    )
    assert plan.executable_recipe == "mc-greek-surface-analysis"
    assert {
        "gbm_monte_carlo",
        "pathwise_greeks",
        "likelihood_ratio_greeks",
        "greek_surface",
        "mc_convergence",
    }.issubset(plan.capabilities)


def test_historical_moments_follow_scipy_default_bias_convention() -> None:
    from scipy.stats import kurtosis, skew

    closes = np.array([
        100.0,
        102.0,
        101.0,
        105.0,
        103.0,
        110.0,
    ])
    fit = historical_log_return_calibration(closes)
    returns = np.log(closes[1:] / closes[:-1])

    assert abs(
        fit.return_skewness - skew(returns)
    ) < 1e-15

    assert abs(
        fit.return_excess_kurtosis
        - kurtosis(returns, fisher=True)
    ) < 1e-15
