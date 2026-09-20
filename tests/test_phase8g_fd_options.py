from __future__ import annotations

import numpy as np

from agent.finance_composition import plan_finance_task
from agent.finance_fd import (
    FiniteDifferenceOptionSpec,
    crank_nicolson_option,
    parse_fd_option_spec,
    richardson_second_order,
    with_grid_size,
)


def test_fd_parser_and_planner(tmp_path):
    instruction = r"""
    # American Option Pricing with Crank-Nicolson Finite Differences
    | Parameter | Value |
    | S_0 (initial stock price) | 100 |
    | K (strike price) | 100 |
    | r (risk-free rate) | 0.05 |
    | q (continuous dividend yield) | 0.0 |
    | sigma (volatility) | 0.30 |
    | T (time to maturity) | 1.0 year |
    | Cash dividend 1 | $2.50 at t = 0.25 |
    Stock domain: S_max = 3K = 300
    Stock steps: N_S = 300
    Time steps: N_T = 600
    Use Crank-Nicolson with PSOR, discrete cash dividends, early exercise boundary,
    delta and Richardson extrapolation.
    """
    spec = parse_fd_option_spec(instruction)
    assert spec.spot == 100.0
    assert spec.strike == 100.0
    assert spec.stock_steps == 300
    assert spec.time_steps == 600
    assert spec.dividends == ((0.25, 2.5),)

    plan = plan_finance_task(instruction, tmp_path)
    assert plan.executable_recipe == "finite-difference-option-analysis"
    assert {
        "crank_nicolson_pde",
        "psor_projection",
        "cash_dividend_jump",
        "exercise_boundary",
        "finite_difference_delta",
        "richardson_extrapolation",
    }.issubset(set(plan.capabilities))


def test_fd_no_dividend_american_call_matches_european_call():
    spec = FiniteDifferenceOptionSpec(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        dividend_yield=0.0,
        volatility=0.30,
        maturity=1.0,
        s_max=300.0,
        stock_steps=80,
        time_steps=160,
        dividends=(),
    )
    american = crank_nicolson_option(
        spec,
        option_type="call",
        exercise_type="american",
        dividends=(),
    )
    european = crank_nicolson_option(
        spec,
        option_type="call",
        exercise_type="european",
        dividends=(),
    )
    assert abs(american.value - european.value) < 2e-3
    assert american.delta > 0.0
    assert european.delta > 0.0


def test_fd_put_boundary_and_richardson():
    spec = FiniteDifferenceOptionSpec(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        dividend_yield=0.0,
        volatility=0.30,
        maturity=1.0,
        s_max=300.0,
        stock_steps=60,
        time_steps=120,
        dividends=((0.25, 2.5), (0.75, 2.5)),
    )
    fine = crank_nicolson_option(
        spec,
        option_type="put",
        exercise_type="american",
        return_grid=True,
        return_boundary=True,
    )
    assert fine.value > 0.0
    assert fine.delta < 0.0
    assert fine.exercise_boundary is not None
    assert np.isclose(fine.exercise_boundary[-1], 100.0)

    coarse_spec = with_grid_size(spec, stock_steps=30, time_steps=60)
    coarse = crank_nicolson_option(
        coarse_spec,
        option_type="put",
        exercise_type="american",
    )
    rich = richardson_second_order(fine.value, coarse.value)
    assert np.isfinite(rich)
