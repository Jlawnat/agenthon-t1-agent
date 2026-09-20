from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from agent.offline_lookback_options import (
    LookbackOptionSkill,
    fixed_strike_lookback_call,
    floating_lookback_call,
    floating_lookback_put,
)


def test_lookback_skill_matches() -> None:
    skill = LookbackOptionSkill()

    assert skill.matches(
        instruction=(
            "Price fixed-strike and floating-strike lookback options "
            "and validate them with Monte Carlo running maxima and minima."
        ),
        task_dir=Path("."),
    )


def test_fixed_call_monotonic_in_strike() -> None:
    spot = 100.0

    values = [
        fixed_strike_lookback_call(
            spot=spot,
            running_max=spot,
            strike=spot * m,
            maturity=0.5,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            volatility=0.20,
        )
        for m in (
            0.90,
            0.95,
            1.00,
            1.05,
            1.10,
        )
    ]

    assert all(
        values[index]
        > values[index + 1]
        for index in range(
            len(
                values
            )
            - 1
        )
    )


def test_fixed_call_parity_when_strike_below_running_max() -> None:
    spot = 100.0
    strike = 90.0
    maturity = 0.75
    rate = 0.05
    sigma = 0.20

    fixed = (
        fixed_strike_lookback_call(
            spot=spot,
            running_max=spot,
            strike=strike,
            maturity=maturity,
            risk_free_rate=rate,
            dividend_yield=0.0,
            volatility=sigma,
        )
    )

    floating_put = (
        floating_lookback_put(
            spot=spot,
            running_max=spot,
            maturity=maturity,
            risk_free_rate=rate,
            dividend_yield=0.0,
            volatility=sigma,
        )
    )

    expected = (
        floating_put
        + spot
        - strike
        * math.exp(
            -rate
            * maturity
        )
    )

    assert np.isclose(
        fixed,
        expected,
        atol=1e-10,
    )


def test_floating_prices_positive_and_increase() -> None:
    spot = 100.0

    calls = []
    puts = []

    for maturity in (
        0.25,
        0.50,
        1.00,
    ):
        calls.append(
            floating_lookback_call(
                spot=spot,
                running_min=spot,
                maturity=maturity,
                risk_free_rate=0.05,
                dividend_yield=0.0,
                volatility=0.20,
            )
        )

        puts.append(
            floating_lookback_put(
                spot=spot,
                running_max=spot,
                maturity=maturity,
                risk_free_rate=0.05,
                dividend_yield=0.0,
                volatility=0.20,
            )
        )

    assert all(
        value > 0.0
        for value in (
            calls
            + puts
        )
    )

    assert (
        calls[0]
        < calls[1]
        < calls[2]
    )

    assert (
        puts[0]
        < puts[1]
        < puts[2]
    )
