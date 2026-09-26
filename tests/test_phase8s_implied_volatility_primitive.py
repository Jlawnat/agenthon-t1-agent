from __future__ import annotations

import math

import pytest

from agent.qf_primitives import (
    black_scholes_price,
    implied_volatility_black_scholes,
)


@pytest.mark.parametrize(
    (
        "spot",
        "strike",
        "rate",
        "dividend_yield",
        "maturity",
        "volatility",
        "option_type",
    ),
    [
        (
            100.0,
            100.0,
            0.03,
            0.00,
            1.0,
            0.20,
            "call",
        ),
        (
            100.0,
            110.0,
            0.04,
            0.02,
            1.5,
            0.35,
            "call",
        ),
        (
            95.0,
            100.0,
            0.025,
            0.01,
            0.75,
            0.28,
            "put",
        ),
        (
            150.0,
            120.0,
            0.05,
            0.03,
            2.0,
            0.55,
            "put",
        ),
    ],
)
def test_implied_volatility_round_trip(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
    volatility: float,
    option_type: str,
) -> None:
    market_price = black_scholes_price(
        spot=spot,
        strike=strike,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        maturity=maturity,
        option_type=option_type,
    )

    implied = implied_volatility_black_scholes(
        market_price=market_price,
        spot=spot,
        strike=strike,
        rate=rate,
        dividend_yield=dividend_yield,
        maturity=maturity,
        option_type=option_type,
    )

    assert implied == pytest.approx(
        volatility,
        rel=1e-8,
        abs=1e-10,
    )


@pytest.mark.parametrize(
    "option_type",
    [
        "call",
        "put",
    ],
)
def test_implied_volatility_handles_lower_price_boundary(
    option_type: str,
) -> None:
    spot = 100.0
    strike = 105.0
    rate = 0.04
    dividend_yield = 0.015
    maturity = 1.25

    discounted_spot = (
        spot
        * math.exp(
            -dividend_yield
            * maturity
        )
    )

    discounted_strike = (
        strike
        * math.exp(
            -rate
            * maturity
        )
    )

    if option_type == "call":
        lower_bound = max(
            discounted_spot
            - discounted_strike,
            0.0,
        )
    else:
        lower_bound = max(
            discounted_strike
            - discounted_spot,
            0.0,
        )

    implied = implied_volatility_black_scholes(
        market_price=lower_bound,
        spot=spot,
        strike=strike,
        rate=rate,
        dividend_yield=dividend_yield,
        maturity=maturity,
        option_type=option_type,
    )

    assert implied == pytest.approx(
        0.0,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    "option_type",
    [
        "call",
        "put",
    ],
)
def test_implied_volatility_rejects_arbitrage_violations(
    option_type: str,
) -> None:
    spot = 100.0
    strike = 105.0
    rate = 0.04
    dividend_yield = 0.015
    maturity = 1.25

    discounted_spot = (
        spot
        * math.exp(
            -dividend_yield
            * maturity
        )
    )

    discounted_strike = (
        strike
        * math.exp(
            -rate
            * maturity
        )
    )

    if option_type == "call":
        lower_bound = max(
            discounted_spot
            - discounted_strike,
            0.0,
        )
        upper_bound = (
            discounted_spot
        )
    else:
        lower_bound = max(
            discounted_strike
            - discounted_spot,
            0.0,
        )
        upper_bound = (
            discounted_strike
        )

    with pytest.raises(
        ValueError,
        match="arbitrage",
    ):
        implied_volatility_black_scholes(
            market_price=(
                lower_bound
                - 1e-4
            ),
            spot=spot,
            strike=strike,
            rate=rate,
            dividend_yield=dividend_yield,
            maturity=maturity,
            option_type=option_type,
        )

    with pytest.raises(
        ValueError,
        match="arbitrage",
    ):
        implied_volatility_black_scholes(
            market_price=(
                upper_bound
                + 1e-4
            ),
            spot=spot,
            strike=strike,
            rate=rate,
            dividend_yield=dividend_yield,
            maturity=maturity,
            option_type=option_type,
        )


def test_implied_volatility_requires_positive_maturity() -> None:
    with pytest.raises(
        ValueError,
        match="maturity",
    ):
        implied_volatility_black_scholes(
            market_price=5.0,
            spot=100.0,
            strike=100.0,
            rate=0.03,
            dividend_yield=0.0,
            maturity=0.0,
            option_type="call",
        )


def test_implied_volatility_rejects_invalid_option_type() -> None:
    with pytest.raises(
        ValueError,
        match="option_type",
    ):
        implied_volatility_black_scholes(
            market_price=10.0,
            spot=100.0,
            strike=100.0,
            rate=0.03,
            dividend_yield=0.0,
            maturity=1.0,
            option_type="straddle",
        )


def test_implied_volatility_detects_insufficient_bracket() -> None:
    true_volatility = 1.20

    market_price = black_scholes_price(
        spot=100.0,
        strike=100.0,
        rate=0.02,
        dividend_yield=0.0,
        volatility=true_volatility,
        maturity=1.0,
        option_type="call",
    )

    with pytest.raises(
        ValueError,
        match="bracket",
    ):
        implied_volatility_black_scholes(
            market_price=market_price,
            spot=100.0,
            strike=100.0,
            rate=0.02,
            dividend_yield=0.0,
            maturity=1.0,
            option_type="call",
            upper_volatility=0.50,
        )
