from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_variance_swap import (
    VarianceSwapReplicationSkill,
    black_scholes_price,
    implied_volatility,
    trapezoidal_strike_widths,
    clean_option_chain,
)


def test_implied_volatility_round_trip() -> None:
    price = black_scholes_price(
        spot=100.0,
        strike=105.0,
        maturity=0.5,
        risk_free_rate=0.04,
        dividend_yield=0.0,
        volatility=0.25,
        option_type="call",
    )

    recovered = implied_volatility(
        market_price=price,
        spot=100.0,
        strike=105.0,
        maturity=0.5,
        risk_free_rate=0.04,
        dividend_yield=0.0,
        option_type="call",
    )

    assert np.isclose(
        recovered,
        0.25,
        atol=1e-8,
    )


def test_nonuniform_trapezoidal_widths() -> None:
    strikes = np.asarray(
        [
            80.0,
            90.0,
            100.0,
            115.0,
        ]
    )

    widths = (
        trapezoidal_strike_widths(
            strikes
        )
    )

    assert np.allclose(
        widths,
        [
            10.0,
            10.0,
            12.5,
            15.0,
        ],
    )


def test_chain_cleaning_and_match() -> None:
    chain = pd.DataFrame(
        {
            "expiry": [
                "90d",
                "90d",
                "90d",
                "30d",
            ],
            "strike": [
                95.0,
                100.0,
                105.0,
                100.0,
            ],
            "call_bid": [
                6.0,
                3.0,
                1.0,
                1.0,
            ],
            "call_ask": [
                6.2,
                3.2,
                1.2,
                1.1,
            ],
            "put_bid": [
                0.8,
                2.0,
                5.0,
                2.0,
            ],
            "put_ask": [
                1.0,
                2.2,
                5.2,
                2.1,
            ],
            "volume": [
                20,
                20,
                20,
                20,
            ],
            "open_interest": [
                100,
                100,
                100,
                100,
            ],
        }
    )

    cleaned, n_input = clean_option_chain(
        chain,
        target_expiry="90d",
        forward_price=101.0,
        min_volume=10,
        max_relative_spread=0.5,
    )

    assert n_input == 3
    assert len(cleaned) == 3

    skill = (
        VarianceSwapReplicationSkill()
    )

    assert skill.matches(
        instruction=(
            "Compute a variance swap fair strike using "
            "static option-chain replication of the log contract."
        ),
        task_dir=Path("."),
    )
