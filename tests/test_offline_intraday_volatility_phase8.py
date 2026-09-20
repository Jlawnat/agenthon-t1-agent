from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_intraday_volatility import (
    IntradayRealizedVolatilitySkill,
    bandi_russell_noise_variance,
    bipower_variation,
    clean_intraday_quotes,
    realized_variance,
)


def test_intraday_volatility_primitives() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-02 09:29:00",
                "2024-01-02 09:30:00",
                "2024-01-02 09:30:00",
                "2024-01-02 09:31:00",
                "2024-01-02 16:00:00",
                "2024-01-02 16:01:00",
            ],
            "mid_price": [
                99.0,
                100.0,
                102.0,
                101.0,
                103.0,
                104.0,
            ],
        }
    )

    cleaned = clean_intraday_quotes(
        frame
    )

    assert len(
        cleaned
    ) == 3

    assert np.isclose(
        cleaned[
            "mid_price"
        ].iloc[
            0
        ],
        101.0,
    )

    returns = np.asarray(
        [
            0.01,
            -0.02,
            0.015,
        ]
    )

    rv = realized_variance(
        returns
    )

    assert np.isclose(
        rv,
        0.000725,
    )

    assert np.isclose(
        bandi_russell_noise_variance(
            returns
        ),
        rv / 6.0,
    )

    expected_bv = (
        np.pi
        / 2.0
        * (
            abs(
                returns[
                    1
                ]
            )
            * abs(
                returns[
                    0
                ]
            )
            + abs(
                returns[
                    2
                ]
            )
            * abs(
                returns[
                    1
                ]
            )
        )
    )

    assert np.isclose(
        bipower_variation(
            returns
        ),
        expected_bv,
    )


def test_intraday_volatility_skill_matches() -> None:
    skill = (
        IntradayRealizedVolatilitySkill()
    )

    assert skill.matches(
        instruction=(
            "Compute minute-level realized volatility and "
            "realized variance with microstructure-noise correction, "
            "Bandi-Russell noise estimation, and bipower variation."
        ),
        task_dir=Path("."),
    )
