from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_curve_immunization import (
    CurveImmunizationSkill,
    bootstrap_discount_factors,
    _fit_svensson,
)


def test_bootstrap_reprices_par_curve() -> None:
    par = pd.DataFrame(
        {
            "maturity_years": [
                1,
                2,
                3,
            ],
            "par_yield": [
                0.04,
                0.042,
                0.043,
            ],
        }
    )

    curve = bootstrap_discount_factors(
        par
    )

    discounts = curve[
        "discount_factor"
    ].to_numpy(
        dtype=float
    )

    for row in par.itertuples(
        index=False
    ):
        maturity = int(
            row.maturity_years
        )
        coupon = float(
            row.par_yield
        )

        price = (
            coupon
            * float(
                np.sum(
                    discounts[
                        :maturity
                    ]
                )
            )
            + discounts[
                maturity
                - 1
            ]
        )

        assert np.isclose(
            price,
            1.0,
            atol=1e-10,
        )


def test_curve_immunization_skill_and_svensson() -> None:
    skill = CurveImmunizationSkill()

    assert skill.matches(
        instruction=(
            "Bootstrap a yield curve, calculate key rate durations, "
            "build a portfolio immunization hedge, and fit a Svensson curve."
        ),
        task_dir=Path("."),
    )

    spot = np.asarray(
        [
            0.0469,
            0.0423,
            0.0399,
            0.0392,
            0.0384,
            0.0385,
            0.0386,
            0.0386,
            0.0387,
            0.0387,
        ],
        dtype=float,
    )

    fitted = _fit_svensson(
        spot
    )

    assert fitted[
        "beta_0"
    ] > 0.0

    assert fitted[
        "lambda_1"
    ] > 0.0

    assert fitted[
        "lambda_2"
    ] > 0.0

    assert fitted[
        "rmse"
    ] < 0.001
