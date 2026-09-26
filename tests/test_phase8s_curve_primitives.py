from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agent.offline_curve_immunization import (
    bootstrap_discount_factors as old_bootstrap,
)

from agent.offline_dual_curve import (
    log_linear_df as old_log_linear_df,
)

from agent.qf_primitives import (
    bootstrap_annual_par_discount_factors,
    continuous_zero_rate_from_discount_factor,
    discount_factor_from_continuous_zero_rate,
    log_linear_discount_factor,
)


@pytest.mark.parametrize(
    ("rate", "maturity"),
    [
        (0.01, 0.25),
        (0.025, 1.0),
        (0.04, 5.0),
        (-0.005, 2.5),
    ],
)
def test_continuous_zero_discount_round_trip(
    rate: float,
    maturity: float,
) -> None:
    discount = (
        discount_factor_from_continuous_zero_rate(
            zero_rate=rate,
            maturity=maturity,
        )
    )

    expected = math.exp(
        -rate * maturity
    )

    assert discount == pytest.approx(
        expected,
        rel=0.0,
        abs=1e-15,
    )

    recovered = (
        continuous_zero_rate_from_discount_factor(
            discount_factor=discount,
            maturity=maturity,
        )
    )

    assert recovered == pytest.approx(
        rate,
        rel=0.0,
        abs=1e-14,
    )


def test_zero_rate_conversion_validates_inputs() -> None:
    with pytest.raises(ValueError):
        continuous_zero_rate_from_discount_factor(
            discount_factor=0.0,
            maturity=1.0,
        )

    with pytest.raises(ValueError):
        continuous_zero_rate_from_discount_factor(
            discount_factor=0.95,
            maturity=0.0,
        )

    with pytest.raises(ValueError):
        discount_factor_from_continuous_zero_rate(
            zero_rate=0.03,
            maturity=-1.0,
        )


@pytest.mark.parametrize(
    "target",
    [
        0.25,
        0.5,
        1.5,
        3.0,
        7.0,
    ],
)
def test_log_linear_discount_factor_matches_phase8r(
    target: float,
) -> None:
    pillars = [
        (0.5, 0.9900),
        (1.0, 0.9750),
        (2.0, 0.9400),
        (5.0, 0.8400),
    ]

    old = old_log_linear_df(
        pillars,
        target,
    )

    new = log_linear_discount_factor(
        pillars=pillars,
        maturity=target,
    )

    assert new == pytest.approx(
        old,
        rel=0.0,
        abs=1e-14,
    )


def test_log_linear_discount_factor_at_zero() -> None:
    assert log_linear_discount_factor(
        pillars=[
            (1.0, 0.97),
            (2.0, 0.93),
        ],
        maturity=0.0,
    ) == pytest.approx(1.0)


def test_log_linear_discount_factor_validates_curve() -> None:
    with pytest.raises(ValueError):
        log_linear_discount_factor(
            pillars=[],
            maturity=1.0,
        )

    with pytest.raises(ValueError):
        log_linear_discount_factor(
            pillars=[
                (1.0, 0.97),
                (1.0, 0.96),
            ],
            maturity=1.0,
        )

    with pytest.raises(ValueError):
        log_linear_discount_factor(
            pillars=[
                (1.0, -0.97),
            ],
            maturity=1.0,
        )


def test_annual_par_bootstrap_matches_phase8r() -> None:
    par_yields = np.asarray(
        [
            0.0200,
            0.0225,
            0.0250,
            0.0275,
            0.0300,
        ],
        dtype=float,
    )

    frame = pd.DataFrame(
        {
            "maturity_years": np.arange(
                1,
                len(par_yields) + 1,
            ),
            "par_yield": par_yields,
        }
    )

    old = old_bootstrap(
        frame
    )

    new = (
        bootstrap_annual_par_discount_factors(
            par_yields
        )
    )

    np.testing.assert_allclose(
        np.asarray(
            new["discount_factors"],
            dtype=float,
        ),
        old[
            "discount_factor"
        ].to_numpy(
            dtype=float
        ),
        rtol=0.0,
        atol=1e-15,
    )

    np.testing.assert_allclose(
        np.asarray(
            new["zero_rates"],
            dtype=float,
        ),
        old[
            "spot_rate"
        ].to_numpy(
            dtype=float
        ),
        rtol=0.0,
        atol=1e-15,
    )

    np.testing.assert_allclose(
        np.asarray(
            new["forward_rates"],
            dtype=float,
        ),
        old[
            "forward_rate"
        ].to_numpy(
            dtype=float
        ),
        rtol=0.0,
        atol=1e-15,
    )


def test_annual_par_bootstrap_reprices_each_par_bond() -> None:
    par_yields = np.asarray(
        [
            0.02,
            0.025,
            0.03,
            0.0325,
        ],
        dtype=float,
    )

    result = (
        bootstrap_annual_par_discount_factors(
            par_yields
        )
    )

    discounts = np.asarray(
        result["discount_factors"],
        dtype=float,
    )

    for maturity, coupon in enumerate(
        par_yields,
        start=1,
    ):
        price = (
            coupon
            * float(
                np.sum(
                    discounts[
                        : maturity - 1
                    ]
                )
            )
            + (
                1.0
                + coupon
            )
            * discounts[
                maturity - 1
            ]
        )

        assert price == pytest.approx(
            1.0,
            rel=0.0,
            abs=1e-14,
        )


def test_annual_par_bootstrap_rejects_invalid_yields() -> None:
    with pytest.raises(ValueError):
        bootstrap_annual_par_discount_factors(
            []
        )

    with pytest.raises(ValueError):
        bootstrap_annual_par_discount_factors(
            [
                0.02,
                float("nan"),
            ]
        )
