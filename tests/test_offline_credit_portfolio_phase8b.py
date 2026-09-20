from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_credit_portfolio import (
    CreditPortfolioRiskSkill,
    _vasicek_and_gordy,
)


def test_credit_portfolio_skill_matches() -> None:
    skill = CreditPortfolioRiskSkill()

    assert skill.matches(
        instruction=(
            "Build a credit portfolio VaR and CVaR model using "
            "Gaussian and Student-t copula default correlation "
            "for obligors with recovery assumptions."
        ),
        task_dir=Path("."),
    )


def test_vasicek_and_gordy_are_monotonic() -> None:
    pd_values = np.asarray(
        [
            0.01,
            0.02,
            0.04,
        ]
    )

    exposure = np.asarray(
        [
            1_000_000.0,
            1_000_000.0,
            1_000_000.0,
        ]
    )

    recovery = np.asarray(
        [
            0.4,
            0.4,
            0.4,
        ]
    )

    rho = np.asarray(
        [
            0.2,
            0.2,
            0.2,
        ]
    )

    result = _vasicek_and_gordy(
        pd_values=pd_values,
        exposure=exposure,
        recovery=recovery,
        rho=rho,
        confidence_levels=[
            0.95,
            0.99,
            0.999,
        ],
    )

    values = [
        result[
            alpha
        ][
            "vasicek_var"
        ]
        for alpha
        in (
            0.95,
            0.99,
            0.999,
        )
    ]

    assert values[
        0
    ] < values[
        1
    ] < values[
        2
    ]

    assert result[
        0.99
    ][
        "granularity_adjustment"
    ] > 0.0
