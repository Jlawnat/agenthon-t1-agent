from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from agent.offline_dual_curve import (
    DualCurveBootstrapSkill,
    act_360,
    thirty_360,
    log_linear_df,
)


def test_dual_curve_primitives() -> None:
    start = date(
        2024,
        3,
        19,
    )

    end = date(
        2024,
        9,
        19,
    )

    assert np.isclose(
        thirty_360(
            start,
            end,
        ),
        0.5,
    )

    assert act_360(
        start,
        end,
    ) > 0.5

    pairs = [
        (
            1.0,
            np.exp(
                -0.04
            ),
        ),
        (
            2.0,
            np.exp(
                -0.10
            ),
        ),
    ]

    interpolated = log_linear_df(
        pairs,
        1.5,
    )

    expected = np.exp(
        0.5
        * np.log(
            pairs[
                0
            ][
                1
            ]
        )
        + 0.5
        * np.log(
            pairs[
                1
            ][
                1
            ]
        )
    )

    assert np.isclose(
        interpolated,
        expected,
    )


def test_dual_curve_skill_matches() -> None:
    skill = DualCurveBootstrapSkill()

    assert skill.matches(
        instruction=(
            "Bootstrap an OIS discount curve and a 3M LIBOR "
            "projection curve in a dual-curve swap framework."
        ),
        task_dir=Path("."),
    )
