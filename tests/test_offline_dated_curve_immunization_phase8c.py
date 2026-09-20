from __future__ import annotations

import numpy as np

from agent.offline_dated_curve_immunization import (
    DatedCurveImmunizationSkill,
    _bootstrap_curve,
)


def test_dated_curve_skill_matches() -> None:
    skill = DatedCurveImmunizationSkill()

    assert skill.matches(
        instruction=(
            "Build a Treasury yield curve and immunize a pension "
            "liability with key-rate-duration hedge constraints."
        ),
        task_dir=None,
    )


def test_market_convention_bootstrap_reference_knots() -> None:
    t0 = {
        "0.25": 0.0543,
        "0.5": 0.0547,
        "1.0": 0.0540,
        "2.0": 0.0487,
        "3.0": 0.0449,
        "5.0": 0.0413,
        "7.0": 0.0397,
        "10.0": 0.0381,
        "20.0": 0.0406,
        "30.0": 0.0385,
    }

    t1 = {
        "0.25": 0.0563,
        "0.5": 0.0558,
        "1.0": 0.0543,
        "2.0": 0.0506,
        "3.0": 0.0482,
        "5.0": 0.0469,
        "7.0": 0.0473,
        "10.0": 0.0470,
        "20.0": 0.0505,
        "30.0": 0.0486,
    }

    c0 = _bootstrap_curve(
        t0
    )

    c1 = _bootstrap_curve(
        t1
    )

    z0 = (
        c0.g_values
        / c0.maturities
    )

    z1 = (
        c1.g_values
        / c1.maturities
    )

    by0 = dict(
        zip(
            c0.maturities,
            z0,
        )
    )

    by1 = dict(
        zip(
            c1.maturities,
            z1,
        )
    )

    assert np.isclose(
        by0[
            0.25
        ],
        0.053935,
        atol=1e-4,
    )

    assert np.isclose(
        by0[
            10.0
        ],
        0.037238,
        atol=1e-4,
    )

    assert np.isclose(
        by1[
            0.25
        ],
        0.055907,
        atol=1e-4,
    )

    assert np.isclose(
        by1[
            20.0
        ],
        0.050930,
        atol=1e-4,
    )
