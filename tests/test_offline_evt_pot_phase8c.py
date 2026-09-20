from __future__ import annotations

import numpy as np

from agent.offline_evt_pot import (
    EvtPotRiskSkill,
    _coverage_tests,
    _evt_var_es,
    _lower_quantile,
)


def test_evt_skill_matches() -> None:
    skill = EvtPotRiskSkill()

    assert skill.matches(
        instruction=(
            "Estimate tail-risk VaR and expected shortfall with EVT "
            "POT-GPD and GARCH-EVT, then perform a rolling backtest."
        ),
        task_dir=None,
    )


def test_lower_quantile_is_not_linear_interpolation() -> None:
    values = np.asarray(
        [
            1.0,
            2.0,
            3.0,
            100.0,
        ]
    )

    assert (
        _lower_quantile(
            values,
            0.75,
        )
        == 3.0
    )


def test_evt_formula_matches_reference_expression() -> None:
    var, es = _evt_var_es(
        probability=0.99,
        threshold=2.0,
        shape=0.2,
        scale=1.5,
        n_total=1000,
        n_exceedances=50,
    )

    tail_probability = (
        1000.0
        / 50.0
        * 0.01
    )

    expected_var = (
        2.0
        + 1.5
        / 0.2
        * (
            tail_probability
            ** (
                -0.2
            )
            - 1.0
        )
    )

    expected_es = (
        expected_var
        / 0.8
        + (
            1.5
            - 0.2
            * 2.0
        )
        / 0.8
    )

    assert np.isclose(
        var,
        expected_var,
    )

    assert np.isclose(
        es,
        expected_es,
    )


def test_coverage_tests_are_finite() -> None:
    indicators = np.asarray(
        [
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
        ],
        dtype=int,
    )

    result = _coverage_tests(
        indicators,
        0.01,
    )

    assert result[
        "violations"
    ] == 2

    for key in (
        "kupiec_lr",
        "kupiec_pvalue",
        "christoffersen_ind_lr",
        "christoffersen_ind_pvalue",
        "cc_lr",
        "cc_pvalue",
    ):
        assert np.isfinite(
            result[
                key
            ]
        )
