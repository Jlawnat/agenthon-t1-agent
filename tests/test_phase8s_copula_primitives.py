from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from agent.offline_copula_fitting import (
    _pseudo as old_pseudo,
)

from agent.offline_copula_sampling import (
    _sample_gaussian as old_sample_gaussian,
    _sample_student_t as old_sample_student_t,
)

from agent.qf_primitives import (
    pseudo_observations,
    sample_gaussian_copula,
    sample_student_t_copula,
)


def test_pseudo_observations_match_phase8r() -> None:
    values = np.asarray(
        [
            3.0,
            1.0,
            2.0,
            2.0,
            7.0,
            -1.0,
        ],
        dtype=float,
    )

    old = old_pseudo(
        values
    )

    new = pseudo_observations(
        values
    )

    np.testing.assert_allclose(
        new,
        old,
        rtol=0.0,
        atol=0.0,
    )


def test_pseudo_observations_use_average_ranks_for_ties() -> None:
    values = np.asarray(
        [
            10.0,
            10.0,
            20.0,
            30.0,
        ],
        dtype=float,
    )

    expected = (
        stats.rankdata(
            values
        )
        / (
            len(values)
            + 1.0
        )
    )

    actual = pseudo_observations(
        values
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_pseudo_observations_stay_inside_open_unit_interval() -> None:
    values = np.arange(
        1.0,
        21.0,
    )

    result = pseudo_observations(
        values
    )

    assert np.all(
        result > 0.0
    )

    assert np.all(
        result < 1.0
    )


def test_pseudo_observations_validate_input() -> None:
    with pytest.raises(ValueError):
        pseudo_observations(
            []
        )

    with pytest.raises(ValueError):
        pseudo_observations(
            [
                1.0,
                np.nan,
                2.0,
            ]
        )


@pytest.mark.parametrize(
    ("n", "rho", "seed"),
    [
        (
            10,
            0.0,
            42,
        ),
        (
            250,
            0.35,
            1234,
        ),
        (
            500,
            -0.60,
            9876,
        ),
    ],
)
def test_gaussian_copula_matches_phase8r_exactly(
    n: int,
    rho: float,
    seed: int,
) -> None:
    old_rng = np.random.default_rng(
        seed
    )

    new_rng = np.random.default_rng(
        seed
    )

    old_u1, old_u2 = old_sample_gaussian(
        n,
        rho,
        old_rng,
    )

    new_u1, new_u2 = sample_gaussian_copula(
        n=n,
        rho=rho,
        rng=new_rng,
    )

    np.testing.assert_array_equal(
        new_u1,
        old_u1,
    )

    np.testing.assert_array_equal(
        new_u2,
        old_u2,
    )


def test_gaussian_copula_outputs_valid_uniforms() -> None:
    rng = np.random.default_rng(
        2468
    )

    u1, u2 = sample_gaussian_copula(
        n=1000,
        rho=0.5,
        rng=rng,
    )

    assert u1.shape == (
        1000,
    )

    assert u2.shape == (
        1000,
    )

    assert np.all(
        (u1 > 0.0)
        & (u1 < 1.0)
    )

    assert np.all(
        (u2 > 0.0)
        & (u2 < 1.0)
    )


@pytest.mark.parametrize(
    ("n", "rho", "degrees_of_freedom", "seed"),
    [
        (
            10,
            0.0,
            4,
            42,
        ),
        (
            250,
            0.40,
            6,
            1234,
        ),
        (
            500,
            -0.55,
            10,
            9876,
        ),
    ],
)
def test_student_t_copula_matches_phase8r_exactly(
    n: int,
    rho: float,
    degrees_of_freedom: int,
    seed: int,
) -> None:
    old_rng = np.random.default_rng(
        seed
    )

    new_rng = np.random.default_rng(
        seed
    )

    old_u1, old_u2 = old_sample_student_t(
        n,
        rho,
        degrees_of_freedom,
        old_rng,
    )

    new_u1, new_u2 = sample_student_t_copula(
        n=n,
        rho=rho,
        degrees_of_freedom=degrees_of_freedom,
        rng=new_rng,
    )

    np.testing.assert_array_equal(
        new_u1,
        old_u1,
    )

    np.testing.assert_array_equal(
        new_u2,
        old_u2,
    )


def test_student_t_copula_outputs_valid_uniforms() -> None:
    rng = np.random.default_rng(
        1357
    )

    u1, u2 = sample_student_t_copula(
        n=1000,
        rho=0.45,
        degrees_of_freedom=5,
        rng=rng,
    )

    assert u1.shape == (
        1000,
    )

    assert u2.shape == (
        1000,
    )

    assert np.all(
        (u1 > 0.0)
        & (u1 < 1.0)
    )

    assert np.all(
        (u2 > 0.0)
        & (u2 < 1.0)
    )


@pytest.mark.parametrize(
    "rho",
    [
        -1.01,
        1.01,
    ],
)
def test_copula_sampling_rejects_invalid_correlation(
    rho: float,
) -> None:
    rng = np.random.default_rng(
        1
    )

    with pytest.raises(ValueError):
        sample_gaussian_copula(
            n=10,
            rho=rho,
            rng=rng,
        )


def test_copula_sampling_validates_sample_size_and_df() -> None:
    rng = np.random.default_rng(
        1
    )

    with pytest.raises(ValueError):
        sample_gaussian_copula(
            n=0,
            rho=0.2,
            rng=rng,
        )

    with pytest.raises(ValueError):
        sample_student_t_copula(
            n=10,
            rho=0.2,
            degrees_of_freedom=0,
            rng=rng,
        )
