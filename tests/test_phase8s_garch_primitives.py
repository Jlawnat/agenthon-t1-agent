from __future__ import annotations

import math

import numpy as np
import pytest

from agent.qf_primitives import (
    fit_garch11_zero_mean,
    garch11_forecast_variance,
)


def _synthetic_returns(
    *,
    n: int = 800,
    seed: int = 1729,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    omega = 2.0e-6
    alpha = 0.07
    beta = 0.90

    returns = np.zeros(n, dtype=float)
    variance = np.empty(n, dtype=float)

    variance[0] = (
        omega
        / (1.0 - alpha - beta)
    )

    for i in range(1, n):
        shock = rng.normal()

        returns[i - 1] = (
            math.sqrt(
                max(
                    variance[i - 1],
                    0.0,
                )
            )
            * shock
        )

        variance[i] = (
            omega
            + alpha * returns[i - 1] ** 2
            + beta * variance[i - 1]
        )

    returns[-1] = (
        math.sqrt(
            max(
                variance[-1],
                0.0,
            )
        )
        * rng.normal()
    )

    return returns


def test_fit_garch11_zero_mean_returns_generic_parameters() -> None:
    returns = _synthetic_returns()

    result = fit_garch11_zero_mean(
        returns,
    )

    assert set(result) >= {
        "omega",
        "alpha",
        "beta",
        "persistence",
        "long_run_variance",
        "conditional_variance",
        "convergence_flag",
    }

    assert result["omega"] > 0.0
    assert 0.0 <= result["alpha"] < 1.0
    assert 0.0 <= result["beta"] < 1.0

    assert result["persistence"] == pytest.approx(
        result["alpha"]
        + result["beta"]
    )

    cond = np.asarray(
        result["conditional_variance"],
        dtype=float,
    )

    assert cond.shape == returns.shape
    assert np.all(np.isfinite(cond))
    assert np.all(cond >= 0.0)

    if result["persistence"] < 1.0:
        assert result["long_run_variance"] == pytest.approx(
            result["omega"]
            / (
                1.0
                - result["persistence"]
            )
        )


def test_fit_garch11_zero_mean_ignores_nonfinite_values() -> None:
    clean = _synthetic_returns(
        n=300,
    )

    dirty = np.concatenate(
        [
            clean[:100],
            np.array(
                [
                    np.nan,
                    np.inf,
                    -np.inf,
                ]
            ),
            clean[100:],
        ]
    )

    clean_fit = fit_garch11_zero_mean(
        clean,
    )
    dirty_fit = fit_garch11_zero_mean(
        dirty,
    )

    assert dirty_fit["omega"] == pytest.approx(
        clean_fit["omega"],
        rel=1e-10,
        abs=1e-14,
    )
    assert dirty_fit["alpha"] == pytest.approx(
        clean_fit["alpha"],
        rel=1e-10,
        abs=1e-14,
    )
    assert dirty_fit["beta"] == pytest.approx(
        clean_fit["beta"],
        rel=1e-10,
        abs=1e-14,
    )


def test_fit_garch11_zero_mean_requires_enough_observations() -> None:
    with pytest.raises(
        ValueError,
        match="at least",
    ):
        fit_garch11_zero_mean(
            [0.01, -0.01, 0.005],
            min_observations=50,
        )


def test_garch11_forecast_variance_matches_manual_recursion() -> None:
    omega = 2.0e-6
    alpha = 0.08
    beta = 0.89

    last_return = -0.012
    last_variance = 9.0e-5

    result = garch11_forecast_variance(
        omega=omega,
        alpha=alpha,
        beta=beta,
        last_return=last_return,
        last_variance=last_variance,
        horizon=5,
    )

    expected = np.empty(
        5,
        dtype=float,
    )

    expected[0] = (
        omega
        + alpha * last_return ** 2
        + beta * last_variance
    )

    persistence = alpha + beta

    for i in range(1, 5):
        expected[i] = (
            omega
            + persistence
            * expected[i - 1]
        )

    actual = np.asarray(
        result["variance_path"],
        dtype=float,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-15,
    )

    assert result[
        "aggregate_variance"
    ] == pytest.approx(
        float(np.sum(expected))
    )


def test_garch11_forecast_variance_validates_inputs() -> None:
    with pytest.raises(ValueError):
        garch11_forecast_variance(
            omega=-1.0,
            alpha=0.1,
            beta=0.8,
            last_return=0.01,
            last_variance=0.0001,
            horizon=5,
        )

    with pytest.raises(ValueError):
        garch11_forecast_variance(
            omega=1e-6,
            alpha=0.1,
            beta=0.8,
            last_return=0.01,
            last_variance=0.0001,
            horizon=0,
        )
