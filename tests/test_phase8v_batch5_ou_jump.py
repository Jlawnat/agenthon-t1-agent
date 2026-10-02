from __future__ import annotations

import math

import numpy as np
import pytest

from agent import offline_ou_jump as mature
from agent.offline_common import ou_jump as extracted


def _levels() -> np.ndarray:
    rng = np.random.default_rng(2026)

    values = [4.0]

    for _ in range(300):
        values.append(
            0.04
            + 0.985 * values[-1]
            + rng.normal(
                0.0,
                0.025,
            )
        )

    return np.asarray(
        values,
        dtype=float,
    )


def test_ou_ar1_fit_matches_mature() -> None:
    levels = _levels()
    dt = 1.0 / 252.0

    expected = mature._fit_ar1(
        levels,
        dt,
    )

    actual = extracted.fit_ou_ar1(
        levels,
        dt=dt,
    )

    assert np.isclose(
        actual.intercept,
        expected["intercept"],
    )

    assert np.isclose(
        actual.slope,
        expected["slope"],
    )

    assert np.isclose(
        actual.kappa,
        expected["kappa"],
    )

    assert np.isclose(
        actual.theta,
        expected["theta"],
    )

    assert np.allclose(
        actual.residuals,
        expected["residuals"],
    )


def test_ou_diffusion_volatility_matches_mature() -> None:
    residual_std = 0.025
    kappa = 3.5
    slope = 0.986

    expected = mature._continuous_sigma(
        residual_std,
        kappa,
        slope,
    )

    actual = extracted.ou_diffusion_volatility(
        residual_std,
        kappa=kappa,
        slope=slope,
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_jump_residual_fit() -> None:
    rng = np.random.default_rng(1234)

    residuals = rng.normal(
        0.0,
        0.01,
        size=200,
    )

    residuals[25] = 0.20
    residuals[150] = -0.18

    dt = 1.0 / 252.0

    result = extracted.fit_residual_jumps(
        residuals,
        dt=dt,
        threshold_std=3.0,
    )

    overall_std = np.std(
        residuals,
        ddof=1,
    )

    mask = (
        np.abs(residuals)
        > 3.0 * overall_std
    )

    jumps = residuals[mask]
    nonjumps = residuals[~mask]

    assert result.n_jumps == int(
        mask.sum()
    )

    assert np.isclose(
        result.jump_intensity,
        mask.sum()
        / (
            residuals.size
            * dt
        ),
    )

    assert np.isclose(
        result.jump_mean,
        np.mean(jumps),
    )

    assert np.isclose(
        result.jump_volatility,
        np.std(
            jumps,
            ddof=1,
        ),
    )

    assert np.isclose(
        result.nonjump_residual_std,
        np.std(
            nonjumps,
            ddof=1,
        ),
    )


def test_conditional_moments_match_mature_formula() -> None:
    x0 = 4.2
    tau = 0.75
    kappa = 1.8
    theta = 3.7
    sigma = 0.30
    intensity = 2.5
    mu_jump = -0.04
    sigma_jump = 0.08

    decay = math.exp(
        -kappa * tau
    )

    decay2 = math.exp(
        -2.0 * kappa * tau
    )

    expected_mean_ou = (
        theta
        + (
            x0 - theta
        )
        * decay
    )

    expected_var_ou = (
        sigma**2
        * (
            1.0 - decay2
        )
        / (
            2.0 * kappa
        )
    )

    expected_jump_mean = (
        intensity
        * mu_jump
        * (
            1.0 - decay
        )
        / kappa
    )

    expected_jump_var = (
        intensity
        * (
            sigma_jump**2
            + mu_jump**2
        )
        * (
            1.0 - decay2
        )
        / (
            2.0 * kappa
        )
    )

    actual = (
        extracted.ou_jump_conditional_moments(
            initial_level=x0,
            horizon=tau,
            kappa=kappa,
            theta=theta,
            diffusion_volatility=sigma,
            jump_intensity=intensity,
            jump_mean=mu_jump,
            jump_volatility=sigma_jump,
        )
    )

    assert np.isclose(
        actual.mean_ou,
        expected_mean_ou,
    )

    assert np.isclose(
        actual.variance_ou,
        expected_var_ou,
    )

    assert np.isclose(
        actual.mean_with_jumps,
        expected_mean_ou
        + expected_jump_mean,
    )

    assert np.isclose(
        actual.variance_with_jumps,
        expected_var_ou
        + expected_jump_var,
    )


def test_stationary_moments() -> None:
    kappa = 2.0
    theta = 1.5
    sigma = 0.25
    intensity = 1.8
    mu_jump = 0.03
    sigma_jump = 0.05

    actual = (
        extracted.ou_jump_stationary_moments(
            kappa=kappa,
            theta=theta,
            diffusion_volatility=sigma,
            jump_intensity=intensity,
            jump_mean=mu_jump,
            jump_volatility=sigma_jump,
        )
    )

    expected_var_ou = (
        sigma**2
        / (
            2.0 * kappa
        )
    )

    expected_mean_jump = (
        theta
        + intensity
        * mu_jump
        / kappa
    )

    expected_var_jump = (
        sigma**2
        + intensity
        * (
            mu_jump**2
            + sigma_jump**2
        )
    ) / (
        2.0 * kappa
    )

    assert np.isclose(
        actual.mean_ou,
        theta,
    )

    assert np.isclose(
        actual.variance_ou,
        expected_var_ou,
    )

    assert np.isclose(
        actual.mean_with_jumps,
        expected_mean_jump,
    )

    assert np.isclose(
        actual.variance_with_jumps,
        expected_var_jump,
    )


def test_zero_jump_conditional_moments_reduce_to_ou() -> None:
    result = (
        extracted.ou_jump_conditional_moments(
            initial_level=2.0,
            horizon=1.0,
            kappa=1.5,
            theta=1.8,
            diffusion_volatility=0.2,
        )
    )

    assert np.isclose(
        result.mean_ou,
        result.mean_with_jumps,
    )

    assert np.isclose(
        result.variance_ou,
        result.variance_with_jumps,
    )


def test_lognormal_moments() -> None:
    mean = -2.0
    variance = 0.04

    expected_mean = math.exp(
        mean
        + 0.5 * variance
    )

    expected_variance = (
        (
            math.exp(variance)
            - 1.0
        )
        * math.exp(
            2.0 * mean
            + variance
        )
    )

    actual_mean, actual_variance = (
        extracted.lognormal_moments(
            mean,
            variance,
        )
    )

    assert np.isclose(
        actual_mean,
        expected_mean,
    )

    assert np.isclose(
        actual_variance,
        expected_variance,
    )


def test_no_detected_jumps_returns_zero_jump_parameters() -> None:
    residuals = np.array(
        [
            -0.01,
            -0.005,
            0.0,
            0.004,
            0.008,
            -0.002,
        ]
    )

    result = extracted.fit_residual_jumps(
        residuals,
        dt=1.0 / 252.0,
        threshold_std=10.0,
    )

    assert result.n_jumps == 0
    assert result.jump_intensity == 0.0
    assert result.jump_mean == 0.0
    assert result.jump_volatility == 0.0


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        extracted.fit_ou_ar1(
            [1.0, 2.0],
            dt=1.0 / 252.0,
        )

    with pytest.raises(ValueError):
        extracted.ou_jump_conditional_moments(
            initial_level=1.0,
            horizon=-1.0,
            kappa=1.0,
            theta=1.0,
            diffusion_volatility=0.2,
        )

    with pytest.raises(ValueError):
        extracted.lognormal_moments(
            0.0,
            -0.1,
        )


from pathlib import Path

from agent.capability_bridge import rank_capabilities


def _task(
    tmp_path: Path,
    instruction: str,
) -> Path:
    task = tmp_path / "task"
    (task / "environment" / "data").mkdir(
        parents=True,
    )
    (task / "instruction.md").write_text(
        instruction,
        encoding="utf-8",
    )
    return task


@pytest.mark.parametrize(
    "instruction",
    (
        (
            "Calibrate an OU Process with Jumps and compute "
            "conditional and stationary moments."
        ),
        (
            "Analyze a geometric mean-reverting jump-diffusion "
            "model in log-space."
        ),
    ),
)
def test_ou_jump_language_selects_capability(
    tmp_path: Path,
    instruction: str,
) -> None:
    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "ou-jump-mean-reversion"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_plain_ou_language_does_not_select_jump_capability(
    tmp_path: Path,
) -> None:
    instruction = (
        "Fit a standard Ornstein-Uhlenbeck process "
        "without jumps."
    )

    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "ou-jump-mean-reversion"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
