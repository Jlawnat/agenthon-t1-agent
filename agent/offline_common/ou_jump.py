from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class OuAr1Fit:
    intercept: float
    slope: float
    kappa: float
    theta: float
    residuals: np.ndarray


@dataclass(frozen=True)
class JumpResidualFit:
    jump_intensity: float
    jump_mean: float
    jump_volatility: float
    n_jumps: int
    nonjump_residual_std: float


@dataclass(frozen=True)
class OuJumpMoments:
    mean_ou: float
    variance_ou: float
    mean_with_jumps: float
    variance_with_jumps: float


def fit_ou_ar1(
    levels: Sequence[float],
    *,
    dt: float,
) -> OuAr1Fit:
    """Fit an OU process through its exact AR(1) discretization."""
    values = np.asarray(
        levels,
        dtype=float,
    ).ravel()

    step = float(dt)

    if (
        values.size < 3
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "levels must contain at least three finite observations."
        )

    if step <= 0.0:
        raise ValueError(
            "dt must be positive."
        )

    x = values[:-1]
    y = values[1:]

    design = np.column_stack(
        [
            np.ones_like(x),
            x,
        ]
    )

    coefficients, *_ = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )

    intercept = float(
        coefficients[0]
    )
    slope = float(
        coefficients[1]
    )

    if not 0.0 < slope < 1.0:
        raise ValueError(
            "OU AR(1) slope must lie in (0, 1)."
        )

    fitted = (
        intercept
        + slope * x
    )

    residuals = (
        y - fitted
    )

    kappa = float(
        -math.log(slope)
        / step
    )

    theta = float(
        intercept
        / (
            1.0 - slope
        )
    )

    return OuAr1Fit(
        intercept=intercept,
        slope=slope,
        kappa=kappa,
        theta=theta,
        residuals=residuals,
    )


def ou_diffusion_volatility(
    residual_std: float,
    *,
    kappa: float,
    slope: float,
) -> float:
    """Convert discrete AR(1) innovation volatility to continuous OU volatility."""
    std = float(residual_std)
    speed = float(kappa)
    autoregressive = float(slope)

    if std < 0.0:
        raise ValueError(
            "residual_std must be non-negative."
        )

    if speed <= 0.0:
        raise ValueError(
            "kappa must be positive."
        )

    if not 0.0 < autoregressive < 1.0:
        raise ValueError(
            "slope must lie in (0, 1)."
        )

    return float(
        std
        * math.sqrt(
            2.0 * speed
            / (
                1.0
                - autoregressive
                * autoregressive
            )
        )
    )


def fit_residual_jumps(
    residuals: Sequence[float],
    *,
    dt: float,
    threshold_std: float = 3.0,
) -> JumpResidualFit:
    """Estimate Poisson jump parameters from thresholded OU residuals."""
    values = np.asarray(
        residuals,
        dtype=float,
    ).ravel()

    step = float(dt)
    threshold = float(
        threshold_std
    )

    if (
        values.size < 2
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "residuals must contain at least two finite observations."
        )

    if step <= 0.0:
        raise ValueError(
            "dt must be positive."
        )

    if threshold <= 0.0:
        raise ValueError(
            "threshold_std must be positive."
        )

    overall_std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    jump_mask = (
        np.abs(values)
        > threshold * overall_std
    )

    jumps = values[
        jump_mask
    ]

    nonjumps = values[
        ~jump_mask
    ]

    if nonjumps.size < 2:
        raise ValueError(
            "at least two non-jump residuals are required."
        )

    n_jumps = int(
        jumps.size
    )

    intensity = float(
        n_jumps
        / (
            values.size
            * step
        )
    )

    jump_mean = (
        float(np.mean(jumps))
        if n_jumps
        else 0.0
    )

    jump_volatility = (
        float(
            np.std(
                jumps,
                ddof=1,
            )
        )
        if n_jumps > 1
        else 0.0
    )

    nonjump_std = float(
        np.std(
            nonjumps,
            ddof=1,
        )
    )

    return JumpResidualFit(
        jump_intensity=intensity,
        jump_mean=jump_mean,
        jump_volatility=jump_volatility,
        n_jumps=n_jumps,
        nonjump_residual_std=nonjump_std,
    )


def ou_jump_conditional_moments(
    *,
    initial_level: float,
    horizon: float,
    kappa: float,
    theta: float,
    diffusion_volatility: float,
    jump_intensity: float = 0.0,
    jump_mean: float = 0.0,
    jump_volatility: float = 0.0,
) -> OuJumpMoments:
    """Conditional first two moments of an OU process with compound Poisson jumps."""
    x0 = float(initial_level)
    tau = float(horizon)
    speed = float(kappa)
    long_run = float(theta)
    sigma = float(
        diffusion_volatility
    )
    intensity = float(
        jump_intensity
    )
    mu_jump = float(
        jump_mean
    )
    sigma_jump = float(
        jump_volatility
    )

    if tau < 0.0:
        raise ValueError(
            "horizon must be non-negative."
        )

    if speed <= 0.0:
        raise ValueError(
            "kappa must be positive."
        )

    if (
        sigma < 0.0
        or intensity < 0.0
        or sigma_jump < 0.0
    ):
        raise ValueError(
            "volatilities and jump intensity must be non-negative."
        )

    decay = math.exp(
        -speed * tau
    )

    decay2 = math.exp(
        -2.0 * speed * tau
    )

    mean_ou = (
        long_run
        + (
            x0 - long_run
        )
        * decay
    )

    variance_ou = (
        sigma
        * sigma
        * (
            1.0 - decay2
        )
        / (
            2.0 * speed
        )
    )

    second_jump_moment = (
        mu_jump
        * mu_jump
        + sigma_jump
        * sigma_jump
    )

    jump_mean_contribution = (
        intensity
        * mu_jump
        * (
            1.0 - decay
        )
        / speed
    )

    jump_variance_contribution = (
        intensity
        * second_jump_moment
        * (
            1.0 - decay2
        )
        / (
            2.0 * speed
        )
    )

    return OuJumpMoments(
        mean_ou=float(mean_ou),
        variance_ou=float(
            variance_ou
        ),
        mean_with_jumps=float(
            mean_ou
            + jump_mean_contribution
        ),
        variance_with_jumps=float(
            variance_ou
            + jump_variance_contribution
        ),
    )


def ou_jump_stationary_moments(
    *,
    kappa: float,
    theta: float,
    diffusion_volatility: float,
    jump_intensity: float = 0.0,
    jump_mean: float = 0.0,
    jump_volatility: float = 0.0,
) -> OuJumpMoments:
    """Stationary OU moments with and without compound Poisson jumps."""
    speed = float(kappa)
    long_run = float(theta)
    sigma = float(
        diffusion_volatility
    )
    intensity = float(
        jump_intensity
    )
    mu_jump = float(
        jump_mean
    )
    sigma_jump = float(
        jump_volatility
    )

    if speed <= 0.0:
        raise ValueError(
            "kappa must be positive."
        )

    if (
        sigma < 0.0
        or intensity < 0.0
        or sigma_jump < 0.0
    ):
        raise ValueError(
            "volatilities and jump intensity must be non-negative."
        )

    variance_ou = (
        sigma
        * sigma
        / (
            2.0 * speed
        )
    )

    second_jump_moment = (
        mu_jump
        * mu_jump
        + sigma_jump
        * sigma_jump
    )

    return OuJumpMoments(
        mean_ou=long_run,
        variance_ou=float(
            variance_ou
        ),
        mean_with_jumps=float(
            long_run
            + intensity
            * mu_jump
            / speed
        ),
        variance_with_jumps=float(
            (
                sigma
                * sigma
                + intensity
                * second_jump_moment
            )
            / (
                2.0 * speed
            )
        ),
    )


def lognormal_moments(
    normal_mean: float,
    normal_variance: float,
) -> tuple[float, float]:
    """Mean and variance of exp(X) when X is normally approximated."""
    mean = float(
        normal_mean
    )
    variance = float(
        normal_variance
    )

    if variance < 0.0:
        raise ValueError(
            "normal_variance must be non-negative."
        )

    level_mean = math.exp(
        mean
        + 0.5 * variance
    )

    level_variance = (
        (
            math.exp(variance)
            - 1.0
        )
        * math.exp(
            2.0 * mean
            + variance
        )
    )

    return (
        float(level_mean),
        float(level_variance),
    )


__all__ = (
    "OuAr1Fit",
    "JumpResidualFit",
    "OuJumpMoments",
    "fit_ou_ar1",
    "ou_diffusion_volatility",
    "fit_residual_jumps",
    "ou_jump_conditional_moments",
    "ou_jump_stationary_moments",
    "lognormal_moments",
)
