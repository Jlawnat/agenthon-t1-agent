from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy import optimize

from agent.qf_primitives import black_scholes_price


@dataclass(frozen=True)
class MertonJumpCalibration:
    diffusion_volatility: float
    jump_intensity: float
    jump_mean: float
    jump_volatility: float
    log_likelihood: float


def merton_jump_negative_log_likelihood(
    log_returns: Sequence[float],
    *,
    dt: float,
    drift: float,
    diffusion_volatility: float,
    jump_intensity: float,
    jump_mean: float,
    jump_volatility: float,
    max_jumps: int = 15,
) -> float:
    """Negative log-likelihood under a Merton normal jump mixture."""
    values = np.asarray(
        log_returns,
        dtype=float,
    ).ravel()

    if (
        values.size == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "log_returns must contain finite observations."
        )

    step = float(dt)
    mu = float(drift)
    sigma = float(diffusion_volatility)
    intensity = float(jump_intensity)
    mu_jump = float(jump_mean)
    sigma_jump = float(jump_volatility)

    if step <= 0.0:
        raise ValueError(
            "dt must be positive."
        )

    if (
        sigma <= 0.0
        or intensity <= 0.0
        or sigma_jump <= 0.0
    ):
        raise ValueError(
            "volatilities and jump intensity must be positive."
        )

    if (
        isinstance(max_jumps, bool)
        or not isinstance(max_jumps, int)
        or max_jumps < 0
    ):
        raise ValueError(
            "max_jumps must be a non-negative integer."
        )

    jump_compensator = (
        math.exp(
            mu_jump
            + 0.5
            * sigma_jump
            * sigma_jump
        )
        - 1.0
    )

    conditional_drift = (
        mu
        - 0.5
        * sigma
        * sigma
        - intensity
        * jump_compensator
    ) * step

    log_likelihood = 0.0

    for observation in values:
        mixture_probability = 0.0
        log_poisson_weight = (
            -intensity
            * step
        )

        for jump_count in range(
            max_jumps + 1
        ):
            mean = (
                conditional_drift
                + jump_count
                * mu_jump
            )

            variance = (
                sigma
                * sigma
                * step
                + jump_count
                * sigma_jump
                * sigma_jump
            )

            density = (
                math.exp(
                    -0.5
                    * (
                        float(observation)
                        - mean
                    )
                    ** 2
                    / variance
                )
                / math.sqrt(
                    2.0
                    * math.pi
                    * variance
                )
            )

            mixture_probability += (
                math.exp(
                    log_poisson_weight
                )
                * density
            )

            if jump_count < max_jumps:
                log_poisson_weight += (
                    math.log(
                        intensity
                        * step
                    )
                    - math.log(
                        jump_count
                        + 1
                    )
                )

        if mixture_probability <= 0.0:
            return math.inf

        log_likelihood += math.log(
            mixture_probability
        )

    return -float(log_likelihood)


def calibrate_merton_jump_diffusion(
    log_returns: Sequence[float],
    *,
    dt: float,
    drift: float,
    bounds: Sequence[tuple[float, float]],
    restarts: int = 8,
    seed: int = 0,
    max_jumps: int = 15,
) -> MertonJumpCalibration:
    """Calibrate Merton diffusion/jump parameters by multi-start MLE."""
    values = np.asarray(
        log_returns,
        dtype=float,
    ).ravel()

    if (
        values.size < 2
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "at least two finite log returns are required."
        )

    parameter_bounds = tuple(
        (
            float(lower),
            float(upper),
        )
        for lower, upper in bounds
    )

    if len(parameter_bounds) != 4:
        raise ValueError(
            "bounds must contain four parameter intervals."
        )

    for lower, upper in parameter_bounds:
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower >= upper
        ):
            raise ValueError(
                "each parameter bound must satisfy lower < upper."
            )

    if (
        isinstance(restarts, bool)
        or not isinstance(restarts, int)
        or restarts < 1
    ):
        raise ValueError(
            "restarts must be a positive integer."
        )

    rng = np.random.RandomState(
        int(seed)
    )

    best_result = None
    best_nll = math.inf

    for _ in range(restarts):
        initial = np.array(
            [
                rng.uniform(
                    lower,
                    upper,
                )
                for lower, upper
                in parameter_bounds
            ],
            dtype=float,
        )

        result = optimize.minimize(
            lambda parameters: (
                merton_jump_negative_log_likelihood(
                    values,
                    dt=dt,
                    drift=drift,
                    diffusion_volatility=float(
                        parameters[0]
                    ),
                    jump_intensity=float(
                        parameters[1]
                    ),
                    jump_mean=float(
                        parameters[2]
                    ),
                    jump_volatility=float(
                        parameters[3]
                    ),
                    max_jumps=max_jumps,
                )
            ),
            initial,
            method="L-BFGS-B",
            bounds=parameter_bounds,
        )

        if (
            np.isfinite(result.fun)
            and float(result.fun)
            < best_nll
        ):
            best_nll = float(
                result.fun
            )
            best_result = result

    if best_result is None:
        raise RuntimeError(
            "Merton jump-diffusion calibration failed."
        )

    sigma, intensity, mu_jump, sigma_jump = (
        float(value)
        for value in best_result.x
    )

    return MertonJumpCalibration(
        diffusion_volatility=sigma,
        jump_intensity=intensity,
        jump_mean=mu_jump,
        jump_volatility=sigma_jump,
        log_likelihood=-best_nll,
    )


def merton_call_price(
    *,
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    diffusion_volatility: float,
    jump_intensity: float,
    jump_mean: float,
    jump_volatility: float,
    n_terms: int = 50,
) -> float:
    """European call price from the Merton Poisson-mixture series."""
    s = float(spot)
    k = float(strike)
    r = float(rate)
    tau = float(maturity)
    sigma = float(diffusion_volatility)
    intensity = float(jump_intensity)
    mu_jump = float(jump_mean)
    sigma_jump = float(jump_volatility)

    if s <= 0.0 or k <= 0.0:
        raise ValueError(
            "spot and strike must be positive."
        )

    if tau <= 0.0:
        raise ValueError(
            "maturity must be positive."
        )

    if (
        sigma <= 0.0
        or intensity < 0.0
        or sigma_jump < 0.0
    ):
        raise ValueError(
            "invalid volatility or jump intensity."
        )

    if (
        isinstance(n_terms, bool)
        or not isinstance(n_terms, int)
        or n_terms < 0
    ):
        raise ValueError(
            "n_terms must be a non-negative integer."
        )

    jump_compensator = (
        math.exp(
            mu_jump
            + 0.5
            * sigma_jump
            * sigma_jump
        )
        - 1.0
    )

    price = 0.0
    log_poisson_weight = (
        -intensity
        * tau
    )

    for jump_count in range(
        n_terms + 1
    ):
        effective_volatility = math.sqrt(
            sigma
            * sigma
            + jump_count
            * sigma_jump
            * sigma_jump
            / tau
        )

        effective_rate = (
            r
            - intensity
            * jump_compensator
            + jump_count
            * math.log(
                1.0
                + jump_compensator
            )
            / tau
        )

        price += (
            math.exp(
                log_poisson_weight
            )
            * black_scholes_price(
                s,
                k,
                effective_rate,
                0.0,
                effective_volatility,
                tau,
                "call",
            )
        )

        if jump_count < n_terms:
            if intensity == 0.0:
                break

            log_poisson_weight += (
                math.log(
                    intensity
                    * tau
                )
                - math.log(
                    jump_count
                    + 1
                )
            )

    return float(price)


def merton_total_volatility(
    diffusion_volatility: float,
    jump_intensity: float,
    jump_mean: float,
    jump_volatility: float,
) -> float:
    """Annualized standard deviation implied by diffusion plus jump variance."""
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

    if (
        sigma < 0.0
        or intensity < 0.0
        or sigma_jump < 0.0
    ):
        raise ValueError(
            "volatilities and jump intensity must be non-negative."
        )

    return float(
        math.sqrt(
            sigma
            * sigma
            + intensity
            * (
                mu_jump
                * mu_jump
                + sigma_jump
                * sigma_jump
            )
        )
    )


__all__ = (
    "MertonJumpCalibration",
    "merton_jump_negative_log_likelihood",
    "calibrate_merton_jump_diffusion",
    "merton_call_price",
    "merton_total_volatility",
)
