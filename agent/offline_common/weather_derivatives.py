from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class OuFit:
    kappa: float
    theta: float
    sigma: float
    intercept: float
    slope: float


@dataclass(frozen=True)
class WeatherOptionMonteCarlo:
    price: float
    standard_error: float
    index_values: np.ndarray
    payoffs: np.ndarray


def heating_degree_days(temperatures: Sequence[float], base_temperature: float) -> np.ndarray:
    """Daily heating-degree contributions for average temperatures."""
    values = np.asarray(temperatures, dtype=float)
    if not np.all(np.isfinite(values)) or not np.isfinite(base_temperature):
        raise ValueError("temperatures must be finite")
    return np.maximum(float(base_temperature) - values, 0.0)


def fit_ou_differences(residuals: Sequence[float], *, time_step: float, adjacent_mask: Sequence[bool] | None = None) -> OuFit:
    """Fit ``dx=alpha+beta*x`` and map it to OU parameters."""
    values = np.asarray(residuals, dtype=float).ravel()
    if values.size < 3 or time_step <= 0.0 or not np.all(np.isfinite(values)):
        raise ValueError("OU fit requires finite residual history and positive time step")
    mask = np.ones(values.size - 1, dtype=bool) if adjacent_mask is None else np.asarray(adjacent_mask, dtype=bool)
    if mask.shape != (values.size - 1,) or mask.sum() < 2:
        raise ValueError("adjacent_mask has invalid shape or too few pairs")
    lagged = values[:-1][mask]
    changes = np.diff(values)[mask]
    design = np.column_stack([np.ones(lagged.size), lagged])
    intercept, slope = np.linalg.lstsq(design, changes, rcond=None)[0]
    kappa = -float(slope) / time_step
    if kappa <= 0.0:
        raise ValueError("estimated OU mean reversion must be positive")
    theta = float(intercept) / (kappa * time_step)
    errors = changes - design @ np.array([intercept, slope])
    sigma = float(errors.std(ddof=1) / math.sqrt(time_step))
    return OuFit(kappa, theta, sigma, float(intercept), float(slope))


def simulate_hdd_option(
    seasonal_temperatures: Sequence[float],
    *,
    base_temperature: float,
    strike: float,
    tick_value: float,
    rate: float,
    time_to_payment: float,
    kappa: float,
    theta: float,
    sigma: float,
    paths: int,
    seed: int,
    time_step: float = 1.0 / 365.0,
    normal_draws: np.ndarray | None = None,
) -> WeatherOptionMonteCarlo:
    """Price an HDD call using the exact OU transition and common random numbers."""
    seasonal = np.asarray(seasonal_temperatures, dtype=float).ravel()
    if seasonal.size == 0 or paths < 2 or kappa <= 0.0 or sigma < 0.0:
        raise ValueError("invalid weather simulation inputs")
    draws = np.asarray(normal_draws, dtype=float) if normal_draws is not None else np.random.default_rng(seed).standard_normal((paths, seasonal.size))
    if draws.shape != (paths, seasonal.size):
        raise ValueError("normal_draws shape must equal paths by days")
    decay = math.exp(-kappa * time_step)
    innovation = sigma * math.sqrt((1.0 - math.exp(-2.0 * kappa * time_step)) / (2.0 * kappa))
    state = np.zeros(paths, dtype=float)
    index_values = np.zeros(paths, dtype=float)
    for day, seasonal_temperature in enumerate(seasonal):
        index_values += np.maximum(base_temperature - (seasonal_temperature + state), 0.0)
        state = state * decay + theta * (1.0 - decay) + innovation * draws[:, day]
    payoffs = np.maximum(index_values - strike, 0.0) * tick_value
    discounted = payoffs * math.exp(-rate * time_to_payment)
    return WeatherOptionMonteCarlo(float(discounted.mean()), float(discounted.std(ddof=1) / math.sqrt(paths)), index_values, payoffs)


def central_difference_sensitivity(price_up: float, price_down: float, bump_size: float) -> float:
    """Central bump-and-revalue sensitivity."""
    if bump_size <= 0.0:
        raise ValueError("bump_size must be positive")
    return float((price_up - price_down) / (2.0 * bump_size))
