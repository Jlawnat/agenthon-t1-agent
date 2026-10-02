from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ReturnProcessSelection:
    selected_process: str
    parameters: dict[str, dict[str, float]]
    aic: dict[str, float]
    expected_return: float | None
    sign: int
    conviction: float
    history_count: int


def gaussian_negative_log_likelihood(
    observations: np.ndarray,
    mean: float | np.ndarray,
    sigma: float,
    *,
    sigma_floor: float = 1e-6,
) -> float:
    """Gaussian negative log likelihood with a volatility floor."""
    values = np.asarray(observations, dtype=float)
    scale = max(float(sigma), float(sigma_floor))
    return float(np.sum(0.5 * np.log(2.0 * np.pi * scale * scale) + ((values - mean) ** 2) / (2.0 * scale * scale)))


def select_return_process(
    returns: Sequence[float],
    *,
    minimum_history: int,
    holding_period_days: int,
    conviction_scale: float,
    jump_threshold_sigma: float,
) -> ReturnProcessSelection:
    """Select martingale, GBM, OU, or two-component jump diffusion by AIC."""
    values = np.asarray(returns, dtype=float).ravel()
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < int(minimum_history):
        return ReturnProcessSelection("insufficient_history", {}, {}, None, 0, 0.0, n)
    if n < 2 or holding_period_days <= 0 or conviction_scale <= 0.0 or jump_threshold_sigma < 0.0:
        raise ValueError("process selection parameters are invalid")

    floor = 1e-6
    full_mean = float(values.mean())
    full_sigma = max(float(values.std(ddof=0)), floor)
    parameters: dict[str, dict[str, float]] = {
        "martingale": {"sigma": full_sigma},
        "gbm": {"mu": full_mean, "sigma": full_sigma},
    }
    aic = {
        "martingale": 2.0 + 2.0 * gaussian_negative_log_likelihood(values, 0.0, full_sigma),
        "gbm": 4.0 + 2.0 * gaussian_negative_log_likelihood(values, full_mean, full_sigma),
    }
    expected = {"martingale": 0.0, "gbm": full_mean}

    x, y = values[:-1], values[1:]
    x_mean, y_mean = float(x.mean()), float(y.mean())
    x_var = float(np.mean((x - x_mean) ** 2))
    phi = 0.0 if x_var <= 1e-12 else float(np.mean((x - x_mean) * (y - y_mean)) / x_var)
    phi = min(max(phi, 1e-4), 0.999)
    intercept = y_mean - phi * x_mean
    ou_mean = intercept / max(1.0 - phi, 1e-6)
    residuals = y - (intercept + phi * x)
    ou_sigma = max(float(np.sqrt(np.mean(residuals**2))), floor)
    theta = -math.log(phi)
    stationary_sigma = ou_sigma / max(math.sqrt(max(1.0 - phi * phi, 1e-12)), floor)
    parameters["ou"] = {
        "intercept": intercept,
        "phi": phi,
        "theta": theta,
        "mu": ou_mean,
        "sigma": ou_sigma,
        "half_life_days": math.log(2.0) / max(theta, 1e-6) * float(holding_period_days),
    }
    aic["ou"] = 6.0 + 2.0 * (
        gaussian_negative_log_likelihood(y, intercept + phi * x, ou_sigma)
        + gaussian_negative_log_likelihood(values[:1], ou_mean, stationary_sigma)
    )
    expected["ou"] = float(intercept + phi * values[-1])

    jump_mask = np.abs(values - full_mean) > float(jump_threshold_sigma) * full_sigma
    jump_intensity = float(jump_mask.mean())
    if jump_mask.any():
        jump_mean = float(values[jump_mask].mean() - full_mean)
        jump_std = max(float(values[jump_mask].std(ddof=0)), floor)
    else:
        jump_mean, jump_std = 0.0, full_sigma
    diffusion = values[~jump_mask]
    diffusion_mean = float(diffusion.mean()) if diffusion.size else full_mean
    diffusion_sigma = max(float(diffusion.std(ddof=0)), floor) if diffusion.size else full_sigma
    mixture_weight = min(max(jump_intensity, 1e-6), 1.0 - 1e-6)
    base_pdf = np.exp(-0.5 * ((values - diffusion_mean) / diffusion_sigma) ** 2) / (math.sqrt(2.0 * math.pi) * diffusion_sigma)
    jump_scale = math.sqrt(diffusion_sigma**2 + jump_std**2)
    jump_pdf = np.exp(-0.5 * ((values - diffusion_mean - jump_mean) / jump_scale) ** 2) / (math.sqrt(2.0 * math.pi) * jump_scale)
    mixture_nll = float(-np.log(np.maximum((1.0 - mixture_weight) * base_pdf + mixture_weight * jump_pdf, 1e-15)).sum())
    name = "merton_jump_diffusion"
    parameters[name] = {
        "mu": diffusion_mean,
        "sigma": diffusion_sigma,
        "jump_intensity": jump_intensity,
        "jump_mean": jump_mean,
        "jump_std": jump_std,
    }
    aic[name] = 10.0 + 2.0 * mixture_nll
    expected[name] = float(diffusion_mean + jump_intensity * jump_mean)

    ordered = sorted(aic, key=lambda candidate: (aic[candidate], candidate))
    winner = ordered[0]
    gap = max(aic[ordered[1]] - aic[winner], 0.0)
    conviction = float(1.0 / (1.0 + math.exp(-gap / float(conviction_scale))))
    forecast = expected[winner]
    sign = 1 if forecast > 0.0 else -1 if forecast < 0.0 else 0
    return ReturnProcessSelection(winner, parameters, aic, float(forecast), sign, conviction, n)
