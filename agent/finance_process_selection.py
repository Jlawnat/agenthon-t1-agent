from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

EPS = 1e-12
SIGMA_FLOOR = 1e-6


@dataclass(frozen=True)
class ProcessSelectionResult:
    history_count: int
    selected_process: str
    martingale_aic: float | None
    gbm_mu: float | None
    gbm_sigma: float | None
    gbm_aic: float | None
    ou_theta: float | None
    ou_mu: float | None
    ou_sigma: float | None
    ou_half_life_days: float | None
    ou_aic: float | None
    merton_mu: float | None
    merton_sigma: float | None
    merton_jump_intensity: float | None
    merton_jump_mean: float | None
    merton_jump_std: float | None
    merton_aic: float | None
    process_expected_return: float | None
    process_sign: int
    process_conviction: float


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def gaussian_logpdf(
    x: np.ndarray,
    mean: float | np.ndarray,
    sigma: float,
) -> np.ndarray:
    sigma = max(float(sigma), SIGMA_FLOOR)
    return (
        -0.5 * np.log(2.0 * np.pi * sigma * sigma)
        - ((x - mean) ** 2) / (2.0 * sigma * sigma)
    )


def gaussian_nll(
    x: np.ndarray,
    mean: float | np.ndarray,
    sigma: float,
) -> float:
    return float(-gaussian_logpdf(x, mean, sigma).sum())


def select_return_process(
    returns: Sequence[float],
    *,
    minimum_history: int,
    holding_period_days: int,
    conviction_scale: float,
    jump_threshold_sigma: float,
) -> ProcessSelectionResult:
    values = np.asarray(returns, dtype=float).ravel()
    values = values[np.isfinite(values)]
    n = int(values.size)

    if n < int(minimum_history):
        return ProcessSelectionResult(
            history_count=n,
            selected_process="insufficient_history",
            martingale_aic=None,
            gbm_mu=None,
            gbm_sigma=None,
            gbm_aic=None,
            ou_theta=None,
            ou_mu=None,
            ou_sigma=None,
            ou_half_life_days=None,
            ou_aic=None,
            merton_mu=None,
            merton_sigma=None,
            merton_jump_intensity=None,
            merton_jump_mean=None,
            merton_jump_std=None,
            merton_aic=None,
            process_expected_return=None,
            process_sign=0,
            process_conviction=0.0,
        )

    martingale_sigma = max(float(values.std(ddof=0)), SIGMA_FLOOR)
    martingale_aic = 2.0 + 2.0 * gaussian_nll(
        values, 0.0, martingale_sigma
    )

    gbm_mu = float(values.mean())
    gbm_sigma = max(float(values.std(ddof=0)), SIGMA_FLOOR)
    gbm_aic = 4.0 + 2.0 * gaussian_nll(
        values, gbm_mu, gbm_sigma
    )

    x = values[:-1]
    y = values[1:]
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_var = float(((x - x_mean) ** 2).mean())
    phi_raw = (
        0.0
        if x_var <= EPS
        else float(
            ((x - x_mean) * (y - y_mean)).mean()
            / x_var
        )
    )
    phi = min(max(phi_raw, 1e-4), 0.999)
    intercept = y_mean - phi * x_mean
    ou_mu = intercept / max(1.0 - phi, 1e-6)
    residuals = y - (intercept + phi * x)
    ou_sigma = max(
        float(np.sqrt((residuals ** 2).mean())),
        SIGMA_FLOOR,
    )
    ou_theta = -math.log(phi)
    ou_half_life_days = (
        math.log(2.0)
        / max(ou_theta, 1e-6)
        * float(holding_period_days)
    )
    stationary_sigma = (
        ou_sigma
        / max(
            math.sqrt(max(1.0 - phi * phi, EPS)),
            SIGMA_FLOOR,
        )
    )
    ou_aic = 6.0 + 2.0 * (
        gaussian_nll(
            y,
            intercept + phi * x,
            ou_sigma,
        )
        + gaussian_nll(
            values[:1],
            ou_mu,
            stationary_sigma,
        )
    )
    ou_expected_return = (
        intercept + phi * float(values[-1])
    )

    full_mu = float(values.mean())
    full_sigma = max(
        float(values.std(ddof=0)),
        SIGMA_FLOOR,
    )
    threshold = (
        float(jump_threshold_sigma)
        * full_sigma
    )
    jump_mask = (
        np.abs(values - full_mu) > threshold
    )

    if jump_mask.any():
        jump_returns = values[jump_mask]
        jump_mean = float(
            jump_returns.mean() - full_mu
        )
        jump_std = max(
            float(jump_returns.std(ddof=0)),
            SIGMA_FLOOR,
        )
    else:
        jump_mean = 0.0
        jump_std = full_sigma

    diffusion = values[~jump_mask]
    if len(diffusion) == 0:
        merton_mu = full_mu
        merton_sigma = full_sigma
    else:
        merton_mu = float(diffusion.mean())
        merton_sigma = max(
            float(diffusion.std(ddof=0)),
            SIGMA_FLOOR,
        )

    jump_intensity = min(
        max(float(jump_mask.mean()), 1e-6),
        1.0 - 1e-6,
    )
    base_pdf = np.exp(
        gaussian_logpdf(
            values,
            merton_mu,
            merton_sigma,
        )
    )
    jump_pdf = np.exp(
        gaussian_logpdf(
            values,
            merton_mu + jump_mean,
            math.sqrt(
                merton_sigma * merton_sigma
                + jump_std * jump_std
            ),
        )
    )
    mixture = (
        (1.0 - jump_intensity) * base_pdf
        + jump_intensity * jump_pdf
    )
    merton_nll = float(
        -np.log(
            np.maximum(mixture, 1e-15)
        ).sum()
    )
    merton_aic = 10.0 + 2.0 * merton_nll
    merton_expected_return = (
        merton_mu
        + jump_intensity * jump_mean
    )

    candidates = [
        ("martingale", martingale_aic, 0.0),
        ("gbm", gbm_aic, gbm_mu),
        ("ou", ou_aic, float(ou_expected_return)),
        (
            "merton_jump_diffusion",
            merton_aic,
            float(merton_expected_return),
        ),
    ]
    ordered = sorted(
        candidates,
        key=lambda item: (item[1], item[0]),
    )
    best = ordered[0]

    if len(ordered) == 1:
        conviction = 1.0
    else:
        second = ordered[1]
        gap = max(
            float(second[1] - best[1]),
            0.0,
        )
        scale = max(
            float(conviction_scale),
            1e-6,
        )
        # Benchmark family convention: map a nonnegative AIC gap
        # to [0, 1] with zero gap -> zero conviction.
        conviction = max(
            0.0,
            min(
                1.0,
                2.0 * sigmoid(gap / scale) - 1.0,
            ),
        )

    sign = (
        1
        if best[2] > 0.0
        else -1
        if best[2] < 0.0
        else 0
    )

    return ProcessSelectionResult(
        history_count=n,
        selected_process=best[0],
        martingale_aic=float(martingale_aic),
        gbm_mu=gbm_mu,
        gbm_sigma=gbm_sigma,
        gbm_aic=float(gbm_aic),
        ou_theta=float(ou_theta),
        ou_mu=float(ou_mu),
        ou_sigma=float(ou_sigma),
        ou_half_life_days=float(ou_half_life_days),
        ou_aic=float(ou_aic),
        merton_mu=float(merton_mu),
        merton_sigma=float(merton_sigma),
        merton_jump_intensity=float(jump_intensity),
        merton_jump_mean=float(jump_mean),
        merton_jump_std=float(jump_std),
        merton_aic=float(merton_aic),
        process_expected_return=float(best[2]),
        process_sign=int(sign),
        process_conviction=float(conviction),
    )
