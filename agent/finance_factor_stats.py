from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class OlsFit:
    coefficients: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    r_squared: float
    adjusted_r_squared: float
    standard_errors: np.ndarray
    t_statistics: np.ndarray
    p_values: np.ndarray
    degrees_of_freedom: int


def ols_with_inference(design: np.ndarray, response: Sequence[float]) -> OlsFit:
    x = np.asarray(design, dtype=float)
    y = np.asarray(response, dtype=float).ravel()
    if x.ndim != 2 or x.shape[0] != y.size:
        raise RuntimeError("OLS requires an n x p design matrix aligned with an n-vector.")
    if x.shape[0] <= x.shape[1]:
        raise RuntimeError("OLS requires more observations than coefficients.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise RuntimeError("OLS inputs contain non-finite values.")

    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ beta
    residuals = y - fitted
    ss_res = float(residuals @ residuals)
    centered = y - float(np.mean(y))
    ss_tot = float(centered @ centered)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    n, p = x.shape
    df = n - p
    adjusted = 1.0 - (1.0 - r2) * (n - 1.0) / df
    mse = ss_res / df

    xtx = x.T @ x
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)

    covariance = mse * xtx_inv
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistics = np.divide(
            beta,
            standard_errors,
            out=np.zeros_like(beta, dtype=float),
            where=standard_errors > 0.0,
        )
    p_values = 2.0 * stats.t.sf(np.abs(t_statistics), df=df)

    return OlsFit(
        coefficients=np.asarray(beta, dtype=float),
        fitted=np.asarray(fitted, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        r_squared=float(r2),
        adjusted_r_squared=float(adjusted),
        standard_errors=np.asarray(standard_errors, dtype=float),
        t_statistics=np.asarray(t_statistics, dtype=float),
        p_values=np.asarray(p_values, dtype=float),
        degrees_of_freedom=int(df),
    )


def optimal_newey_west_lag(n_observations: int) -> int:
    if n_observations <= 0:
        raise RuntimeError("Newey-West lag requires a positive observation count.")
    return int(math.floor(4.0 * (float(n_observations) / 100.0) ** (2.0 / 9.0)))


def newey_west_covariance(
    design: np.ndarray,
    residuals: Sequence[float],
    *,
    lag: int,
) -> np.ndarray:
    x = np.asarray(design, dtype=float)
    e = np.asarray(residuals, dtype=float).ravel()
    if x.ndim != 2 or x.shape[0] != e.size:
        raise RuntimeError("Newey-West inputs are misaligned.")
    if lag < 0 or lag >= x.shape[0]:
        raise RuntimeError("Newey-West lag must be in [0, n).")

    xtx = x.T @ x
    try:
        inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(xtx)

    e2 = e * e
    meat = (x * e2[:, None]).T @ x

    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1.0)
        products = e[offset:] * e[:-offset]
        gamma = (x[offset:] * products[:, None]).T @ x[:-offset]
        meat += weight * (gamma + gamma.T)

    covariance = inv @ meat @ inv
    return 0.5 * (covariance + covariance.T)


def durbin_watson(residuals: Sequence[float]) -> float:
    e = np.asarray(residuals, dtype=float).ravel()
    denominator = float(e @ e)
    if e.size < 2 or denominator <= 0.0:
        return 0.0
    return float(np.sum(np.diff(e) ** 2) / denominator)


def variance_inflation_factors(factors: np.ndarray) -> np.ndarray:
    values = np.asarray(factors, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise RuntimeError("VIF requires an n x k factor matrix with k >= 2.")
    n, k = values.shape
    out = np.empty(k, dtype=float)

    for target in range(k):
        y = values[:, target]
        others = np.delete(values, target, axis=1)
        design = np.column_stack([np.ones(n), others])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        residuals = y - design @ beta
        ss_res = float(residuals @ residuals)
        centered = y - float(np.mean(y))
        ss_tot = float(centered @ centered)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
        out[target] = 1.0 / max(1.0 - r2, 1e-15)

    return out


@dataclass(frozen=True)
class GrsResult:
    statistic: float
    p_value: float
    df1: int
    df2: int


def grs_joint_alpha_test(
    alphas: Sequence[float],
    residual_matrix: np.ndarray,
    factor_matrix: np.ndarray,
) -> GrsResult:
    alpha = np.asarray(alphas, dtype=float).ravel()
    residuals = np.asarray(residual_matrix, dtype=float)
    factors = np.asarray(factor_matrix, dtype=float)

    if residuals.ndim != 2 or factors.ndim != 2:
        raise RuntimeError("GRS requires residual and factor matrices.")
    if residuals.shape[0] != factors.shape[0] or residuals.shape[1] != alpha.size:
        raise RuntimeError("GRS inputs are misaligned.")

    t_obs = residuals.shape[0]
    n_assets = residuals.shape[1]
    n_factors = factors.shape[1]
    df2 = t_obs - n_assets - n_factors
    if df2 <= 0:
        raise RuntimeError("GRS requires T > N + K.")

    sigma = (residuals.T @ residuals) / float(t_obs)
    mu_f = np.mean(factors, axis=0)
    demeaned = factors - mu_f
    omega = (demeaned.T @ demeaned) / float(t_obs)

    sigma_inv = np.linalg.inv(sigma)
    omega_inv = np.linalg.inv(omega)

    numerator = float(alpha @ sigma_inv @ alpha)
    denominator = 1.0 + float(mu_f @ omega_inv @ mu_f)
    statistic = ((t_obs - n_assets - n_factors) / n_assets) * numerator / denominator
    p_value = float(stats.f.sf(statistic, n_assets, df2))
    return GrsResult(
        statistic=float(statistic),
        p_value=p_value,
        df1=int(n_assets),
        df2=int(df2),
    )


def rolling_ols_coefficient(
    design: np.ndarray,
    response: Sequence[float],
    *,
    window: int,
    coefficient_index: int,
) -> np.ndarray:
    x = np.asarray(design, dtype=float)
    y = np.asarray(response, dtype=float).ravel()
    if x.ndim != 2 or x.shape[0] != y.size:
        raise RuntimeError("Rolling OLS inputs are misaligned.")
    if window <= x.shape[1] or window > x.shape[0]:
        raise RuntimeError("Rolling OLS window is invalid.")
    if not 0 <= coefficient_index < x.shape[1]:
        raise RuntimeError("Rolling OLS coefficient index is invalid.")

    out = np.empty(x.shape[0] - window + 1, dtype=float)
    for start in range(out.size):
        end = start + window
        beta = np.linalg.lstsq(x[start:end], y[start:end], rcond=None)[0]
        out[start] = float(beta[coefficient_index])
    return out
