from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import gammaln


def gaussian_copula_loglik(
    u1: Sequence[float],
    u2: Sequence[float],
    rho: float,
) -> float:
    x = np.asarray(u1, dtype=float)
    y = np.asarray(u2, dtype=float)

    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError(
            "u1 and u2 must be equal-length one-dimensional arrays."
        )

    correlation = float(rho)

    if not -0.9999 < correlation < 0.9999:
        raise ValueError(
            "rho must lie strictly between -0.9999 and 0.9999."
        )

    eps = 1e-10

    x = np.clip(x, eps, 1.0 - eps)
    y = np.clip(y, eps, 1.0 - eps)

    z1 = stats.norm.ppf(x)
    z2 = stats.norm.ppf(y)

    rho2 = correlation * correlation

    log_density = (
        -0.5 * np.log(1.0 - rho2)
        - (
            rho2 * (z1 * z1 + z2 * z2)
            - 2.0 * correlation * z1 * z2
        )
        / (
            2.0 * (1.0 - rho2)
        )
    )

    return float(
        np.sum(log_density)
    )


def student_t_copula_loglik(
    u1: Sequence[float],
    u2: Sequence[float],
    rho: float,
    degrees_of_freedom: float,
) -> float:
    x = np.asarray(u1, dtype=float)
    y = np.asarray(u2, dtype=float)

    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError(
            "u1 and u2 must be equal-length one-dimensional arrays."
        )

    correlation = float(rho)
    nu = float(degrees_of_freedom)

    if not -0.9999 < correlation < 0.9999:
        raise ValueError(
            "rho must lie strictly between -0.9999 and 0.9999."
        )

    if nu <= 2.0:
        raise ValueError(
            "degrees_of_freedom must exceed 2."
        )

    eps = 1e-10

    x = np.clip(x, eps, 1.0 - eps)
    y = np.clip(y, eps, 1.0 - eps)

    z1 = stats.t.ppf(x, nu)
    z2 = stats.t.ppf(y, nu)

    rho2 = correlation * correlation
    n = x.size

    loglik = n * (
        gammaln(
            (nu + 2.0) / 2.0
        )
        - gammaln(
            nu / 2.0
        )
        - math.log(
            nu * math.pi
        )
        - 0.5
        * math.log(
            1.0 - rho2
        )
    )

    quadratic = (
        z1 * z1
        + z2 * z2
        - 2.0
        * correlation
        * z1
        * z2
    ) / (
        nu
        * (
            1.0 - rho2
        )
    )

    loglik += np.sum(
        -(
            nu + 2.0
        )
        / 2.0
        * np.log1p(
            quadratic
        )
    )

    loglik -= np.sum(
        stats.t.logpdf(
            z1,
            nu,
        )
    )

    loglik -= np.sum(
        stats.t.logpdf(
            z2,
            nu,
        )
    )

    return float(loglik)


def clayton_copula_loglik(
    u1: Sequence[float],
    u2: Sequence[float],
    theta: float,
) -> float:
    x = np.asarray(u1, dtype=float)
    y = np.asarray(u2, dtype=float)
    parameter = float(theta)

    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError(
            "u1 and u2 must be equal-length one-dimensional arrays."
        )

    if parameter <= 0.0:
        raise ValueError(
            "theta must be positive."
        )

    eps = 1e-10

    x = np.clip(x, eps, 1.0 - eps)
    y = np.clip(y, eps, 1.0 - eps)

    total = (
        x ** (-parameter)
        + y ** (-parameter)
        - 1.0
    )

    if np.any(total <= 0.0):
        raise ValueError(
            "invalid Clayton copula state."
        )

    log_density = (
        math.log(
            1.0 + parameter
        )
        + (
            -1.0 - parameter
        )
        * (
            np.log(x)
            + np.log(y)
        )
        + (
            -1.0 / parameter
            - 2.0
        )
        * np.log(total)
    )

    return float(
        np.sum(log_density)
    )


def gumbel_copula_loglik(
    u1: Sequence[float],
    u2: Sequence[float],
    theta: float,
) -> float:
    x = np.asarray(u1, dtype=float)
    y = np.asarray(u2, dtype=float)
    parameter = float(theta)

    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError(
            "u1 and u2 must be equal-length one-dimensional arrays."
        )

    if parameter < 1.0:
        raise ValueError(
            "theta must be at least 1."
        )

    eps = 1e-10

    x = np.clip(x, eps, 1.0 - eps)
    y = np.clip(y, eps, 1.0 - eps)

    lx = -np.log(x)
    ly = -np.log(y)

    tx = lx**parameter
    ty = ly**parameter

    total = tx + ty
    generator = total ** (
        1.0 / parameter
    )

    log_density = (
        -generator
        - np.log(x)
        - np.log(y)
        + (
            parameter - 1.0
        )
        * (
            np.log(lx)
            + np.log(ly)
        )
        - (
            2.0
            - 1.0 / parameter
        )
        * np.log(total)
        + np.log(
            generator
            + parameter
            - 1.0
        )
    )

    return float(
        np.sum(log_density)
    )


def kendall_tau_to_gaussian_rho(
    kendall_tau: float,
) -> float:
    tau = float(kendall_tau)

    if not -1.0 <= tau <= 1.0:
        raise ValueError(
            "kendall_tau must lie in [-1, 1]."
        )

    return float(
        math.sin(
            math.pi
            * tau
            / 2.0
        )
    )


def fit_student_t_degrees_of_freedom(
    u1: Sequence[float],
    u2: Sequence[float],
    *,
    rho: float,
    lower: float = 2.1,
    upper: float = 100.0,
) -> float:
    if lower <= 2.0 or upper <= lower:
        raise ValueError(
            "require 2 < lower < upper."
        )

    result = minimize_scalar(
        lambda nu: -student_t_copula_loglik(
            u1,
            u2,
            rho,
            nu,
        ),
        bounds=(
            float(lower),
            float(upper),
        ),
        method="bounded",
    )

    if not result.success:
        raise RuntimeError(
            "Student-t degrees-of-freedom fit failed."
        )

    return float(result.x)


def empirical_tail_dependence(
    u1: Sequence[float],
    u2: Sequence[float],
    *,
    quantile: float = 0.95,
) -> tuple[float, float]:
    x = np.asarray(u1, dtype=float)
    y = np.asarray(u2, dtype=float)

    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError(
            "u1 and u2 must be equal-length one-dimensional arrays."
        )

    q = float(quantile)

    if not 0.5 < q < 1.0:
        raise ValueError(
            "quantile must lie strictly between 0.5 and 1."
        )

    upper_x = x > q
    upper_y = y > q

    lower_threshold = (
        1.0 - q
    )

    lower_x = x < lower_threshold
    lower_y = y < lower_threshold

    upper_denominator = int(
        np.sum(upper_x)
    )

    lower_denominator = int(
        np.sum(lower_x)
    )

    lambda_upper = (
        0.0
        if upper_denominator == 0
        else float(
            np.sum(
                upper_x
                & upper_y
            )
            / upper_denominator
        )
    )

    lambda_lower = (
        0.0
        if lower_denominator == 0
        else float(
            np.sum(
                lower_x
                & lower_y
            )
            / lower_denominator
        )
    )

    return (
        lambda_lower,
        lambda_upper,
    )


__all__ = (
    "gaussian_copula_loglik",
    "student_t_copula_loglik",
    "clayton_copula_loglik",
    "gumbel_copula_loglik",
    "kendall_tau_to_gaussian_rho",
    "fit_student_t_degrees_of_freedom",
    "empirical_tail_dependence",
)
