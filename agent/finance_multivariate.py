from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2


@dataclass(frozen=True)
class DccFit:
    alpha: float
    beta: float
    persistence: float
    log_likelihood: float
    correlations: np.ndarray


def standardized_residuals(
    returns: np.ndarray,
    conditional_variances: np.ndarray,
) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    variances = np.asarray(conditional_variances, dtype=float)
    if values.shape != variances.shape or values.ndim != 2:
        raise RuntimeError("Returns and conditional variances must be T x N matrices.")
    std = np.sqrt(np.maximum(variances, 1e-18))
    z = values / std
    if not np.all(np.isfinite(z)):
        raise RuntimeError("Standardized residuals contain non-finite values.")
    return z


def _dcc_qbar(z: np.ndarray) -> np.ndarray:
    values = np.asarray(z, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise RuntimeError("DCC requires at least two residual series.")
    # Unconditional second-moment matrix of standardized residuals.
    # Under the zero-mean GARCH specification this is the standard DCC Q-bar.
    return (values.T @ values) / float(values.shape[0])


def dcc_correlation_path(
    standardized: np.ndarray,
    alpha: float,
    beta: float,
    *,
    qbar: np.ndarray | None = None,
) -> np.ndarray:
    z = np.asarray(standardized, dtype=float)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 2:
        raise RuntimeError("DCC requires a T x N residual matrix with T,N >= 2.")
    if alpha <= 0.0 or beta <= 0.0 or alpha + beta >= 1.0:
        raise RuntimeError("DCC parameters must be positive with alpha + beta < 1.")
    if not np.all(np.isfinite(z)):
        raise RuntimeError("DCC residual matrix contains non-finite values.")

    base = _dcc_qbar(z) if qbar is None else np.asarray(qbar, dtype=float)
    n = z.shape[1]
    if base.shape != (n, n):
        raise RuntimeError("DCC unconditional covariance has the wrong shape.")

    out = np.empty((z.shape[0], n, n), dtype=float)
    q = base.copy()
    for t in range(z.shape[0]):
        if t > 0:
            previous = z[t - 1]
            q = (
                (1.0 - alpha - beta) * base
                + alpha * np.outer(previous, previous)
                + beta * q
            )
        diagonal = np.sqrt(np.maximum(np.diag(q), 1e-18))
        r = q / np.outer(diagonal, diagonal)
        r = 0.5 * (r + r.T)
        np.fill_diagonal(r, 1.0)
        out[t] = r
    return out


def dcc_log_likelihood(
    standardized: np.ndarray,
    alpha: float,
    beta: float,
    *,
    qbar: np.ndarray | None = None,
) -> float:
    z = np.asarray(standardized, dtype=float)
    correlations = dcc_correlation_path(z, alpha, beta, qbar=qbar)
    total = 0.0
    for row, r in zip(z, correlations, strict=True):
        sign, logdet = np.linalg.slogdet(r)
        if sign <= 0.0 or not np.isfinite(logdet):
            return float("-inf")
        try:
            quadratic = float(row @ np.linalg.solve(r, row))
        except np.linalg.LinAlgError:
            return float("-inf")
        identity_quadratic = float(row @ row)
        total += -0.5 * (logdet + quadratic - identity_quadratic)
    return float(total)


def fit_dcc_qml(
    standardized: np.ndarray,
    *,
    start: tuple[float, float] = (0.05, 0.93),
) -> DccFit:
    z = np.asarray(standardized, dtype=float)
    qbar = _dcc_qbar(z)

    def objective(x: np.ndarray) -> float:
        alpha = float(x[0])
        beta = float(x[1])
        if alpha <= 0.0 or beta <= 0.0 or alpha + beta >= 0.999999:
            excess = max(alpha + beta - 0.999998, 0.0)
            return 1e10 + 1e8 * excess * excess
        ll = dcc_log_likelihood(z, alpha, beta, qbar=qbar)
        return -ll if np.isfinite(ll) else 1e12

    result = minimize(
        objective,
        np.asarray(start, dtype=float),
        method="L-BFGS-B",
        bounds=((1e-8, 0.999), (1e-8, 0.999)),
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"DCC QML optimization failed: {result.message}")
    alpha = float(result.x[0])
    beta = float(result.x[1])
    if alpha <= 0.0 or beta <= 0.0 or alpha + beta >= 1.0:
        raise RuntimeError("DCC QML returned a non-stationary parameter pair.")
    correlations = dcc_correlation_path(z, alpha, beta, qbar=qbar)
    return DccFit(
        alpha=alpha,
        beta=beta,
        persistence=alpha + beta,
        log_likelihood=float(-result.fun),
        correlations=correlations,
    )


def conditional_covariance_path(
    conditional_std: np.ndarray,
    correlations: np.ndarray,
) -> np.ndarray:
    sigma = np.asarray(conditional_std, dtype=float)
    r = np.asarray(correlations, dtype=float)
    if sigma.ndim != 2 or r.shape != (sigma.shape[0], sigma.shape[1], sigma.shape[1]):
        raise RuntimeError("Conditional standard deviations and correlations are misaligned.")
    return r * sigma[:, :, None] * sigma[:, None, :]


def portfolio_conditional_volatility(
    covariances: np.ndarray,
    weights: Sequence[float],
) -> np.ndarray:
    h = np.asarray(covariances, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    if h.ndim != 3 or h.shape[1:] != (w.size, w.size):
        raise RuntimeError("Portfolio covariance path and weights are misaligned.")
    variance = np.einsum("i,tij,j->t", w, h, w)
    return np.sqrt(np.maximum(variance, 0.0))


def stationary_bootstrap_rows(
    values: np.ndarray,
    *,
    n_samples: int,
    restart_probability: float,
    seed: int,
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0:
        raise RuntimeError("Stationary bootstrap requires a non-empty T x N matrix.")
    if n_samples <= 0 or not 0.0 < restart_probability <= 1.0:
        raise RuntimeError("Invalid stationary-bootstrap settings.")

    rng = np.random.default_rng(seed)
    indices = np.empty(n_samples, dtype=np.int64)

    # Circular Politis-Romano stationary bootstrap.
    #
    # Keep the RNG draw ordering stable: establish an initial position,
    # then start the first block explicitly. This matters for reproducible
    # seeded bootstrap output.
    pos = int(rng.integers(0, x.shape[0]))

    for i in range(n_samples):
        if i == 0 or float(rng.random()) < restart_probability:
            pos = int(rng.integers(0, x.shape[0]))
        else:
            pos = (pos + 1) % x.shape[0]
        indices[i] = pos

    return x[indices]

def filtered_historical_var_es(
    bootstrap_standardized: np.ndarray,
    conditional_std: np.ndarray,
    weights: Sequence[float],
    *,
    tail_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(bootstrap_standardized, dtype=float)
    sigma = np.asarray(conditional_std, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    if z.ndim != 2 or sigma.ndim != 2 or z.shape[1] != sigma.shape[1] or w.size != z.shape[1]:
        raise RuntimeError("FHS residuals, volatilities and weights are misaligned.")
    if not 0.0 < tail_probability < 0.5:
        raise RuntimeError("FHS tail probability must lie in (0, 0.5).")

    var = np.empty(sigma.shape[0], dtype=float)
    es = np.empty(sigma.shape[0], dtype=float)
    percentile = 100.0 * tail_probability
    for t in range(sigma.shape[0]):
        simulated = (z * sigma[t]) @ w
        threshold = float(np.percentile(simulated, percentile))
        losses = -simulated
        var[t] = max(-threshold, 0.0)
        tail = losses[simulated <= threshold]
        es[t] = max(float(np.mean(tail)) if tail.size else var[t], var[t])
    return var, es


@dataclass(frozen=True)
class CoverageBacktest:
    num_observations: int
    num_violations: int
    violation_rate: float
    expected_rate: float
    kupiec_lr_stat: float
    kupiec_p_value: float
    christoffersen_lr_ind: float
    christoffersen_lr_cc: float
    christoffersen_p_value: float


def _binomial_log_likelihood(successes: int, trials: int, probability: float) -> float:
    if trials < 0 or successes < 0 or successes > trials:
        raise RuntimeError("Invalid binomial counts.")
    failures = trials - successes
    total = 0.0
    if successes:
        if probability <= 0.0:
            return float("-inf")
        total += successes * math.log(probability)
    if failures:
        if probability >= 1.0:
            return float("-inf")
        total += failures * math.log1p(-probability)
    return float(total)


def coverage_backtest(
    violations: Sequence[int],
    *,
    expected_rate: float,
) -> CoverageBacktest:
    v = np.asarray(violations, dtype=int).ravel()
    if v.size == 0 or not np.isin(v, [0, 1]).all():
        raise RuntimeError("Coverage test requires a non-empty binary violation series.")
    if not 0.0 < expected_rate < 1.0:
        raise RuntimeError("Expected violation rate must lie in (0, 1).")

    n = int(v.size)
    x = int(v.sum())
    observed = x / n
    ll_null = _binomial_log_likelihood(x, n, expected_rate)
    ll_alt = _binomial_log_likelihood(x, n, observed)
    lr_uc = max(-2.0 * (ll_null - ll_alt), 0.0)
    kupiec_p = float(chi2.sf(lr_uc, 1))

    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))

    row0 = n00 + n01
    row1 = n10 + n11
    total_transitions = row0 + row1
    pi = (n01 + n11) / total_transitions if total_transitions else 0.0
    pi01 = n01 / row0 if row0 else 0.0
    pi11 = n11 / row1 if row1 else 0.0

    ll_independent = (
        _binomial_log_likelihood(n01 + n11, total_transitions, pi)
        if total_transitions
        else 0.0
    )
    ll_markov = (
        (_binomial_log_likelihood(n01, row0, pi01) if row0 else 0.0)
        + (_binomial_log_likelihood(n11, row1, pi11) if row1 else 0.0)
    )
    lr_ind = max(-2.0 * (ll_independent - ll_markov), 0.0)
    lr_cc = lr_uc + lr_ind
    cc_p = float(chi2.sf(lr_cc, 2))
    return CoverageBacktest(
        num_observations=n,
        num_violations=x,
        violation_rate=float(observed),
        expected_rate=float(expected_rate),
        kupiec_lr_stat=float(lr_uc),
        kupiec_p_value=kupiec_p,
        christoffersen_lr_ind=float(lr_ind),
        christoffersen_lr_cc=float(lr_cc),
        christoffersen_p_value=cc_p,
    )
