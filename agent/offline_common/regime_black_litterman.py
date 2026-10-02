from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class GaussianHmm2StateResult:
    means: np.ndarray
    standard_deviations: np.ndarray
    initial_probabilities: np.ndarray
    transition_matrix: np.ndarray
    posterior_probabilities: np.ndarray
    iterations: int


@dataclass(frozen=True)
class RegimeCovarianceResult:
    bull_covariance: np.ndarray
    bear_covariance: np.ndarray
    bull_mask: np.ndarray
    bear_mask: np.ndarray


@dataclass(frozen=True)
class BlackLittermanResult:
    implied_returns: np.ndarray
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    weights: np.ndarray


def fit_two_state_gaussian_hmm(
    observations: Sequence[float],
    *,
    initial_probabilities: Sequence[float],
    transition_matrix: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> GaussianHmm2StateResult:
    """Fit a two-state Gaussian HMM with Baum-Welch EM."""
    values = np.asarray(
        observations,
        dtype=float,
    ).ravel()

    if (
        values.size < 4
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "observations must contain at least four finite values."
        )

    pi = np.asarray(
        initial_probabilities,
        dtype=float,
    ).ravel()

    transition = np.asarray(
        transition_matrix,
        dtype=float,
    ).copy()

    if pi.shape != (2,):
        raise ValueError(
            "initial_probabilities must have length two."
        )

    if transition.shape != (2, 2):
        raise ValueError(
            "transition_matrix must have shape (2, 2)."
        )

    if (
        np.any(pi < 0.0)
        or not np.isclose(
            pi.sum(),
            1.0,
        )
    ):
        raise ValueError(
            "initial probabilities must be non-negative and sum to one."
        )

    if (
        np.any(transition < 0.0)
        or not np.allclose(
            transition.sum(axis=1),
            1.0,
        )
    ):
        raise ValueError(
            "transition rows must be non-negative and sum to one."
        )

    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError(
            "max_iterations must be a positive integer."
        )

    threshold = float(tolerance)

    if threshold <= 0.0:
        raise ValueError(
            "tolerance must be positive."
        )

    midpoint = values.size // 2

    means = np.array(
        [
            np.mean(
                values[:midpoint]
            ),
            np.mean(
                values[midpoint:]
            ),
        ],
        dtype=float,
    )

    stds = np.array(
        [
            np.std(
                values[:midpoint],
                ddof=1,
            ),
            np.std(
                values[midpoint:],
                ddof=1,
            ),
        ],
        dtype=float,
    )

    stds = np.maximum(
        stds,
        1e-8,
    )

    posterior = None
    completed_iterations = 0

    for iteration in range(
        max_iterations
    ):
        log_emission = np.zeros(
            (
                values.size,
                2,
            ),
            dtype=float,
        )

        for state in range(2):
            log_emission[:, state] = (
                norm.logpdf(
                    values,
                    means[state],
                    stds[state],
                )
            )

        alpha = np.zeros_like(
            log_emission
        )

        alpha[0] = (
            np.log(
                pi + 1e-300
            )
            + log_emission[0]
        )

        for t in range(
            1,
            values.size,
        ):
            for destination in range(2):
                alpha[
                    t,
                    destination,
                ] = (
                    np.logaddexp(
                        alpha[
                            t - 1,
                            0,
                        ]
                        + math.log(
                            transition[
                                0,
                                destination,
                            ]
                            + 1e-300
                        ),
                        alpha[
                            t - 1,
                            1,
                        ]
                        + math.log(
                            transition[
                                1,
                                destination,
                            ]
                            + 1e-300
                        ),
                    )
                    + log_emission[
                        t,
                        destination,
                    ]
                )

        beta = np.zeros_like(
            log_emission
        )

        for t in range(
            values.size - 2,
            -1,
            -1,
        ):
            for origin in range(2):
                beta[
                    t,
                    origin,
                ] = np.logaddexp(
                    math.log(
                        transition[
                            origin,
                            0,
                        ]
                        + 1e-300
                    )
                    + log_emission[
                        t + 1,
                        0,
                    ]
                    + beta[
                        t + 1,
                        0,
                    ],
                    math.log(
                        transition[
                            origin,
                            1,
                        ]
                        + 1e-300
                    )
                    + log_emission[
                        t + 1,
                        1,
                    ]
                    + beta[
                        t + 1,
                        1,
                    ],
                )

        log_posterior = (
            alpha + beta
        )

        normalizer = np.logaddexp(
            log_posterior[:, 0],
            log_posterior[:, 1],
        )

        log_posterior -= (
            normalizer[:, None]
        )

        posterior = np.exp(
            log_posterior
        )

        old_means = means.copy()

        for state in range(2):
            weights = posterior[
                :,
                state,
            ]

            denominator = float(
                weights.sum()
            )

            if denominator <= 0.0:
                raise RuntimeError(
                    "HMM state received zero posterior mass."
                )

            means[state] = float(
                np.sum(
                    weights
                    * values
                )
                / denominator
            )

            variance = float(
                np.sum(
                    weights
                    * (
                        values
                        - means[state]
                    )
                    ** 2
                )
                / denominator
            )

            stds[state] = math.sqrt(
                max(
                    variance,
                    1e-16,
                )
            )

        pi = posterior[0].copy()

        new_transition = np.zeros(
            (2, 2),
            dtype=float,
        )

        log_probability = np.logaddexp(
            alpha[-1, 0],
            alpha[-1, 1],
        )

        for origin in range(2):
            for destination in range(2):
                total = 0.0

                for t in range(
                    values.size - 1
                ):
                    total += math.exp(
                        alpha[
                            t,
                            origin,
                        ]
                        + math.log(
                            transition[
                                origin,
                                destination,
                            ]
                            + 1e-300
                        )
                        + log_emission[
                            t + 1,
                            destination,
                        ]
                        + beta[
                            t + 1,
                            destination,
                        ]
                        - log_probability
                    )

                new_transition[
                    origin,
                    destination,
                ] = total

            row_total = float(
                new_transition[
                    origin
                ].sum()
            )

            if row_total <= 0.0:
                raise RuntimeError(
                    "HMM transition row received zero mass."
                )

            new_transition[
                origin
            ] /= row_total

        transition = new_transition
        completed_iterations = (
            iteration + 1
        )

        if (
            np.max(
                np.abs(
                    means
                    - old_means
                )
            )
            < threshold
        ):
            break

    if posterior is None:
        raise RuntimeError(
            "HMM failed to produce posterior probabilities."
        )

    return GaussianHmm2StateResult(
        means=means.copy(),
        standard_deviations=stds.copy(),
        initial_probabilities=pi.copy(),
        transition_matrix=transition.copy(),
        posterior_probabilities=posterior.copy(),
        iterations=completed_iterations,
    )


def hard_regime_covariances(
    returns: np.ndarray,
    posterior_probabilities: np.ndarray,
    *,
    bull_state: int,
    threshold: float = 0.5,
    minimum_observations: int = 11,
) -> RegimeCovarianceResult:
    """Compute hard-assignment bull/bear sample covariance matrices."""
    matrix = np.asarray(
        returns,
        dtype=float,
    )

    posterior = np.asarray(
        posterior_probabilities,
        dtype=float,
    )

    if (
        matrix.ndim != 2
        or not np.all(
            np.isfinite(matrix)
        )
    ):
        raise ValueError(
            "returns must be a finite two-dimensional matrix."
        )

    if posterior.shape != (
        matrix.shape[0],
        2,
    ):
        raise ValueError(
            "posterior_probabilities must have shape (n_observations, 2)."
        )

    state = int(bull_state)

    if state not in {
        0,
        1,
    }:
        raise ValueError(
            "bull_state must be zero or one."
        )

    cutoff = float(threshold)

    if not 0.0 < cutoff < 1.0:
        raise ValueError(
            "threshold must lie strictly between zero and one."
        )

    if (
        isinstance(
            minimum_observations,
            bool,
        )
        or not isinstance(
            minimum_observations,
            int,
        )
        or minimum_observations < 2
    ):
        raise ValueError(
            "minimum_observations must be at least two."
        )

    bull_mask = (
        posterior[:, state]
        > cutoff
    )

    bear_mask = ~bull_mask

    unconditional = np.cov(
        matrix.T,
        ddof=1,
    )

    bull_covariance = (
        np.cov(
            matrix[
                bull_mask
            ].T,
            ddof=1,
        )
        if int(
            bull_mask.sum()
        )
        >= minimum_observations
        else unconditional.copy()
    )

    bear_covariance = (
        np.cov(
            matrix[
                bear_mask
            ].T,
            ddof=1,
        )
        if int(
            bear_mask.sum()
        )
        >= minimum_observations
        else unconditional.copy()
    )

    return RegimeCovarianceResult(
        bull_covariance=np.asarray(
            bull_covariance,
            dtype=float,
        ),
        bear_covariance=np.asarray(
            bear_covariance,
            dtype=float,
        ),
        bull_mask=bull_mask.copy(),
        bear_mask=bear_mask.copy(),
    )


def black_litterman_posterior(
    covariance: np.ndarray,
    market_weights: Sequence[float],
    *,
    risk_aversion: float,
    tau: float,
    view_matrix: np.ndarray,
    view_returns: Sequence[float],
    view_covariance: np.ndarray,
    normalize_gross: bool = True,
) -> BlackLittermanResult:
    """Black-Litterman posterior mean and implied optimal weights."""
    sigma = np.asarray(
        covariance,
        dtype=float,
    )

    weights = np.asarray(
        market_weights,
        dtype=float,
    ).ravel()

    p_matrix = np.asarray(
        view_matrix,
        dtype=float,
    )

    q_vector = np.asarray(
        view_returns,
        dtype=float,
    ).ravel()

    omega = np.asarray(
        view_covariance,
        dtype=float,
    )

    if (
        sigma.ndim != 2
        or sigma.shape[0]
        != sigma.shape[1]
    ):
        raise ValueError(
            "covariance must be square."
        )

    n_assets = sigma.shape[0]

    if weights.shape != (
        n_assets,
    ):
        raise ValueError(
            "market_weights length must match covariance dimension."
        )

    if (
        p_matrix.ndim != 2
        or p_matrix.shape[1]
        != n_assets
    ):
        raise ValueError(
            "view_matrix has incompatible dimensions."
        )

    n_views = p_matrix.shape[0]

    if q_vector.shape != (
        n_views,
    ):
        raise ValueError(
            "view_returns length must match number of views."
        )

    if omega.shape != (
        n_views,
        n_views,
    ):
        raise ValueError(
            "view_covariance has incompatible dimensions."
        )

    delta = float(
        risk_aversion
    )

    scaling = float(tau)

    if delta <= 0.0:
        raise ValueError(
            "risk_aversion must be positive."
        )

    if scaling <= 0.0:
        raise ValueError(
            "tau must be positive."
        )

    implied_returns = (
        delta
        * sigma
        @ weights
    )

    tau_sigma = (
        scaling
        * sigma
    )

    inverse_tau_sigma = (
        np.linalg.inv(
            tau_sigma
        )
    )

    inverse_omega = (
        np.linalg.inv(
            omega
        )
    )

    posterior_covariance = (
        np.linalg.inv(
            inverse_tau_sigma
            + p_matrix.T
            @ inverse_omega
            @ p_matrix
        )
    )

    posterior_mean = (
        posterior_covariance
        @ (
            inverse_tau_sigma
            @ implied_returns
            + p_matrix.T
            @ inverse_omega
            @ q_vector
        )
    )

    portfolio_weights = (
        np.linalg.solve(
            delta
            * sigma,
            posterior_mean,
        )
    )

    if normalize_gross:
        gross = float(
            np.sum(
                np.abs(
                    portfolio_weights
                )
            )
        )

        if gross <= 0.0:
            raise ValueError(
                "portfolio gross exposure is zero."
            )

        portfolio_weights = (
            portfolio_weights
            / gross
        )

    return BlackLittermanResult(
        implied_returns=np.asarray(
            implied_returns,
            dtype=float,
        ),
        posterior_mean=np.asarray(
            posterior_mean,
            dtype=float,
        ),
        posterior_covariance=np.asarray(
            posterior_covariance,
            dtype=float,
        ),
        weights=np.asarray(
            portfolio_weights,
            dtype=float,
        ),
    )


__all__ = (
    "GaussianHmm2StateResult",
    "RegimeCovarianceResult",
    "BlackLittermanResult",
    "fit_two_state_gaussian_hmm",
    "hard_regime_covariances",
    "black_litterman_posterior",
)
