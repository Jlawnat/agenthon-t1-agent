from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FactorNeutralPortfolio:
    target_exposures: np.ndarray
    hedge_weights: np.ndarray
    neutral_weights: np.ndarray
    neutral_exposures: np.ndarray


def factor_exposures(
    components: np.ndarray,
    weights: Sequence[float],
) -> np.ndarray:
    """Compute portfolio exposure to row-wise factor loadings."""
    loadings = np.asarray(
        components,
        dtype=float,
    )

    portfolio = np.asarray(
        weights,
        dtype=float,
    ).ravel()

    if loadings.ndim != 2:
        raise ValueError(
            "components must be a two-dimensional matrix."
        )

    if portfolio.shape != (
        loadings.shape[1],
    ):
        raise ValueError(
            "weights length must match number of assets."
        )

    if (
        not np.all(np.isfinite(loadings))
        or not np.all(np.isfinite(portfolio))
    ):
        raise ValueError(
            "components and weights must be finite."
        )

    return np.asarray(
        loadings @ portfolio,
        dtype=float,
    )


def minimum_norm_factor_neutral_hedge(
    components: np.ndarray,
    target_weights: Sequence[float],
    *,
    dollar_neutral_hedge: bool = True,
) -> FactorNeutralPortfolio:
    """Construct the minimum-norm hedge that neutralizes factor exposures."""
    loadings = np.asarray(
        components,
        dtype=float,
    )

    target = np.asarray(
        target_weights,
        dtype=float,
    ).ravel()

    if loadings.ndim != 2:
        raise ValueError(
            "components must be a two-dimensional matrix."
        )

    if target.shape != (
        loadings.shape[1],
    ):
        raise ValueError(
            "target_weights length must match number of assets."
        )

    if (
        not np.all(np.isfinite(loadings))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError(
            "components and target_weights must be finite."
        )

    target_exposure = (
        loadings @ target
    )

    if dollar_neutral_hedge:
        constraint_matrix = np.vstack(
            [
                loadings,
                np.ones(
                    (
                        1,
                        loadings.shape[1],
                    ),
                    dtype=float,
                ),
            ]
        )

        target_vector = np.append(
            -target_exposure,
            0.0,
        )
    else:
        constraint_matrix = loadings

        target_vector = (
            -target_exposure
        )

    hedge, *_ = np.linalg.lstsq(
        constraint_matrix,
        target_vector,
        rcond=None,
    )

    neutral = (
        target + hedge
    )

    neutral_exposure = (
        loadings @ neutral
    )

    return FactorNeutralPortfolio(
        target_exposures=np.asarray(
            target_exposure,
            dtype=float,
        ),
        hedge_weights=np.asarray(
            hedge,
            dtype=float,
        ),
        neutral_weights=np.asarray(
            neutral,
            dtype=float,
        ),
        neutral_exposures=np.asarray(
            neutral_exposure,
            dtype=float,
        ),
    )


def annualized_sharpe_ratio(
    returns: Sequence[float],
    *,
    periods_per_year: float,
) -> float:
    """Annualized zero-risk-free Sharpe ratio using sample volatility."""
    values = np.asarray(
        returns,
        dtype=float,
    ).ravel()

    annualization = float(
        periods_per_year
    )

    if (
        values.size < 2
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "returns must contain at least two finite observations."
        )

    if annualization <= 0.0:
        raise ValueError(
            "periods_per_year must be positive."
        )

    volatility = float(
        np.std(
            values,
            ddof=1,
        )
    )

    if volatility == 0.0:
        return 0.0

    return float(
        np.mean(values)
        / volatility
        * math.sqrt(
            annualization
        )
    )


def residual_variance_r_squared(
    residual_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    *,
    ddof: int = 0,
) -> float:
    """Variance-reduction R²: 1 - Var(residual) / Var(benchmark)."""
    residual = np.asarray(
        residual_returns,
        dtype=float,
    ).ravel()

    benchmark = np.asarray(
        benchmark_returns,
        dtype=float,
    ).ravel()

    if (
        residual.shape != benchmark.shape
        or residual.size < 2
    ):
        raise ValueError(
            "return series must have matching lengths of at least two."
        )

    if (
        not np.all(np.isfinite(residual))
        or not np.all(np.isfinite(benchmark))
    ):
        raise ValueError(
            "return series must be finite."
        )

    benchmark_variance = float(
        np.var(
            benchmark,
            ddof=ddof,
        )
    )

    if benchmark_variance <= 0.0:
        raise ValueError(
            "benchmark variance must be positive."
        )

    residual_variance = float(
        np.var(
            residual,
            ddof=ddof,
        )
    )

    return float(
        1.0
        - residual_variance
        / benchmark_variance
    )


__all__ = (
    "FactorNeutralPortfolio",
    "factor_exposures",
    "minimum_norm_factor_neutral_hedge",
    "annualized_sharpe_ratio",
    "residual_variance_r_squared",
)
