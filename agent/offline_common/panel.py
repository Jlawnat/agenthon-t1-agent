from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def zscore_ddof0(values: Sequence[float]) -> np.ndarray:
    """Population-standardize a cross-section using ddof=0."""
    array = np.asarray(values, dtype=float).ravel()

    if array.size == 0:
        return np.asarray([], dtype=float)

    mean = float(array.mean())
    std = float(array.std(ddof=0))

    if std == 0.0 or not np.isfinite(std):
        return np.zeros_like(array, dtype=float)

    return np.asarray(
        (array - mean) / std,
        dtype=float,
    )


def equal_count_bucket_labels(
    n_observations: int,
    *,
    buckets: int,
) -> np.ndarray:
    """Assign deterministic 1-based equal-count labels to ranked rows."""
    n = int(n_observations)
    q = int(buckets)

    if n <= 0:
        raise RuntimeError(
            "Bucket assignment requires at least one observation."
        )
    if q <= 0:
        raise RuntimeError(
            "Bucket assignment requires a positive bucket count."
        )

    ranks = np.arange(n, dtype=np.int64)

    return np.minimum(
        q,
        (q * ranks) // n + 1,
    )


def spearman_rank_correlation(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """Spearman correlation using average ranks for ties."""
    first_values = np.asarray(first)
    second_values = np.asarray(second)

    if (
        first_values.ndim != 1
        or second_values.ndim != 1
        or first_values.size != second_values.size
    ):
        raise RuntimeError(
            "Rank-correlation inputs must be aligned one-dimensional arrays."
        )

    rank_first = (
        pd.Series(first_values)
        .rank(method="average")
        .to_numpy(dtype=float)
    )
    rank_second = (
        pd.Series(second_values)
        .rank(method="average")
        .to_numpy(dtype=float)
    )

    if (
        np.std(rank_first) == 0.0
        or np.std(rank_second) == 0.0
    ):
        return float("nan")

    value = float(
        np.corrcoef(
            rank_first,
            rank_second,
        )[0, 1]
    )

    return (
        value
        if np.isfinite(value)
        else float("nan")
    )


def residualize_against_controls(
    values: Sequence[float],
    controls: np.ndarray,
    *,
    add_intercept: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Project values on controls and return residuals and coefficients."""
    y = np.asarray(values, dtype=float).ravel()
    x = np.asarray(controls, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if x.ndim != 2 or x.shape[0] != y.size:
        raise RuntimeError(
            "Residualization requires aligned observations and controls."
        )

    if add_intercept:
        x = np.column_stack(
            [
                np.ones(y.size, dtype=float),
                x,
            ]
        )

    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
    ):
        raise RuntimeError(
            "Residualization inputs contain non-finite values."
        )

    coefficients = (
        np.linalg.pinv(x.T @ x)
        @ (x.T @ y)
    )

    residuals = y - x @ coefficients

    return (
        np.asarray(residuals, dtype=float),
        np.asarray(coefficients, dtype=float),
    )


def beta_neutral_projection(
    weights: Sequence[float],
    beta: Sequence[float],
) -> np.ndarray:
    """Project weights away from intercept and one beta exposure."""
    w = np.asarray(weights, dtype=float).ravel()
    b = np.asarray(beta, dtype=float).ravel()

    if w.size != b.size:
        raise RuntimeError(
            "Weights and beta exposures must be aligned."
        )
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(b)):
        raise RuntimeError(
            "Neutral-projection inputs contain non-finite values."
        )

    design = np.column_stack(
        [
            np.ones(w.size, dtype=float),
            b,
        ]
    )

    projected = w - design @ (
        np.linalg.pinv(design.T @ design)
        @ (design.T @ w)
    )

    return np.asarray(
        projected,
        dtype=float,
    )
