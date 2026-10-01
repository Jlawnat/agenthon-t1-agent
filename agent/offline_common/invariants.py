from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def finite_numeric_mask(
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return True where numeric values are finite."""
    array = np.asarray(values, dtype=float)
    return np.isfinite(array)


def probability_bounds_mask(
    values: Sequence[float] | np.ndarray,
    *,
    upper: float = 1.0,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Return True where probabilities satisfy configured bounds."""
    array = np.asarray(values, dtype=float)

    if upper <= 0.0:
        raise RuntimeError(
            "Probability upper bound must be positive."
        )
    if tolerance < 0.0:
        raise RuntimeError(
            "Probability tolerance must be nonnegative."
        )

    return (
        np.isfinite(array)
        & (array >= -tolerance)
        & (array <= upper + tolerance)
    )


def is_positive_semidefinite(
    matrix: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> bool:
    """Check whether a finite symmetric matrix is PSD within tolerance."""
    array = np.asarray(matrix, dtype=float)

    if (
        array.ndim != 2
        or array.shape[0] != array.shape[1]
    ):
        raise RuntimeError(
            "PSD validation requires a square matrix."
        )

    if not np.all(np.isfinite(array)):
        return False

    if not np.allclose(
        array,
        array.T,
        rtol=0.0,
        atol=tolerance,
    ):
        return False

    symmetric = 0.5 * (
        array + array.T
    )

    minimum_eigenvalue = float(
        np.min(
            np.linalg.eigvalsh(
                symmetric
            )
        )
    )

    return minimum_eigenvalue >= -tolerance


def weight_sum_error(
    weights: Sequence[float],
    *,
    target: float = 1.0,
) -> float:
    """Return signed total-weight reconciliation error."""
    array = np.asarray(
        weights,
        dtype=float,
    ).ravel()

    if not np.all(np.isfinite(array)):
        raise RuntimeError(
            "Weights contain non-finite values."
        )

    return float(
        array.sum() - float(target)
    )


def causal_order_mask(
    earlier: Sequence[object],
    later: Sequence[object],
    *,
    strict: bool = False,
) -> np.ndarray:
    """Validate row-wise causal ordering of timestamp sequences."""
    left = pd.to_datetime(
        pd.Series(earlier),
        errors="coerce",
        utc=True,
    )
    right = pd.to_datetime(
        pd.Series(later),
        errors="coerce",
        utc=True,
    )

    if len(left) != len(right):
        raise RuntimeError(
            "Causal timestamp inputs must be aligned."
        )

    valid = (
        left.notna()
        & right.notna()
    )

    if strict:
        ordered = left < right
    else:
        ordered = left <= right

    return (
        valid
        & ordered
    ).to_numpy(dtype=bool)


def reconciliation_close(
    actual: Sequence[float],
    components: np.ndarray,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> np.ndarray:
    """Check actual values against the row-wise sum of components."""
    actual_values = np.asarray(
        actual,
        dtype=float,
    ).ravel()

    component_values = np.asarray(
        components,
        dtype=float,
    )

    if component_values.ndim == 1:
        component_values = (
            component_values.reshape(-1, 1)
        )

    if (
        component_values.ndim != 2
        or component_values.shape[0]
        != actual_values.size
    ):
        raise RuntimeError(
            "Reconciliation inputs are misaligned."
        )

    expected = component_values.sum(
        axis=1
    )

    finite = (
        np.isfinite(actual_values)
        & np.isfinite(component_values).all(
            axis=1
        )
    )

    close = np.isclose(
        actual_values,
        expected,
        atol=atol,
        rtol=rtol,
    )

    return finite & close
