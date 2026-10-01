from __future__ import annotations

import numpy as np


def _as_finite_array(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=float,
    )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return array


def _history_matrix(
    history: np.ndarray,
) -> np.ndarray:
    array = _as_finite_array(
        history,
        name="history",
    )

    if array.ndim != 2:
        raise ValueError(
            "history must be two-dimensional."
        )

    if (
        array.shape[0] == 0
        or array.shape[1] == 0
    ):
        raise ValueError(
            "history must not be empty."
        )

    return array


def normalize_profile(
    values: np.ndarray,
) -> np.ndarray:
    """Clip negatives to zero and normalize values to sum to one."""
    array = _as_finite_array(
        values,
        name="values",
    )

    if array.ndim != 1:
        raise ValueError(
            "values must be one-dimensional."
        )

    clipped = np.maximum(
        array,
        0.0,
    )

    total = float(
        np.sum(
            clipped
        )
    )

    if total <= 0.0:
        raise ValueError(
            "profile must have positive total weight."
        )

    return (
        clipped
        / total
    )


def mean_volume_profile(
    history: np.ndarray,
) -> np.ndarray:
    """Normalized mean profile across historical observations."""
    array = _history_matrix(
        history
    )

    return normalize_profile(
        np.mean(
            array,
            axis=0,
        )
    )


def median_volume_profile(
    history: np.ndarray,
) -> np.ndarray:
    """Normalized median profile across historical observations."""
    array = _history_matrix(
        history
    )

    return normalize_profile(
        np.median(
            array,
            axis=0,
        )
    )


def exponentially_weighted_volume_profile(
    history: np.ndarray,
    *,
    half_life: float,
) -> np.ndarray:
    """Normalized exponentially weighted profile, oldest to newest."""
    array = _history_matrix(
        history
    )

    half_life_value = float(
        half_life
    )

    if (
        not np.isfinite(
            half_life_value
        )
        or half_life_value <= 0.0
    ):
        raise ValueError(
            "half_life must be positive and finite."
        )

    lags = np.arange(
        len(array) - 1,
        -1,
        -1,
        dtype=float,
    )

    weights = np.exp(
        -np.log(2.0)
        * lags
        / half_life_value
    )

    return normalize_profile(
        np.average(
            array,
            axis=0,
            weights=weights,
        )
    )


def winsorized_mean_volume_profile(
    history: np.ndarray,
    *,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> np.ndarray:
    """Normalize a column-wise winsorized historical mean profile."""
    array = _history_matrix(
        history
    )

    lower = float(
        lower_percentile
    )
    upper = float(
        upper_percentile
    )

    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower < 0.0
        or upper > 100.0
        or lower > upper
    ):
        raise ValueError(
            "percentiles must satisfy "
            "0 <= lower_percentile <= upper_percentile <= 100."
        )

    low = np.percentile(
        array,
        lower,
        axis=0,
    )

    high = np.percentile(
        array,
        upper,
        axis=0,
    )

    winsorized = np.clip(
        array,
        low,
        high,
    )

    return normalize_profile(
        np.mean(
            winsorized,
            axis=0,
        )
    )


def profile_r_squared(
    realized: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """R-squared between realized and predicted profiles."""
    actual = _as_finite_array(
        realized,
        name="realized",
    )

    forecast = _as_finite_array(
        predicted,
        name="predicted",
    )

    if (
        actual.ndim != 1
        or forecast.ndim != 1
    ):
        raise ValueError(
            "realized and predicted must be one-dimensional."
        )

    if actual.shape != forecast.shape:
        raise ValueError(
            "realized and predicted must have identical shapes."
        )

    denominator = float(
        np.sum(
            (
                actual
                - np.mean(actual)
            )
            ** 2
        )
    )

    numerator = float(
        np.sum(
            (
                actual
                - forecast
            )
            ** 2
        )
    )

    if denominator == 0.0:
        return (
            1.0
            if np.array_equal(
                actual,
                forecast,
            )
            else 0.0
        )

    return float(
        1.0
        - numerator
        / denominator
    )


def largest_remainder_allocation(
    quantity: int,
    weights: np.ndarray,
) -> np.ndarray:
    """Allocate an integer quantity proportionally with stable tie-breaking."""
    if (
        isinstance(quantity, bool)
        or not isinstance(quantity, int)
        or quantity < 0
    ):
        raise ValueError(
            "quantity must be a non-negative integer."
        )

    normalized = normalize_profile(
        weights
    )

    raw = (
        quantity
        * normalized
    )

    base = np.floor(
        raw
    ).astype(
        np.int64
    )

    remainder = (
        quantity
        - int(
            base.sum()
        )
    )

    if remainder > 0:
        fractional = (
            raw
            - base
        )

        indices = np.arange(
            len(base)
        )

        order = np.lexsort(
            (
                indices,
                -fractional,
            )
        )

        base[
            order[:remainder]
        ] += 1

    return base
