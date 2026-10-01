from __future__ import annotations

from collections.abc import Hashable, Mapping

import numpy as np


def _finite_float(
    value: float,
    *,
    name: str,
) -> float:
    result = float(value)

    if not np.isfinite(result):
        raise ValueError(
            f"{name} must be finite."
        )

    return result


def _validated_weights(
    weights: Mapping[Hashable, float],
    *,
    name: str,
) -> dict[Hashable, float]:
    result: dict[Hashable, float] = {}

    for key, value in weights.items():
        result[key] = _finite_float(
            value,
            name=f"{name}[{key!r}]",
        )

    return result


def two_way_turnover(
    previous: Mapping[Hashable, float],
    current: Mapping[Hashable, float],
) -> float:
    """L1 turnover between two portfolio weight mappings."""
    previous_weights = _validated_weights(
        previous,
        name="previous",
    )
    current_weights = _validated_weights(
        current,
        name="current",
    )

    identifiers = (
        set(previous_weights)
        | set(current_weights)
    )

    return float(
        sum(
            abs(
                current_weights.get(
                    identifier,
                    0.0,
                )
                - previous_weights.get(
                    identifier,
                    0.0,
                )
            )
            for identifier
            in identifiers
        )
    )


def one_way_turnover(
    previous: Mapping[Hashable, float],
    current: Mapping[Hashable, float],
) -> float:
    """One-way turnover, equal to half the L1 weight change."""
    return float(
        0.5
        * two_way_turnover(
            previous,
            current,
        )
    )


def transaction_cost(
    turnover: float,
    rate: float,
) -> float:
    """Linear transaction cost from turnover and decimal cost rate."""
    turnover_value = _finite_float(
        turnover,
        name="turnover",
    )
    rate_value = _finite_float(
        rate,
        name="rate",
    )

    if turnover_value < 0.0:
        raise ValueError(
            "turnover must be non-negative."
        )

    if rate_value < 0.0:
        raise ValueError(
            "rate must be non-negative."
        )

    return float(
        turnover_value
        * rate_value
    )


def net_return_after_cost(
    gross_return: float,
    turnover: float,
    rate: float,
) -> float:
    """Subtract a linear transaction cost from gross return."""
    gross = _finite_float(
        gross_return,
        name="gross_return",
    )

    return float(
        gross
        - transaction_cost(
            turnover,
            rate,
        )
    )


def lag_positions(
    weights: np.ndarray,
    *,
    periods: int = 1,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Lag position rows so decisions cannot affect earlier returns."""
    if (
        isinstance(periods, bool)
        or not isinstance(periods, int)
        or periods < 0
    ):
        raise ValueError(
            "periods must be a non-negative integer."
        )

    array = np.asarray(
        weights,
        dtype=float,
    )

    if array.ndim not in (
        1,
        2,
    ):
        raise ValueError(
            "weights must be one- or two-dimensional."
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            "weights must be finite."
        )

    fill = _finite_float(
        fill_value,
        name="fill_value",
    )

    if periods == 0:
        return array.copy()

    result = np.full(
        array.shape,
        fill,
        dtype=float,
    )

    if periods >= len(array):
        return result

    result[periods:] = (
        array[:-periods]
    )

    return result


def rowwise_portfolio_returns(
    weights: np.ndarray,
    asset_returns: np.ndarray,
) -> np.ndarray:
    """Compute row-wise weighted portfolio returns."""
    weight_array = np.asarray(
        weights,
        dtype=float,
    )
    return_array = np.asarray(
        asset_returns,
        dtype=float,
    )

    if (
        weight_array.ndim != 2
        or return_array.ndim != 2
    ):
        raise ValueError(
            "weights and asset_returns must be two-dimensional."
        )

    if (
        weight_array.shape
        != return_array.shape
    ):
        raise ValueError(
            "weights and asset_returns must have identical shapes."
        )

    if not np.all(
        np.isfinite(weight_array)
    ):
        raise ValueError(
            "weights must be finite."
        )

    if not np.all(
        np.isfinite(return_array)
    ):
        raise ValueError(
            "asset_returns must be finite."
        )

    return np.sum(
        weight_array
        * return_array,
        axis=1,
    )
