from __future__ import annotations

import math

import numpy as np


def _finite_vector(
    values,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=float,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return array


def simple_forward_rate(
    start_discount_factor: float,
    end_discount_factor: float,
    accrual: float,
) -> float:
    """Simple forward rate implied by projection discount factors."""
    start_df = float(
        start_discount_factor
    )
    end_df = float(
        end_discount_factor
    )
    year_fraction = float(
        accrual
    )

    if not all(
        math.isfinite(value)
        for value in (
            start_df,
            end_df,
            year_fraction,
        )
    ):
        raise ValueError(
            "discount factors and accrual must be finite."
        )

    if (
        start_df <= 0.0
        or end_df <= 0.0
    ):
        raise ValueError(
            "discount factors must be positive."
        )

    if year_fraction <= 0.0:
        raise ValueError(
            "accrual must be positive."
        )

    return float(
        (
            start_df
            / end_df
            - 1.0
        )
        / year_fraction
    )


def forward_discount_factor(
    anchor_discount_factor: float,
    target_discount_factor: float,
) -> float:
    """Discount factor from an anchor date to a later target date."""
    anchor = float(
        anchor_discount_factor
    )
    target = float(
        target_discount_factor
    )

    if not all(
        math.isfinite(value)
        for value in (
            anchor,
            target,
        )
    ):
        raise ValueError(
            "discount factors must be finite."
        )

    if (
        anchor <= 0.0
        or target <= 0.0
    ):
        raise ValueError(
            "discount factors must be positive."
        )

    return float(
        target
        / anchor
    )


def fixed_leg_annuity(
    accruals,
    discount_factors,
) -> float:
    """PV per unit fixed coupon rate."""
    tau = _finite_vector(
        accruals,
        name="accruals",
    )
    dfs = _finite_vector(
        discount_factors,
        name="discount_factors",
    )

    if tau.shape != dfs.shape:
        raise ValueError(
            "accruals and discount_factors must have identical shapes."
        )

    if np.any(tau <= 0.0):
        raise ValueError(
            "accruals must be positive."
        )

    if np.any(dfs <= 0.0):
        raise ValueError(
            "discount_factors must be positive."
        )

    return float(
        np.sum(
            tau
            * dfs
        )
    )


def floating_leg_pv_per_unit(
    forward_rates,
    accruals,
    discount_factors,
) -> float:
    """PV per unit notional of floating coupons."""
    forwards = _finite_vector(
        forward_rates,
        name="forward_rates",
    )
    tau = _finite_vector(
        accruals,
        name="accruals",
    )
    dfs = _finite_vector(
        discount_factors,
        name="discount_factors",
    )

    if not (
        forwards.shape
        == tau.shape
        == dfs.shape
    ):
        raise ValueError(
            "forward_rates, accruals, and discount_factors "
            "must have identical shapes."
        )

    if np.any(tau <= 0.0):
        raise ValueError(
            "accruals must be positive."
        )

    if np.any(dfs <= 0.0):
        raise ValueError(
            "discount_factors must be positive."
        )

    return float(
        np.sum(
            forwards
            * tau
            * dfs
        )
    )


def par_swap_rate(
    *,
    fixed_accruals,
    fixed_discount_factors,
    floating_forward_rates,
    floating_accruals,
    floating_discount_factors,
) -> float:
    """Par fixed rate in a generic dual-curve swap valuation."""
    annuity = fixed_leg_annuity(
        fixed_accruals,
        fixed_discount_factors,
    )

    floating_pv = (
        floating_leg_pv_per_unit(
            floating_forward_rates,
            floating_accruals,
            floating_discount_factors,
        )
    )

    if annuity <= 0.0:
        raise ValueError(
            "fixed-leg annuity must be positive."
        )

    return float(
        floating_pv
        / annuity
    )
