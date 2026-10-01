from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def trapezoidal_strike_widths(
    strikes: Sequence[float],
) -> np.ndarray:
    """Trapezoidal strike widths for a sorted non-uniform strike grid."""
    k = np.asarray(
        strikes,
        dtype=float,
    )

    if (
        k.ndim != 1
        or k.size < 2
        or not np.all(np.isfinite(k))
    ):
        raise ValueError(
            "strikes must contain at least two finite values."
        )

    if np.any(
        np.diff(k) <= 0.0
    ):
        raise ValueError(
            "strikes must be strictly increasing."
        )

    widths = np.empty_like(k)

    widths[0] = (
        k[1]
        - k[0]
    )

    widths[-1] = (
        k[-1]
        - k[-2]
    )

    if k.size > 2:
        widths[1:-1] = (
            k[2:]
            - k[:-2]
        ) / 2.0

    return widths


def variance_swap_fair_variance(
    strikes: Sequence[float],
    otm_option_prices: Sequence[float],
    *,
    risk_free_rate: float,
    maturity: float,
) -> float:
    """Fair variance from static OTM-option log-contract replication."""
    k = np.asarray(
        strikes,
        dtype=float,
    )

    prices = np.asarray(
        otm_option_prices,
        dtype=float,
    )

    if (
        k.ndim != 1
        or prices.ndim != 1
        or k.shape != prices.shape
        or k.size < 2
    ):
        raise ValueError(
            "strikes and prices must be equal-length one-dimensional arrays."
        )

    if (
        not np.all(np.isfinite(prices))
        or np.any(prices < 0.0)
    ):
        raise ValueError(
            "option prices must be finite and non-negative."
        )

    t = float(maturity)
    r = float(risk_free_rate)

    if (
        not math.isfinite(t)
        or not math.isfinite(r)
        or t <= 0.0
    ):
        raise ValueError(
            "maturity must be positive and rates must be finite."
        )

    widths = trapezoidal_strike_widths(k)

    fair_variance = (
        2.0
        * math.exp(r * t)
        / t
        * np.sum(
            widths
            * prices
            / (
                k
                * k
            )
        )
    )

    return float(fair_variance)


def interpolate_at_forward(
    lower_strike: float,
    upper_strike: float,
    lower_value: float,
    upper_value: float,
    forward_price: float,
) -> float:
    """Linear interpolation between strikes that straddle a forward."""
    k_low = float(lower_strike)
    k_high = float(upper_strike)
    value_low = float(lower_value)
    value_high = float(upper_value)
    forward = float(forward_price)

    if not all(
        math.isfinite(value)
        for value in (
            k_low,
            k_high,
            value_low,
            value_high,
            forward,
        )
    ):
        raise ValueError(
            "all interpolation inputs must be finite."
        )

    if not (
        k_low
        < forward
        <= k_high
    ):
        raise ValueError(
            "strikes must straddle the forward."
        )

    weight = (
        forward
        - k_low
    ) / (
        k_high
        - k_low
    )

    return float(
        (
            1.0
            - weight
        )
        * value_low
        + weight
        * value_high
    )


def variance_swap_pnl(
    realized_volatility: float,
    fair_variance: float,
    *,
    variance_notional: float = 1.0,
) -> float:
    """P&L of a long variance swap."""
    realized_vol = float(
        realized_volatility
    )
    strike_var = float(
        fair_variance
    )
    notional = float(
        variance_notional
    )

    if not all(
        math.isfinite(value)
        for value in (
            realized_vol,
            strike_var,
            notional,
        )
    ):
        raise ValueError(
            "inputs must be finite."
        )

    if (
        realized_vol < 0.0
        or strike_var < 0.0
    ):
        raise ValueError(
            "volatility and variance must be non-negative."
        )

    return float(
        notional
        * (
            realized_vol
            * realized_vol
            - strike_var
        )
    )


__all__ = (
    "trapezoidal_strike_widths",
    "variance_swap_fair_variance",
    "interpolate_at_forward",
    "variance_swap_pnl",
)
