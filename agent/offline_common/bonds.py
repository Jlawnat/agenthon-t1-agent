from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class YieldDurationResult:
    yield_to_maturity: float
    macaulay_duration: float
    modified_duration: float


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


def discounted_cashflow_price(
    times,
    cash_flows,
    zero_rates,
) -> float:
    """Price cash flows using continuously compounded zero rates."""
    t = _finite_vector(
        times,
        name="times",
    )
    cash = _finite_vector(
        cash_flows,
        name="cash_flows",
    )
    rates = _finite_vector(
        zero_rates,
        name="zero_rates",
    )

    if not (
        t.shape
        == cash.shape
        == rates.shape
    ):
        raise ValueError(
            "times, cash_flows, and zero_rates must have identical shapes."
        )

    if np.any(t < 0.0):
        raise ValueError(
            "times must be non-negative."
        )

    return float(
        np.sum(
            cash
            * np.exp(
                -rates
                * t
            )
        )
    )


def yield_and_durations(
    *,
    price: float,
    times,
    cash_flows,
    frequency: int,
) -> YieldDurationResult:
    """Solve periodic-compounding YTM and Macaulay/modified duration."""
    price_value = float(
        price
    )

    if (
        not np.isfinite(price_value)
        or price_value <= 0.0
    ):
        raise ValueError(
            "price must be positive and finite."
        )

    if (
        isinstance(frequency, bool)
        or not isinstance(frequency, int)
        or frequency <= 0
    ):
        raise ValueError(
            "frequency must be a positive integer."
        )

    t = _finite_vector(
        times,
        name="times",
    )
    cash = _finite_vector(
        cash_flows,
        name="cash_flows",
    )

    if t.shape != cash.shape:
        raise ValueError(
            "times and cash_flows must have identical shapes."
        )

    if np.any(t <= 0.0):
        raise ValueError(
            "times must be positive."
        )

    def residual(
        ytm: float,
    ) -> float:
        base = (
            1.0
            + ytm
            / frequency
        )

        if base <= 0.0:
            return 1e12

        present_value = np.sum(
            cash
            / (
                base
                ** (
                    frequency
                    * t
                )
            )
        )

        return float(
            present_value
            - price_value
        )

    ytm = float(
        brentq(
            residual,
            -0.95,
            1.0,
            xtol=1e-14,
            rtol=1e-14,
        )
    )

    discounted = (
        cash
        / (
            (
                1.0
                + ytm
                / frequency
            )
            ** (
                frequency
                * t
            )
        )
    )

    macaulay = float(
        np.sum(
            t
            * discounted
        )
        / price_value
    )

    modified = float(
        macaulay
        / (
            1.0
            + ytm
            / frequency
        )
    )

    return YieldDurationResult(
        yield_to_maturity=ytm,
        macaulay_duration=macaulay,
        modified_duration=modified,
    )


def z_spread_from_continuous_curve(
    *,
    price: float,
    times,
    cash_flows,
    zero_rates,
    lower_bound: float = -0.05,
    upper_bound: float = 0.50,
) -> float:
    """Solve a constant spread over continuously compounded zero rates."""
    price_value = float(
        price
    )

    if (
        not np.isfinite(price_value)
        or price_value <= 0.0
    ):
        raise ValueError(
            "price must be positive and finite."
        )

    t = _finite_vector(
        times,
        name="times",
    )
    cash = _finite_vector(
        cash_flows,
        name="cash_flows",
    )
    rates = _finite_vector(
        zero_rates,
        name="zero_rates",
    )

    if not (
        t.shape
        == cash.shape
        == rates.shape
    ):
        raise ValueError(
            "times, cash_flows, and zero_rates must have identical shapes."
        )

    lower = float(
        lower_bound
    )
    upper = float(
        upper_bound
    )

    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower >= upper
    ):
        raise ValueError(
            "spread bounds must be finite and increasing."
        )

    def residual(
        spread: float,
    ) -> float:
        return float(
            np.sum(
                cash
                * np.exp(
                    -(
                        rates
                        + spread
                    )
                    * t
                )
            )
            - price_value
        )

    return float(
        brentq(
            residual,
            lower,
            upper,
            xtol=1e-14,
            rtol=1e-14,
        )
    )


def parallel_duration_convexity(
    *,
    base_price: float,
    price_up: float,
    price_down: float,
    bump: float,
) -> tuple[float, float]:
    """Central finite-difference effective duration and convexity."""
    base = float(
        base_price
    )
    up = float(
        price_up
    )
    down = float(
        price_down
    )
    shift = float(
        bump
    )

    values = np.asarray(
        [
            base,
            up,
            down,
            shift,
        ],
        dtype=float,
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "prices and bump must be finite."
        )

    if base <= 0.0:
        raise ValueError(
            "base_price must be positive."
        )

    if shift <= 0.0:
        raise ValueError(
            "bump must be positive."
        )

    duration = float(
        (
            down
            - up
        )
        / (
            2.0
            * base
            * shift
        )
    )

    convexity = float(
        (
            down
            + up
            - 2.0
            * base
        )
        / (
            base
            * shift
            * shift
        )
    )

    return (
        duration,
        convexity,
    )


def symmetric_key_rate_durations(
    *,
    base_value: float,
    values_up,
    values_down,
    bump: float,
) -> np.ndarray:
    """Key-rate durations from symmetric up/down repriced values."""
    base = float(
        base_value
    )
    shift = float(
        bump
    )

    up = _finite_vector(
        values_up,
        name="values_up",
    )
    down = _finite_vector(
        values_down,
        name="values_down",
    )

    if up.shape != down.shape:
        raise ValueError(
            "values_up and values_down must have identical shapes."
        )

    if (
        not np.isfinite(base)
        or base <= 0.0
    ):
        raise ValueError(
            "base_value must be positive and finite."
        )

    if (
        not np.isfinite(shift)
        or shift <= 0.0
    ):
        raise ValueError(
            "bump must be positive and finite."
        )

    return (
        down
        - up
    ) / (
        2.0
        * base
        * shift
    )


def dv01_from_duration(
    price: float,
    duration: float,
) -> float:
    """Approximate absolute PV change for a one-basis-point yield move."""
    price_value = float(
        price
    )
    duration_value = float(
        duration
    )

    if not np.all(
        np.isfinite(
            [
                price_value,
                duration_value,
            ]
        )
    ):
        raise ValueError(
            "price and duration must be finite."
        )

    return float(
        abs(
            price_value
            * duration_value
            * 1e-4
        )
    )
