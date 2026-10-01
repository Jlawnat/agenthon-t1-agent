from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class CapFloorStripResult:
    cap_price: float
    floor_price: float
    caplets: np.ndarray
    floorlets: np.ndarray
    swap_value: float
    parity_error: float


def black_caplet(
    *,
    forward_rate: float,
    strike: float,
    volatility: float,
    fixing_time: float,
    discount_factor: float,
    accrual: float,
    notional: float = 1.0,
) -> float:
    forward = float(forward_rate)
    k = float(strike)
    sigma = float(volatility)
    t = float(fixing_time)
    df = float(discount_factor)
    tau = float(accrual)
    n = float(notional)

    if forward <= 0.0 or k <= 0.0:
        raise ValueError(
            "forward_rate and strike must be positive."
        )

    if sigma < 0.0 or t < 0.0:
        raise ValueError(
            "volatility and fixing_time must be non-negative."
        )

    if df <= 0.0 or tau <= 0.0 or n <= 0.0:
        raise ValueError(
            "discount_factor, accrual, and notional must be positive."
        )

    if t == 0.0 or sigma == 0.0:
        return float(
            n
            * tau
            * df
            * max(
                forward - k,
                0.0,
            )
        )

    root_t = math.sqrt(t)

    d1 = (
        math.log(
            forward / k
        )
        + 0.5
        * sigma
        * sigma
        * t
    ) / (
        sigma
        * root_t
    )

    d2 = (
        d1
        - sigma
        * root_t
    )

    return float(
        n
        * tau
        * df
        * (
            forward
            * norm.cdf(d1)
            - k
            * norm.cdf(d2)
        )
    )


def black_floorlet(
    *,
    forward_rate: float,
    strike: float,
    volatility: float,
    fixing_time: float,
    discount_factor: float,
    accrual: float,
    notional: float = 1.0,
) -> float:
    forward = float(forward_rate)
    k = float(strike)
    sigma = float(volatility)
    t = float(fixing_time)
    df = float(discount_factor)
    tau = float(accrual)
    n = float(notional)

    if forward <= 0.0 or k <= 0.0:
        raise ValueError(
            "forward_rate and strike must be positive."
        )

    if sigma < 0.0 or t < 0.0:
        raise ValueError(
            "volatility and fixing_time must be non-negative."
        )

    if df <= 0.0 or tau <= 0.0 or n <= 0.0:
        raise ValueError(
            "discount_factor, accrual, and notional must be positive."
        )

    if t == 0.0 or sigma == 0.0:
        return float(
            n
            * tau
            * df
            * max(
                k - forward,
                0.0,
            )
        )

    root_t = math.sqrt(t)

    d1 = (
        math.log(
            forward / k
        )
        + 0.5
        * sigma
        * sigma
        * t
    ) / (
        sigma
        * root_t
    )

    d2 = (
        d1
        - sigma
        * root_t
    )

    return float(
        n
        * tau
        * df
        * (
            k
            * norm.cdf(-d2)
            - forward
            * norm.cdf(-d1)
        )
    )


def black_cap_floor_strip(
    forward_rates: Sequence[float],
    discount_factors: Sequence[float],
    *,
    strike: float,
    volatility: float,
    accrual: float,
    notional: float = 1.0,
) -> CapFloorStripResult:
    forwards = np.asarray(
        forward_rates,
        dtype=float,
    )

    dfs = np.asarray(
        discount_factors,
        dtype=float,
    )

    if (
        forwards.ndim != 1
        or dfs.ndim != 1
        or forwards.size == 0
        or forwards.shape != dfs.shape
    ):
        raise ValueError(
            "forward_rates and discount_factors must be "
            "non-empty one-dimensional arrays of identical shape."
        )

    if (
        not np.all(np.isfinite(forwards))
        or not np.all(np.isfinite(dfs))
    ):
        raise ValueError(
            "forward_rates and discount_factors must be finite."
        )

    caplets = np.asarray(
        [
            black_caplet(
                forward_rate=float(forward),
                strike=strike,
                volatility=volatility,
                fixing_time=index * float(accrual),
                discount_factor=float(df),
                accrual=accrual,
                notional=notional,
            )
            for index, (forward, df)
            in enumerate(
                zip(
                    forwards,
                    dfs,
                )
            )
        ],
        dtype=float,
    )

    floorlets = np.asarray(
        [
            black_floorlet(
                forward_rate=float(forward),
                strike=strike,
                volatility=volatility,
                fixing_time=index * float(accrual),
                discount_factor=float(df),
                accrual=accrual,
                notional=notional,
            )
            for index, (forward, df)
            in enumerate(
                zip(
                    forwards,
                    dfs,
                )
            )
        ],
        dtype=float,
    )

    swap_value = float(
        float(notional)
        * float(accrual)
        * np.sum(
            dfs
            * (
                forwards
                - float(strike)
            )
        )
    )

    cap_price = float(
        np.sum(caplets)
    )

    floor_price = float(
        np.sum(floorlets)
    )

    parity_error = float(
        cap_price
        - floor_price
        - swap_value
    )

    return CapFloorStripResult(
        cap_price=cap_price,
        floor_price=floor_price,
        caplets=caplets,
        floorlets=floorlets,
        swap_value=swap_value,
        parity_error=parity_error,
    )


__all__ = (
    "CapFloorStripResult",
    "black_caplet",
    "black_floorlet",
    "black_cap_floor_strip",
)
