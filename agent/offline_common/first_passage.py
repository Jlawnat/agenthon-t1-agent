from __future__ import annotations

import math

from agent.qf_primitives import (
    brownian_running_max_hit_probability,
    brownian_running_min_hit_probability,
)


def upper_first_passage_cdf(
    drift: float,
    volatility: float,
    time: float,
    barrier: float,
) -> float:
    """CDF for first passage of drifted Brownian motion to an upper barrier."""
    mu = float(drift)
    sigma = float(volatility)
    horizon = float(time)
    level = float(barrier)

    if sigma <= 0.0:
        raise ValueError(
            "volatility must be positive."
        )

    if horizon < 0.0:
        raise ValueError(
            "time must be non-negative."
        )

    if level <= 0.0:
        raise ValueError(
            "upper barrier must be positive."
        )

    if horizon == 0.0:
        return 0.0

    return float(
        brownian_running_max_hit_probability(
            mu,
            sigma,
            horizon,
            level,
        )
    )


def lower_first_passage_cdf(
    drift: float,
    volatility: float,
    time: float,
    barrier: float,
) -> float:
    """CDF for first passage of drifted Brownian motion to a lower barrier."""
    mu = float(drift)
    sigma = float(volatility)
    horizon = float(time)
    level = float(barrier)

    if sigma <= 0.0:
        raise ValueError(
            "volatility must be positive."
        )

    if horizon < 0.0:
        raise ValueError(
            "time must be non-negative."
        )

    if level >= 0.0:
        raise ValueError(
            "lower barrier must be negative."
        )

    if horizon == 0.0:
        return 0.0

    return float(
        brownian_running_min_hit_probability(
            mu,
            sigma,
            horizon,
            level,
        )
    )


def expected_upper_first_passage_time(
    drift: float,
    barrier: float,
) -> float:
    """Mean hitting time for a positive upper barrier when drift is positive."""
    mu = float(drift)
    level = float(barrier)

    if level <= 0.0:
        raise ValueError(
            "upper barrier must be positive."
        )

    if mu <= 0.0:
        return math.inf

    return float(
        level / mu
    )


def expected_lower_first_passage_time(
    drift: float,
    barrier: float,
) -> float:
    """Mean hitting time for a negative lower barrier when drift points downward."""
    mu = float(drift)
    level = float(barrier)

    if level >= 0.0:
        raise ValueError(
            "lower barrier must be negative."
        )

    if mu >= 0.0:
        return math.inf

    return float(
        level / mu
    )


__all__ = (
    "upper_first_passage_cdf",
    "lower_first_passage_cdf",
    "expected_upper_first_passage_time",
    "expected_lower_first_passage_time",
)
