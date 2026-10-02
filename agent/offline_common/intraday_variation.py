from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def realized_variance(
    log_returns: Sequence[float],
) -> float:
    """Sum of squared log returns."""
    values = np.asarray(
        log_returns,
        dtype=float,
    ).ravel()

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "log_returns must be finite."
        )

    return float(
        np.sum(
            values * values
        )
    )


def bandi_russell_noise_variance(
    log_returns: Sequence[float],
) -> float:
    """Bandi-Russell noise-variance estimate RV / (2n)."""
    values = np.asarray(
        log_returns,
        dtype=float,
    ).ravel()

    if (
        values.size == 0
        or not np.all(
            np.isfinite(values)
        )
    ):
        raise ValueError(
            "log_returns must contain at least one finite value."
        )

    return float(
        realized_variance(values)
        / (
            2.0
            * values.size
        )
    )


def additive_noise_corrected_variance(
    realized_variance_value: float,
    *,
    n_returns: int,
    noise_variance: float,
) -> float:
    """Subtract standard additive microstructure-noise bias."""
    rv = float(
        realized_variance_value
    )

    noise = float(
        noise_variance
    )

    if (
        isinstance(
            n_returns,
            bool,
        )
        or not isinstance(
            n_returns,
            int,
        )
        or n_returns < 0
    ):
        raise ValueError(
            "n_returns must be a non-negative integer."
        )

    if noise < 0.0:
        raise ValueError(
            "noise_variance must be non-negative."
        )

    return float(
        rv
        - 2.0
        * n_returns
        * noise
    )


def bipower_variation(
    log_returns: Sequence[float],
) -> float:
    """Barndorff-Nielsen-Shephard bipower variation."""
    values = np.asarray(
        log_returns,
        dtype=float,
    ).ravel()

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "log_returns must be finite."
        )

    if values.size < 2:
        return 0.0

    return float(
        (
            math.pi
            / 2.0
        )
        * np.sum(
            np.abs(
                values[1:]
            )
            * np.abs(
                values[:-1]
            )
        )
    )


def periodic_sampling_mask(
    offsets: Sequence[int],
    *,
    frequency: int,
) -> np.ndarray:
    """Select observations lying on an integer periodic sampling grid."""
    values = np.asarray(
        offsets
    ).ravel()

    if (
        isinstance(
            frequency,
            bool,
        )
        or not isinstance(
            frequency,
            int,
        )
        or frequency <= 0
    ):
        raise ValueError(
            "frequency must be a positive integer."
        )

    if not np.all(
        np.isfinite(values.astype(float))
    ):
        raise ValueError(
            "offsets must be finite."
        )

    return (
        values.astype(int)
        % frequency
        == 0
    )


def annualized_volatility_from_variance(
    variance: float,
    *,
    periods_per_year: float,
    clip_negative: bool = True,
) -> float:
    """Annualize a per-period variance."""
    value = float(
        variance
    )

    periods = float(
        periods_per_year
    )

    if periods <= 0.0:
        raise ValueError(
            "periods_per_year must be positive."
        )

    if clip_negative:
        value = max(
            value,
            0.0,
        )
    elif value < 0.0:
        raise ValueError(
            "variance must be non-negative when clipping is disabled."
        )

    return float(
        math.sqrt(
            value
            * periods
        )
    )


__all__ = (
    "realized_variance",
    "bandi_russell_noise_variance",
    "additive_noise_corrected_variance",
    "bipower_variation",
    "periodic_sampling_mask",
    "annualized_volatility_from_variance",
)
