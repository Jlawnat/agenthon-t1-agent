from __future__ import annotations

import math

import numpy as np


def _finite_float(
    value,
    *,
    name: str,
) -> float:
    result = float(
        value
    )

    if not math.isfinite(
        result
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return result


def covered_interest_forward(
    spot: float,
    *,
    base_rate: float,
    quote_rate: float,
    base_year_fraction: float,
    quote_year_fraction: float,
) -> float:
    """
    Forward for a BASE/QUOTE FX pair under simple-compounded rates.

    The spot and forward are units of quote currency per unit of base
    currency.
    """
    s = _finite_float(
        spot,
        name="spot",
    )
    rb = _finite_float(
        base_rate,
        name="base_rate",
    )
    rq = _finite_float(
        quote_rate,
        name="quote_rate",
    )
    tb = _finite_float(
        base_year_fraction,
        name="base_year_fraction",
    )
    tq = _finite_float(
        quote_year_fraction,
        name="quote_year_fraction",
    )

    if s <= 0.0:
        raise ValueError(
            "spot must be positive."
        )

    if (
        tb < 0.0
        or tq < 0.0
    ):
        raise ValueError(
            "year fractions must be non-negative."
        )

    base_growth = (
        1.0
        + rb
        * tb
    )

    quote_growth = (
        1.0
        + rq
        * tq
    )

    if (
        base_growth <= 0.0
        or quote_growth <= 0.0
    ):
        raise ValueError(
            "simple-compounding growth factors must be positive."
        )

    return float(
        s
        * quote_growth
        / base_growth
    )


def invert_bid_ask(
    bid: float,
    ask: float,
) -> tuple[
    float,
    float,
]:
    """Invert a positive FX bid/ask quote with the correct side flip."""
    b = _finite_float(
        bid,
        name="bid",
    )
    a = _finite_float(
        ask,
        name="ask",
    )

    if (
        b <= 0.0
        or a <= 0.0
        or b > a
    ):
        raise ValueError(
            "require 0 < bid <= ask."
        )

    return (
        float(
            1.0
            / a
        ),
        float(
            1.0
            / b
        ),
    )


def synthetic_cross_bid_ask(
    *,
    base_usd_bid: float,
    base_usd_ask: float,
    quote_usd_bid: float,
    quote_usd_ask: float,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Construct BASE/QUOTE from BASE/USD and QUOTE/USD quotes.

    Inputs are USD units per unit of each currency, corresponding to
    BASE/USD and QUOTE/USD quotes.
    """
    bb = _finite_float(
        base_usd_bid,
        name="base_usd_bid",
    )
    ba = _finite_float(
        base_usd_ask,
        name="base_usd_ask",
    )
    qb = _finite_float(
        quote_usd_bid,
        name="quote_usd_bid",
    )
    qa = _finite_float(
        quote_usd_ask,
        name="quote_usd_ask",
    )

    if (
        bb <= 0.0
        or ba <= 0.0
        or qb <= 0.0
        or qa <= 0.0
        or bb > ba
        or qb > qa
    ):
        raise ValueError(
            "FX inputs must satisfy 0 < bid <= ask."
        )

    bid = float(
        bb
        / qa
    )

    ask = float(
        ba
        / qb
    )

    mid = float(
        (
            bid
            + ask
        )
        / 2.0
    )

    return (
        bid,
        ask,
        mid,
    )


def forward_points(
    spot: float,
    forward: float,
    *,
    pip_multiplier: float = 10000.0,
) -> float:
    s = _finite_float(
        spot,
        name="spot",
    )
    f = _finite_float(
        forward,
        name="forward",
    )
    multiplier = _finite_float(
        pip_multiplier,
        name="pip_multiplier",
    )

    if (
        s <= 0.0
        or f <= 0.0
    ):
        raise ValueError(
            "spot and forward must be positive."
        )

    if multiplier <= 0.0:
        raise ValueError(
            "pip_multiplier must be positive."
        )

    return float(
        (
            f
            - s
        )
        * multiplier
    )


def implied_quote_rate_from_forward(
    spot: float,
    forward: float,
    *,
    base_rate: float,
    base_year_fraction: float,
    quote_year_fraction: float,
) -> float:
    """
    Recover the quote-currency simple rate from a BASE/QUOTE forward.
    """
    s = _finite_float(
        spot,
        name="spot",
    )
    f = _finite_float(
        forward,
        name="forward",
    )
    rb = _finite_float(
        base_rate,
        name="base_rate",
    )
    tb = _finite_float(
        base_year_fraction,
        name="base_year_fraction",
    )
    tq = _finite_float(
        quote_year_fraction,
        name="quote_year_fraction",
    )

    if (
        s <= 0.0
        or f <= 0.0
    ):
        raise ValueError(
            "spot and forward must be positive."
        )

    if tb < 0.0:
        raise ValueError(
            "base_year_fraction must be non-negative."
        )

    if tq <= 0.0:
        raise ValueError(
            "quote_year_fraction must be positive."
        )

    return float(
        (
            (
                f
                / s
            )
            * (
                1.0
                + rb
                * tb
            )
            - 1.0
        )
        / tq
    )


def continuous_basis_bps(
    quoted_forward: float,
    cip_forward: float,
    year_fraction: float,
) -> float:
    """
    Continuous FX basis satisfying
    quoted = CIP * exp(-basis * tau).
    """
    quoted = _finite_float(
        quoted_forward,
        name="quoted_forward",
    )
    cip = _finite_float(
        cip_forward,
        name="cip_forward",
    )
    tau = _finite_float(
        year_fraction,
        name="year_fraction",
    )

    if (
        quoted <= 0.0
        or cip <= 0.0
    ):
        raise ValueError(
            "forward rates must be positive."
        )

    if tau <= 0.0:
        raise ValueError(
            "year_fraction must be positive."
        )

    return float(
        -math.log(
            quoted
            / cip
        )
        / tau
        * 10000.0
    )


def forward_from_continuous_basis(
    cip_forward: float,
    basis_bps: float,
    year_fraction: float,
) -> float:
    cip = _finite_float(
        cip_forward,
        name="cip_forward",
    )
    basis = _finite_float(
        basis_bps,
        name="basis_bps",
    )
    tau = _finite_float(
        year_fraction,
        name="year_fraction",
    )

    if cip <= 0.0:
        raise ValueError(
            "cip_forward must be positive."
        )

    if tau < 0.0:
        raise ValueError(
            "year_fraction must be non-negative."
        )

    return float(
        cip
        * math.exp(
            -(
                basis
                / 10000.0
            )
            * tau
        )
    )


def log_linear_forward(
    spot: float,
    node_times,
    node_forwards,
    target_time: float,
) -> float:
    """Log-linear interpolation of positive FX forwards anchored at spot."""
    s = _finite_float(
        spot,
        name="spot",
    )
    target = _finite_float(
        target_time,
        name="target_time",
    )

    times = np.asarray(
        node_times,
        dtype=float,
    )
    forwards = np.asarray(
        node_forwards,
        dtype=float,
    )

    if (
        times.ndim != 1
        or forwards.ndim != 1
        or times.shape != forwards.shape
    ):
        raise ValueError(
            "node_times and node_forwards must be matching one-dimensional arrays."
        )

    if times.size == 0:
        raise ValueError(
            "at least one forward node is required."
        )

    if (
        not np.all(
            np.isfinite(times)
        )
        or not np.all(
            np.isfinite(forwards)
        )
    ):
        raise ValueError(
            "forward nodes must be finite."
        )

    if (
        s <= 0.0
        or np.any(
            forwards <= 0.0
        )
    ):
        raise ValueError(
            "spot and forward nodes must be positive."
        )

    if (
        np.any(
            times <= 0.0
        )
        or np.any(
            np.diff(times)
            <= 0.0
        )
    ):
        raise ValueError(
            "node_times must be positive and strictly increasing."
        )

    if target <= 0.0:
        return s

    xs = np.concatenate(
        (
            np.array(
                [0.0]
            ),
            times,
        )
    )

    values = np.concatenate(
        (
            np.array(
                [s]
            ),
            forwards,
        )
    )

    return float(
        math.exp(
            np.interp(
                target,
                xs,
                np.log(
                    values
                ),
            )
        )
    )
