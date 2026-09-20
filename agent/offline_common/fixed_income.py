from __future__ import annotations

from dataclasses import dataclass
import math

from scipy.optimize import brentq


@dataclass(frozen=True)
class BootstrappedCurve:
    maturities: tuple[float, ...]
    zero_rates: tuple[float, ...]
    discount_factors: tuple[float, ...]
    forward_rates: tuple[float, ...]


def discount_factor_from_zero(
    zero_rate: float,
    maturity: float,
) -> float:
    return float(
        math.exp(
            -float(
                zero_rate
            )
            * float(
                maturity
            )
        )
    )


def interpolate_zero_rate(
    maturity: float,
    *,
    known_maturities: list[float],
    known_zero_rates: list[float],
) -> float:
    if not known_maturities:
        raise RuntimeError(
            "At least one zero-rate knot is required."
        )

    t = float(
        maturity
    )

    if t <= known_maturities[0]:
        return float(
            known_zero_rates[
                0
            ]
        )

    if t >= known_maturities[-1]:
        return float(
            known_zero_rates[
                -1
            ]
        )

    for index in range(
        len(
            known_maturities
        )
        - 1
    ):
        left_t = float(
            known_maturities[
                index
            ]
        )
        right_t = float(
            known_maturities[
                index
                + 1
            ]
        )

        if left_t <= t <= right_t:
            left_z = float(
                known_zero_rates[
                    index
                ]
            )
            right_z = float(
                known_zero_rates[
                    index
                    + 1
                ]
            )

            weight = (
                t
                - left_t
            ) / (
                right_t
                - left_t
            )

            return float(
                left_z
                + weight
                * (
                    right_z
                    - left_z
                )
            )

    raise RuntimeError(
        f"Could not interpolate zero rate at maturity {t}."
    )


def _candidate_discount_factor(
    maturity: float,
    *,
    previous_maturities: list[float],
    previous_zero_rates: list[float],
    current_maturity: float,
    current_zero_rate: float,
) -> float:
    knots_t = [
        *previous_maturities,
        float(
            current_maturity
        ),
    ]
    knots_z = [
        *previous_zero_rates,
        float(
            current_zero_rate
        ),
    ]

    z = interpolate_zero_rate(
        float(
            maturity
        ),
        known_maturities=knots_t,
        known_zero_rates=knots_z,
    )

    return discount_factor_from_zero(
        z,
        float(
            maturity
        ),
    )


def bootstrap_par_curve(
    maturities: list[float],
    par_rates: list[float],
    *,
    coupon_frequency: int,
) -> BootstrappedCurve:
    if len(
        maturities
    ) != len(
        par_rates
    ):
        raise RuntimeError(
            "maturities and par_rates must have the same length."
        )

    if not maturities:
        raise RuntimeError(
            "Yield curve input is empty."
        )

    if coupon_frequency <= 0:
        raise RuntimeError(
            "coupon_frequency must be positive."
        )

    pairs = sorted(
        (
            float(
                maturity
            ),
            float(
                par_rate
            ),
        )
        for maturity, par_rate
        in zip(
            maturities,
            par_rates,
        )
    )

    sorted_maturities = [
        pair[
            0
        ]
        for pair
        in pairs
    ]
    sorted_par_rates = [
        pair[
            1
        ]
        for pair
        in pairs
    ]

    if any(
        maturity <= 0.0
        for maturity
        in sorted_maturities
    ):
        raise RuntimeError(
            "All maturities must be positive."
        )

    if any(
        later <= earlier
        for earlier, later
        in zip(
            sorted_maturities[:-1],
            sorted_maturities[1:],
        )
    ):
        raise RuntimeError(
            "Maturities must be unique."
        )

    known_maturities: list[
        float
    ] = []
    known_zero_rates: list[
        float
    ] = []

    frequency = float(
        coupon_frequency
    )

    for maturity, par_rate in zip(
        sorted_maturities,
        sorted_par_rates,
    ):
        number_of_payments_float = (
            maturity
            * frequency
        )

        number_of_payments = int(
            round(
                number_of_payments_float
            )
        )

        if not math.isclose(
            number_of_payments_float,
            float(
                number_of_payments
            ),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise RuntimeError(
                "Maturity must lie on the coupon-payment grid."
            )

        coupon = (
            par_rate
            / frequency
        )

        def par_price_error(
            candidate_zero: float,
        ) -> float:
            price = 0.0

            for payment_index in range(
                1,
                number_of_payments
            ):
                payment_time = (
                    payment_index
                    / frequency
                )

                df = _candidate_discount_factor(
                    payment_time,
                    previous_maturities=(
                        known_maturities
                    ),
                    previous_zero_rates=(
                        known_zero_rates
                    ),
                    current_maturity=(
                        maturity
                    ),
                    current_zero_rate=(
                        candidate_zero
                    ),
                )

                price += (
                    coupon
                    * df
                )

            final_df = (
                discount_factor_from_zero(
                    candidate_zero,
                    maturity,
                )
            )

            price += (
                1.0
                + coupon
            ) * final_df

            return float(
                price
                - 1.0
            )

        lower = -0.10
        upper = 0.50

        lower_value = (
            par_price_error(
                lower
            )
        )
        upper_value = (
            par_price_error(
                upper
            )
        )

        if (
            lower_value
            * upper_value
            > 0.0
        ):
            # Expand the bracket for unusual but still valid curves.
            lower = -0.50
            upper = 2.00
            lower_value = (
                par_price_error(
                    lower
                )
            )
            upper_value = (
                par_price_error(
                    upper
                )
            )

        if (
            lower_value
            * upper_value
            > 0.0
        ):
            raise RuntimeError(
                f"Could not bracket zero rate for maturity {maturity}."
            )

        zero_rate = float(
            brentq(
                par_price_error,
                lower,
                upper,
                xtol=1e-14,
                rtol=1e-14,
                maxiter=200,
            )
        )

        known_maturities.append(
            maturity
        )
        known_zero_rates.append(
            zero_rate
        )

    discount_factors = [
        discount_factor_from_zero(
            zero_rate,
            maturity,
        )
        for maturity, zero_rate
        in zip(
            known_maturities,
            known_zero_rates,
        )
    ]

    forward_rates: list[
        float
    ] = []

    for maturity, zero_rate in zip(
        known_maturities,
        known_zero_rates,
    ):
        if math.isclose(
            maturity,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            forward_rate = (
                zero_rate
            )
        elif maturity > 1.0:
            previous_time = (
                maturity
                - 1.0
            )

            previous_zero = (
                interpolate_zero_rate(
                    previous_time,
                    known_maturities=(
                        known_maturities
                    ),
                    known_zero_rates=(
                        known_zero_rates
                    ),
                )
            )

            forward_rate = (
                zero_rate
                * maturity
                - previous_zero
                * previous_time
            )
        else:
            # For sub-1Y curve points, the natural continuously
            # compounded forward from time zero equals the zero rate.
            forward_rate = (
                zero_rate
            )

        forward_rates.append(
            float(
                forward_rate
            )
        )

    return BootstrappedCurve(
        maturities=tuple(
            known_maturities
        ),
        zero_rates=tuple(
            known_zero_rates
        ),
        discount_factors=tuple(
            discount_factors
        ),
        forward_rates=tuple(
            forward_rates
        ),
    )


def reprice_par_bond(
    *,
    maturity: float,
    par_rate: float,
    coupon_frequency: int,
    curve: BootstrappedCurve,
) -> float:
    frequency = float(
        coupon_frequency
    )

    number_of_payments = int(
        round(
            maturity
            * frequency
        )
    )

    coupon = (
        float(
            par_rate
        )
        / frequency
    )

    maturity_knots = list(
        curve.maturities
    )
    zero_knots = list(
        curve.zero_rates
    )

    price = 0.0

    for payment_index in range(
        1,
        number_of_payments
    ):
        payment_time = (
            payment_index
            / frequency
        )

        zero_rate = (
            interpolate_zero_rate(
                payment_time,
                known_maturities=(
                    maturity_knots
                ),
                known_zero_rates=(
                    zero_knots
                ),
            )
        )

        price += (
            coupon
            * discount_factor_from_zero(
                zero_rate,
                payment_time,
            )
        )

    maturity_zero = (
        interpolate_zero_rate(
            maturity,
            known_maturities=(
                maturity_knots
            ),
            known_zero_rates=(
                zero_knots
            ),
        )
    )

    price += (
        1.0
        + coupon
    ) * discount_factor_from_zero(
        maturity_zero,
        maturity,
    )

    return float(
        price
    )
