from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class TwoAssetCalibration:
    S1_0: float
    S2_0: float
    sigma1: float
    sigma2: float
    rho: float
    n_returns: int


def calibrate_two_asset_gbm(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    annualization: int = 252,
) -> TwoAssetCalibration:
    required = {
        "date",
        "close",
    }

    if not required.issubset(
        set(
            str(column).lower()
            for column
            in first.columns
        )
    ):
        raise RuntimeError(
            "First asset data requires date and close columns."
        )

    if not required.issubset(
        set(
            str(column).lower()
            for column
            in second.columns
        )
    ):
        raise RuntimeError(
            "Second asset data requires date and close columns."
        )

    def normalize(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        by_lower = {
            str(column).lower(): str(column)
            for column
            in frame.columns
        }

        result = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    frame[
                        by_lower[
                            "date"
                        ]
                    ],
                    errors="raise",
                ),
                "close": pd.to_numeric(
                    frame[
                        by_lower[
                            "close"
                        ]
                    ],
                    errors="raise",
                ).astype(float),
            }
        )

        return (
            result
            .sort_values(
                "date",
                kind="stable",
            )
            .drop_duplicates(
                subset=[
                    "date"
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

    first_n = normalize(
        first
    )
    second_n = normalize(
        second
    )

    aligned = first_n.merge(
        second_n,
        on="date",
        how="inner",
        suffixes=(
            "_1",
            "_2",
        ),
    )

    if len(
        aligned
    ) < 3:
        raise RuntimeError(
            "Not enough aligned observations for two-asset calibration."
        )

    returns1 = np.log(
        aligned[
            "close_1"
        ].to_numpy(
            dtype=float
        )[
            1:
        ]
        / aligned[
            "close_1"
        ].to_numpy(
            dtype=float
        )[
            :-1
        ]
    )

    returns2 = np.log(
        aligned[
            "close_2"
        ].to_numpy(
            dtype=float
        )[
            1:
        ]
        / aligned[
            "close_2"
        ].to_numpy(
            dtype=float
        )[
            :-1
        ]
    )

    sigma1 = float(
        np.std(
            returns1,
            ddof=1,
        )
        * math.sqrt(
            float(
                annualization
            )
        )
    )

    sigma2 = float(
        np.std(
            returns2,
            ddof=1,
        )
        * math.sqrt(
            float(
                annualization
            )
        )
    )

    rho = float(
        np.corrcoef(
            returns1,
            returns2,
        )[
            0,
            1,
        ]
    )

    return TwoAssetCalibration(
        S1_0=float(
            aligned[
                "close_1"
            ].iloc[
                -1
            ]
        ),
        S2_0=float(
            aligned[
                "close_2"
            ].iloc[
                -1
            ]
        ),
        sigma1=sigma1,
        sigma2=sigma2,
        rho=rho,
        n_returns=int(
            len(
                returns1
            )
        ),
    )


def exchange_volatility(
    sigma1: float,
    sigma2: float,
    rho: float,
) -> float:
    variance = (
        sigma1
        * sigma1
        + sigma2
        * sigma2
        - 2.0
        * rho
        * sigma1
        * sigma2
    )

    return float(
        math.sqrt(
            max(
                variance,
                0.0,
            )
        )
    )


def margrabe_price(
    *,
    S1: float,
    S2: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    T: float,
    q1: float,
    q2: float,
) -> tuple[
    float,
    float,
]:
    sigma_exchange = (
        exchange_volatility(
            sigma1,
            sigma2,
            rho,
        )
    )

    if (
        T <= 0.0
        or sigma_exchange <= 0.0
    ):
        price = max(
            S1
            - S2,
            0.0,
        )

        return (
            float(
                price
            ),
            float(
                sigma_exchange
            ),
        )

    root_t = math.sqrt(
        T
    )

    d1 = (
        math.log(
            S1
            / S2
        )
        + (
            q2
            - q1
            + 0.5
            * sigma_exchange
            * sigma_exchange
        )
        * T
    ) / (
        sigma_exchange
        * root_t
    )

    d2 = (
        d1
        - sigma_exchange
        * root_t
    )

    price = (
        S1
        * math.exp(
            -q1
            * T
        )
        * stats.norm.cdf(
            d1
        )
        - S2
        * math.exp(
            -q2
            * T
        )
        * stats.norm.cdf(
            d2
        )
    )

    return (
        float(
            price
        ),
        float(
            sigma_exchange
        ),
    )


def kirk_spread_price(
    *,
    S1: float,
    S2: float,
    K: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    T: float,
    r: float,
    q1: float,
    q2: float,
) -> float:
    if T <= 0.0:
        return float(
            max(
                S1
                - S2
                - K,
                0.0,
            )
        )

    forward1 = (
        S1
        * math.exp(
            (
                r
                - q1
            )
            * T
        )
    )

    forward2 = (
        S2
        * math.exp(
            (
                r
                - q2
            )
            * T
        )
    )

    denominator = (
        forward2
        + K
    )

    if denominator <= 0.0:
        raise RuntimeError(
            "Kirk approximation requires F2 + K > 0."
        )

    beta = (
        forward2
        / denominator
    )

    sigma_kirk = math.sqrt(
        max(
            sigma1
            * sigma1
            + beta
            * beta
            * sigma2
            * sigma2
            - 2.0
            * rho
            * sigma1
            * sigma2
            * beta,
            0.0,
        )
    )

    if sigma_kirk <= 0.0:
        return float(
            math.exp(
                -r
                * T
            )
            * max(
                forward1
                - denominator,
                0.0,
            )
        )

    root_t = math.sqrt(
        T
    )

    d1 = (
        math.log(
            forward1
            / denominator
        )
        + 0.5
        * sigma_kirk
        * sigma_kirk
        * T
    ) / (
        sigma_kirk
        * root_t
    )

    d2 = (
        d1
        - sigma_kirk
        * root_t
    )

    price = (
        math.exp(
            -r
            * T
        )
        * (
            forward1
            * stats.norm.cdf(
                d1
            )
            - denominator
            * stats.norm.cdf(
                d2
            )
        )
    )

    return float(
        max(
            price,
            0.0,
        )
    )


def monte_carlo_spread_prices(
    *,
    S1: float,
    S2: float,
    strikes: list[float],
    sigma1: float,
    sigma2: float,
    rho: float,
    T: float,
    r: float,
    q1: float,
    q2: float,
    n_paths: int,
    rng: np.random.Generator,
) -> list[
    tuple[
        float,
        float,
    ]
]:
    if n_paths <= 1:
        raise RuntimeError(
            "Monte Carlo requires at least two paths."
        )

    z1 = rng.standard_normal(
        n_paths
    )

    z2_independent = (
        rng.standard_normal(
            n_paths
        )
    )

    z2 = (
        rho
        * z1
        + math.sqrt(
            max(
                1.0
                - rho
                * rho,
                0.0,
            )
        )
        * z2_independent
    )

    root_t = math.sqrt(
        T
    )

    terminal1 = (
        S1
        * np.exp(
            (
                r
                - q1
                - 0.5
                * sigma1
                * sigma1
            )
            * T
            + sigma1
            * root_t
            * z1
        )
    )

    terminal2 = (
        S2
        * np.exp(
            (
                r
                - q2
                - 0.5
                * sigma2
                * sigma2
            )
            * T
            + sigma2
            * root_t
            * z2
        )
    )

    discount = math.exp(
        -r
        * T
    )

    results = []

    for strike in strikes:
        payoff = np.maximum(
            terminal1
            - terminal2
            - strike,
            0.0,
        )

        discounted = (
            discount
            * payoff
        )

        mean = float(
            np.mean(
                discounted
            )
        )

        standard_error = float(
            np.std(
                discounted,
                ddof=1,
            )
            / math.sqrt(
                float(
                    n_paths
                )
            )
        )

        results.append(
            (
                mean,
                standard_error,
            )
        )

    return results
