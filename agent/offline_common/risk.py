from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class RiskSpec:
    notional: float
    confidence_levels: tuple[float, ...]
    age_lambda: float
    ewma_lambda: float
    horizon_days: int
    backtest_min_obs: int


def parse_risk_spec(
    instruction: str,
) -> RiskSpec:
    lowered = instruction.lower()

    money_match = re.search(
        r"portfolio notional value:\s*\$?([\d,]+(?:\.\d+)?)",
        lowered,
    )
    notional = (
        float(
            money_match.group(1).replace(
                ",",
                "",
            )
        )
        if money_match
        else 1_000_000.0
    )

    age_match = re.search(
        r"decay parameter lambda\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        lowered,
    )
    age_lambda = (
        float(
            age_match.group(1)
        )
        if age_match
        else 0.98
    )

    ewma_match = re.search(
        r"ewma with lambda\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        lowered,
    )
    ewma_lambda = (
        float(
            ewma_match.group(1)
        )
        if ewma_match
        else 0.94
    )

    horizon_match = re.search(
        r"(\d+)-day horizon",
        lowered,
    )
    horizon_days = (
        int(
            horizon_match.group(1)
        )
        if horizon_match
        else 10
    )

    minimum_match = re.search(
        r"minimum of\s+(\d+)\s+observations",
        lowered,
    )
    backtest_min_obs = (
        int(
            minimum_match.group(1)
        )
        if minimum_match
        else 252
    )

    levels = []

    for token in re.findall(
        r"(\d+(?:\.\d+)?)%",
        instruction,
    ):
        value = (
            float(
                token
            )
            / 100.0
        )
        if 0.5 < value < 1.0:
            levels.append(
                value
            )

    confidence_levels = tuple(
        sorted(
            set(
                levels
            )
        )
    )

    if not confidence_levels:
        confidence_levels = (
            0.95,
            0.99,
        )

    return RiskSpec(
        notional=notional,
        confidence_levels=confidence_levels,
        age_lambda=age_lambda,
        ewma_lambda=ewma_lambda,
        horizon_days=horizon_days,
        backtest_min_obs=backtest_min_obs,
    )


def aligned_log_returns(
    frame: pd.DataFrame,
    *,
    date_column: str,
    symbol_column: str,
    close_column: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, int],
]:
    data = frame[
        [
            date_column,
            symbol_column,
            close_column,
        ]
    ].copy()

    data[
        date_column
    ] = pd.to_datetime(
        data[
            date_column
        ],
        errors="raise",
    )

    data[
        close_column
    ] = pd.to_numeric(
        data[
            close_column
        ],
        errors="raise",
    )

    symbols = sorted(
        str(value)
        for value
        in data[
            symbol_column
        ].dropna().unique()
    )

    if len(
        symbols
    ) < 2:
        raise RuntimeError(
            "Risk engine requires at least two symbols."
        )

    counts = {
        symbol: int(
            (
                data[
                    symbol_column
                ].astype(str)
                == symbol
            ).sum()
        )
        for symbol
        in symbols
    }

    prices = (
        data.pivot_table(
            index=date_column,
            columns=symbol_column,
            values=close_column,
            aggfunc="last",
        )
        .sort_index()
    )

    prices.columns = [
        str(column)
        for column
        in prices.columns
    ]

    prices = prices[
        symbols
    ].dropna(
        how="any"
    )

    log_returns = np.log(
        prices
        / prices.shift(1)
    ).dropna(
        how="any"
    )

    return (
        prices,
        log_returns,
        counts,
    )


def historical_var_es(
    returns: np.ndarray,
    *,
    alpha: float,
    notional: float,
) -> tuple[float, float]:
    values = np.asarray(
        returns,
        dtype=float,
    )

    quantile = float(
        np.quantile(
            values,
            1.0 - alpha,
        )
    )

    tail = values[
        values <= quantile
    ]

    var_value = (
        -quantile
        * notional
    )

    es_value = (
        -float(
            np.mean(
                tail
            )
        )
        * notional
    )

    return (
        float(
            var_value
        ),
        float(
            es_value
        ),
    )


def normal_var_es(
    returns: np.ndarray,
    *,
    alpha: float,
    notional: float,
) -> tuple[float, float]:
    values = np.asarray(
        returns,
        dtype=float,
    )

    mu = float(
        np.mean(
            values
        )
    )
    sigma = float(
        np.std(
            values,
            ddof=1,
        )
    )

    z = float(
        stats.norm.ppf(
            1.0 - alpha
        )
    )

    quantile = (
        mu
        + sigma
        * z
    )

    tail_mean = (
        mu
        - sigma
        * stats.norm.pdf(
            z
        )
        / (
            1.0
            - alpha
        )
    )

    return (
        float(
            -quantile
            * notional
        ),
        float(
            -tail_mean
            * notional
        ),
    )


def student_t_var_es(
    returns: np.ndarray,
    *,
    alpha: float,
    notional: float,
) -> tuple[float, float]:
    values = np.asarray(
        returns,
        dtype=float,
    )

    df, loc, scale = stats.t.fit(
        values
    )

    q_standard = float(
        stats.t.ppf(
            1.0 - alpha,
            df,
        )
    )

    quantile = (
        float(
            loc
        )
        + float(
            scale
        )
        * q_standard
    )

    if df <= 1.0:
        tail_mean = quantile
    else:
        density = float(
            stats.t.pdf(
                q_standard,
                df,
            )
        )

        standardized_tail_mean = (
            -(
                df
                + q_standard
                * q_standard
            )
            / (
                df
                - 1.0
            )
            * density
            / (
                1.0
                - alpha
            )
        )

        tail_mean = (
            float(
                loc
            )
            + float(
                scale
            )
            * standardized_tail_mean
        )

    return (
        float(
            -quantile
            * notional
        ),
        float(
            -tail_mean
            * notional
        ),
    )


def exponential_weights(
    count: int,
    *,
    decay: float,
) -> np.ndarray:
    if count <= 0:
        raise RuntimeError(
            "Weight count must be positive."
        )

    ages = np.arange(
        count - 1,
        -1,
        -1,
        dtype=float,
    )

    weights = (
        float(
            decay
        )
        ** ages
    )

    return (
        weights
        / weights.sum()
    )


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    probability: float,
) -> float:
    values = np.asarray(
        values,
        dtype=float,
    )
    weights = np.asarray(
        weights,
        dtype=float,
    )

    order = np.argsort(
        values,
        kind="stable",
    )

    sorted_values = values[
        order
    ]
    sorted_weights = weights[
        order
    ]

    cumulative = np.cumsum(
        sorted_weights
    )

    index = int(
        np.searchsorted(
            cumulative,
            probability,
            side="left",
        )
    )

    index = min(
        index,
        len(
            sorted_values
        )
        - 1,
    )

    return float(
        sorted_values[
            index
        ]
    )


def age_weighted_var_es(
    returns: np.ndarray,
    *,
    alpha: float,
    notional: float,
    decay: float,
) -> tuple[float, float]:
    values = np.asarray(
        returns,
        dtype=float,
    )

    weights = exponential_weights(
        len(
            values
        ),
        decay=decay,
    )

    quantile = weighted_quantile(
        values,
        weights,
        probability=(
            1.0
            - alpha
        ),
    )

    mask = (
        values <= quantile
    )

    tail_weight = float(
        weights[
            mask
        ].sum()
    )

    if tail_weight <= 0.0:
        tail_mean = quantile
    else:
        tail_mean = float(
            np.sum(
                values[
                    mask
                ]
                * weights[
                    mask
                ]
            )
            / tail_weight
        )

    return (
        float(
            -quantile
            * notional
        ),
        float(
            -tail_mean
            * notional
        ),
    )


def ewma_covariance(
    asset_returns: np.ndarray,
    *,
    decay: float,
) -> np.ndarray:
    matrix = np.asarray(
        asset_returns,
        dtype=float,
    )

    if (
        matrix.ndim != 2
        or matrix.shape[
            0
        ] < 2
    ):
        raise RuntimeError(
            "EWMA covariance requires a 2D return matrix."
        )

    covariance = np.cov(
        matrix,
        rowvar=False,
        ddof=1,
    )

    for observation in matrix:
        outer = np.outer(
            observation,
            observation,
        )

        covariance = (
            decay
            * covariance
            + (
                1.0
                - decay
            )
            * outer
        )

    return covariance


def ewma_portfolio_volatility(
    asset_returns: np.ndarray,
    *,
    decay: float,
    weights: np.ndarray,
) -> float:
    covariance = ewma_covariance(
        asset_returns,
        decay=decay,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    variance = float(
        weights
        @ covariance
        @ weights
    )

    return float(
        math.sqrt(
            max(
                variance,
                0.0,
            )
        )
    )


def conditional_normal_var_es(
    sigma: float,
    *,
    alpha: float,
    notional: float,
) -> tuple[float, float]:
    # RiskMetrics-style conditional normal VaR uses zero conditional mean.
    z = float(
        stats.norm.ppf(
            1.0
            - alpha
        )
    )

    var_value = (
        -sigma
        * z
        * notional
    )

    es_value = (
        sigma
        * stats.norm.pdf(
            z
        )
        / (
            1.0
            - alpha
        )
        * notional
    )

    return (
        float(
            var_value
        ),
        float(
            es_value
        ),
    )


def overlapping_horizon_returns(
    returns: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    values = np.asarray(
        returns,
        dtype=float,
    )

    if horizon <= 0:
        raise RuntimeError(
            "Horizon must be positive."
        )

    if len(
        values
    ) < horizon:
        raise RuntimeError(
            "Not enough returns for requested horizon."
        )

    cumulative = np.cumsum(
        np.insert(
            values,
            0,
            0.0,
        )
    )

    return (
        cumulative[
            horizon:
        ]
        - cumulative[
            :-horizon
        ]
    )


def expanding_historical_backtest(
    returns: np.ndarray,
    *,
    alpha: float,
    minimum_observations: int,
) -> tuple[
    int,
    int,
    float,
    float,
]:
    values = np.asarray(
        returns,
        dtype=float,
    )

    exceedances = 0

    for index in range(
        minimum_observations,
        len(
            values
        ),
    ):
        threshold = float(
            np.quantile(
                values[
                    :index
                ],
                1.0
                - alpha,
            )
        )

        if values[
            index
        ] < threshold:
            exceedances += 1

    test_days = (
        len(
            values
        )
        - minimum_observations
    )

    exceedance_rate = (
        float(
            exceedances
            / test_days
        )
        if test_days > 0
        else 0.0
    )

    expected_rate = (
        1.0
        - alpha
    )

    if (
        test_days <= 0
        or exceedances in {
            0,
            test_days,
        }
    ):
        kupiec_pvalue = 1.0
    else:
        observed_rate = (
            exceedances
            / test_days
        )

        log_null = (
            (
                test_days
                - exceedances
            )
            * math.log(
                1.0
                - expected_rate
            )
            + exceedances
            * math.log(
                expected_rate
            )
        )

        log_alt = (
            (
                test_days
                - exceedances
            )
            * math.log(
                1.0
                - observed_rate
            )
            + exceedances
            * math.log(
                observed_rate
            )
        )

        lr_pof = (
            -2.0
            * (
                log_null
                - log_alt
            )
        )

        kupiec_pvalue = float(
            stats.chi2.sf(
                lr_pof,
                1,
            )
        )

    return (
        int(
            test_days
        ),
        int(
            exceedances
        ),
        float(
            exceedance_rate
        ),
        float(
            kupiec_pvalue
        ),
    )
