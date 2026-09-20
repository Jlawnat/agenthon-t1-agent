from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata


@dataclass(frozen=True)
class EventStudySpec:
    estimation_start: int = -130
    estimation_end: int = -11
    event_start: int = -5
    event_end: int = 5
    minimum_estimation_observations: int = 60


@dataclass
class EventRecord:
    event_id: int
    ticker: str
    event_date: str
    alpha: float
    beta: float
    r_squared: float
    sigma: float
    n_obs: int
    market_mean: float
    market_ss: float
    estimation_dates: list[str]
    estimation_residuals: np.ndarray
    event_relative_days: np.ndarray
    event_dates: list[str]
    event_market_returns: np.ndarray
    event_abnormal_returns: np.ndarray
    event_standardized_returns: np.ndarray


def _log_returns(
    prices: pd.DataFrame,
    *,
    date_column: str,
    value_columns: list[str],
) -> pd.DataFrame:
    frame = prices[
        [
            date_column,
            *value_columns,
        ]
    ].copy()

    frame[
        date_column
    ] = pd.to_datetime(
        frame[
            date_column
        ],
        errors="raise",
    )

    frame = (
        frame
        .sort_values(
            date_column,
            kind="stable",
        )
        .drop_duplicates(
            subset=[
                date_column
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    for column in value_columns:
        frame[
            column
        ] = pd.to_numeric(
            frame[
                column
            ],
            errors="coerce",
        )

    returns = np.log(
        frame[
            value_columns
        ]
        / frame[
            value_columns
        ].shift(
            1
        )
    )

    result = returns.iloc[
        1:
    ].copy()

    result.insert(
        0,
        "date",
        frame[
            date_column
        ].iloc[
            1:
        ].to_numpy(),
    )

    return result.dropna(
        how="any"
    ).reset_index(
        drop=True
    )


def prepare_event_study_returns(
    stock_prices: pd.DataFrame,
    market_prices: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    stock_by_lower = {
        str(column).lower(): str(column)
        for column
        in stock_prices.columns
    }
    market_by_lower = {
        str(column).lower(): str(column)
        for column
        in market_prices.columns
    }

    if "date" not in stock_by_lower:
        raise RuntimeError(
            "Stock price data requires a date column."
        )

    if "date" not in market_by_lower:
        raise RuntimeError(
            "Market price data requires a date column."
        )

    stock_date = (
        stock_by_lower[
            "date"
        ]
    )
    market_date = (
        market_by_lower[
            "date"
        ]
    )

    tickers = [
        str(column)
        for column
        in stock_prices.columns
        if str(column) != stock_date
    ]

    if not tickers:
        raise RuntimeError(
            "Stock price data contains no ticker columns."
        )

    market_candidates = [
        str(column)
        for column
        in market_prices.columns
        if str(column) != market_date
    ]

    if not market_candidates:
        raise RuntimeError(
            "Market price data contains no price column."
        )

    market_column = (
        market_by_lower.get(
            "adj_close"
        )
        or market_by_lower.get(
            "close"
        )
        or market_candidates[
            0
        ]
    )

    stock_returns = _log_returns(
        stock_prices,
        date_column=stock_date,
        value_columns=tickers,
    )

    market_returns = _log_returns(
        market_prices,
        date_column=market_date,
        value_columns=[
            market_column
        ],
    ).rename(
        columns={
            market_column: "market_return"
        }
    )

    merged = stock_returns.merge(
        market_returns[
            [
                "date",
                "market_return",
            ]
        ],
        on="date",
        how="inner",
    ).sort_values(
        "date",
        kind="stable",
    ).reset_index(
        drop=True
    )

    return (
        merged,
        tickers,
    )


def _fit_market_model(
    stock_returns: np.ndarray,
    market_returns: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
    np.ndarray,
    float,
    float,
]:
    y = np.asarray(
        stock_returns,
        dtype=float,
    )
    x = np.asarray(
        market_returns,
        dtype=float,
    )

    X = np.column_stack(
        (
            np.ones(
                len(
                    x
                ),
                dtype=float,
            ),
            x,
        )
    )

    coefficients, *_ = (
        np.linalg.lstsq(
            X,
            y,
            rcond=None,
        )
    )

    alpha = float(
        coefficients[
            0
        ]
    )
    beta = float(
        coefficients[
            1
        ]
    )

    fitted = (
        X
        @ coefficients
    )

    residuals = (
        y
        - fitted
    )

    rss = float(
        np.sum(
            residuals
            * residuals
        )
    )

    centered = (
        y
        - np.mean(
            y
        )
    )

    tss = float(
        np.sum(
            centered
            * centered
        )
    )

    r_squared = (
        float(
            1.0
            - rss
            / tss
        )
        if tss > 0.0
        else 0.0
    )

    degrees_freedom = (
        len(
            y
        )
        - 2
    )

    if degrees_freedom <= 0:
        raise RuntimeError(
            "Not enough observations for market-model residual variance."
        )

    sigma = float(
        math.sqrt(
            rss
            / degrees_freedom
        )
    )

    market_mean = float(
        np.mean(
            x
        )
    )

    market_ss = float(
        np.sum(
            (
                x
                - market_mean
            )
            ** 2
        )
    )

    return (
        alpha,
        beta,
        r_squared,
        sigma,
        residuals,
        market_mean,
        market_ss,
    )


def build_event_records(
    merged_returns: pd.DataFrame,
    events: pd.DataFrame,
    *,
    spec: EventStudySpec,
) -> list[
    EventRecord
]:
    events_by_lower = {
        str(column).lower(): str(column)
        for column
        in events.columns
    }

    if not {
        "ticker",
        "event_date",
    }.issubset(
        events_by_lower
    ):
        raise RuntimeError(
            "Event data requires ticker and event_date columns."
        )

    ticker_column = (
        events_by_lower[
            "ticker"
        ]
    )
    event_date_column = (
        events_by_lower[
            "event_date"
        ]
    )

    working_events = events[
        [
            ticker_column,
            event_date_column,
        ]
    ].copy()

    working_events[
        event_date_column
    ] = pd.to_datetime(
        working_events[
            event_date_column
        ],
        errors="raise",
    )

    date_strings = (
        merged_returns[
            "date"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
        .tolist()
    )

    date_to_index = {
        date: index
        for index, date
        in enumerate(
            date_strings
        )
    }

    records: list[
        EventRecord
    ] = []

    relative_days = np.arange(
        spec.event_start,
        spec.event_end
        + 1,
        dtype=int,
    )

    for event_id, row in working_events.iterrows():
        ticker = str(
            row[
                ticker_column
            ]
        )

        if ticker not in merged_returns.columns:
            continue

        event_date = (
            pd.Timestamp(
                row[
                    event_date_column
                ]
            )
            .strftime(
                "%Y-%m-%d"
            )
        )

        event_index = (
            date_to_index.get(
                event_date
            )
        )

        if event_index is None:
            continue

        estimation_start_index = (
            event_index
            + spec.estimation_start
        )

        estimation_end_index = (
            event_index
            + spec.estimation_end
        )

        event_start_index = (
            event_index
            + spec.event_start
        )

        event_end_index = (
            event_index
            + spec.event_end
        )

        if (
            event_start_index < 0
            or event_end_index
            >= len(
                merged_returns
            )
        ):
            continue

        actual_estimation_start = max(
            0,
            estimation_start_index,
        )

        estimation = (
            merged_returns.iloc[
                actual_estimation_start:
                estimation_end_index
                + 1
            ]
        )

        if len(
            estimation
        ) < (
            spec.minimum_estimation_observations
        ):
            continue

        event_window = (
            merged_returns.iloc[
                event_start_index:
                event_end_index
                + 1
            ]
        )

        expected_event_count = (
            spec.event_end
            - spec.event_start
            + 1
        )

        if len(
            event_window
        ) != expected_event_count:
            continue

        stock_est = (
            estimation[
                ticker
            ].to_numpy(
                dtype=float
            )
        )

        market_est = (
            estimation[
                "market_return"
            ].to_numpy(
                dtype=float
            )
        )

        if (
            not np.all(
                np.isfinite(
                    stock_est
                )
            )
            or not np.all(
                np.isfinite(
                    market_est
                )
            )
        ):
            continue

        (
            alpha,
            beta,
            r_squared,
            sigma,
            residuals,
            market_mean,
            market_ss,
        ) = _fit_market_model(
            stock_est,
            market_est,
        )

        stock_event = (
            event_window[
                ticker
            ].to_numpy(
                dtype=float
            )
        )

        market_event = (
            event_window[
                "market_return"
            ].to_numpy(
                dtype=float
            )
        )

        if (
            not np.all(
                np.isfinite(
                    stock_event
                )
            )
            or not np.all(
                np.isfinite(
                    market_event
                )
            )
        ):
            continue

        abnormal = (
            stock_event
            - (
                alpha
                + beta
                * market_event
            )
        )

        if (
            market_ss <= 0.0
            or sigma <= 0.0
        ):
            standardized = (
                abnormal.copy()
            )
        else:
            correction = (
                1.0
                + 1.0
                / len(
                    estimation
                )
                + (
                    (
                        market_event
                        - market_mean
                    )
                    ** 2
                )
                / market_ss
            )

            standardized = (
                abnormal
                / (
                    sigma
                    * np.sqrt(
                        correction
                    )
                )
            )

        records.append(
            EventRecord(
                event_id=int(
                    event_id
                ),
                ticker=ticker,
                event_date=event_date,
                alpha=alpha,
                beta=beta,
                r_squared=r_squared,
                sigma=sigma,
                n_obs=int(
                    len(
                        estimation
                    )
                ),
                market_mean=(
                    market_mean
                ),
                market_ss=(
                    market_ss
                ),
                estimation_dates=(
                    estimation[
                        "date"
                    ]
                    .dt.strftime(
                        "%Y-%m-%d"
                    )
                    .tolist()
                ),
                estimation_residuals=(
                    residuals
                ),
                event_relative_days=(
                    relative_days.copy()
                ),
                event_dates=(
                    event_window[
                        "date"
                    ]
                    .dt.strftime(
                        "%Y-%m-%d"
                    )
                    .tolist()
                ),
                event_market_returns=(
                    market_event
                ),
                event_abnormal_returns=(
                    abnormal
                ),
                event_standardized_returns=(
                    standardized
                ),
            )
        )

    return records


def corrado_rank_statistics(
    records: list[
        EventRecord
    ],
) -> tuple[
    float,
    float,
]:
    if not records:
        raise RuntimeError(
            "Corrado test requires valid events."
        )

    event_length = len(
        records[
            0
        ].event_abnormal_returns
    )

    centered_event_ranks = []

    individual_variances = []

    for record in records:
        combined = np.concatenate(
            (
                record.estimation_residuals,
                record.event_abnormal_returns,
            )
        )

        ranks = rankdata(
            combined,
            method="average",
        )

        total = len(
            combined
        )

        centered = (
            ranks
            / (
                total
                + 1.0
            )
            - 0.5
        )

        centered_event_ranks.append(
            centered[
                -event_length:
            ]
        )

        individual_variances.append(
            (
                total
                - 1.0
            )
            / (
                12.0
                * (
                    total
                    + 1.0
                )
            )
        )

    event_rank_matrix = np.vstack(
        centered_event_ranks
    )

    mean_rank_by_day = np.mean(
        event_rank_matrix,
        axis=0,
    )

    n_events = len(
        records
    )

    standard_error = float(
        math.sqrt(
            sum(
                individual_variances
            )
            / (
                n_events
                * n_events
            )
        )
    )

    if standard_error <= 0.0:
        return (
            0.0,
            0.0,
        )

    day_zero_index = int(
        np.where(
            records[
                0
            ].event_relative_days
            == 0
        )[0][0]
    )

    z_day0 = float(
        mean_rank_by_day[
            day_zero_index
        ]
        / standard_error
    )

    z_full = float(
        np.sum(
            mean_rank_by_day
        )
        / (
            math.sqrt(
                float(
                    event_length
                )
            )
            * standard_error
        )
    )

    return (
        z_full,
        z_day0,
    )


def average_pairwise_residual_correlation(
    records: list[
        EventRecord
    ],
    *,
    minimum_overlap: int = 10,
) -> float:
    correlations = []

    residual_maps = [
        {
            date: float(
                residual
            )
            for date, residual
            in zip(
                record.estimation_dates,
                record.estimation_residuals,
            )
        }
        for record
        in records
    ]

    for first_index in range(
        len(
            records
        )
    ):
        for second_index in range(
            first_index
            + 1,
            len(
                records
            ),
        ):
            first_map = (
                residual_maps[
                    first_index
                ]
            )
            second_map = (
                residual_maps[
                    second_index
                ]
            )

            common_dates = sorted(
                set(
                    first_map
                )
                & set(
                    second_map
                )
            )

            if len(
                common_dates
            ) < minimum_overlap:
                continue

            first_values = np.asarray(
                [
                    first_map[
                        date
                    ]
                    for date
                    in common_dates
                ],
                dtype=float,
            )

            second_values = np.asarray(
                [
                    second_map[
                        date
                    ]
                    for date
                    in common_dates
                ],
                dtype=float,
            )

            if (
                np.std(
                    first_values,
                    ddof=1,
                )
                <= 0.0
                or np.std(
                    second_values,
                    ddof=1,
                )
                <= 0.0
            ):
                continue

            correlation = float(
                np.corrcoef(
                    first_values,
                    second_values,
                )[
                    0,
                    1
                ]
            )

            if math.isfinite(
                correlation
            ):
                correlations.append(
                    correlation
                )

    if not correlations:
        return 0.0

    return float(
        np.mean(
            correlations
        )
    )


def _cross_sectional_t(
    values: np.ndarray,
) -> float:
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(
        values
    ) < 2:
        return 0.0

    standard_deviation = float(
        np.std(
            values,
            ddof=1,
        )
    )

    if standard_deviation <= 0.0:
        return 0.0

    return float(
        math.sqrt(
            float(
                len(
                    values
                )
            )
        )
        * float(
            np.mean(
                values
            )
        )
        / standard_deviation
    )


def kolari_pynnonen_statistics(
    records: list[
        EventRecord
    ],
) -> tuple[
    float,
    float,
    float,
]:
    if not records:
        raise RuntimeError(
            "KP test requires valid events."
        )

    standardized = np.vstack(
        [
            record.event_standardized_returns
            for record
            in records
        ]
    )

    relative_days = (
        records[
            0
        ].event_relative_days
    )

    day_zero_index = int(
        np.where(
            relative_days
            == 0
        )[0][0]
    )

    bmp_day0 = (
        _cross_sectional_t(
            standardized[
                :,
                day_zero_index
            ]
        )
    )

    cumulative_standardized = (
        np.sum(
            standardized,
            axis=1,
        )
    )

    bmp_full = (
        _cross_sectional_t(
            cumulative_standardized
        )
    )

    rho_bar = (
        average_pairwise_residual_correlation(
            records
        )
    )

    n_events = len(
        records
    )

    denominator = (
        1.0
        + (
            n_events
            - 1
        )
        * rho_bar
    )

    numerator = (
        1.0
        - rho_bar
    )

    if (
        denominator <= 0.0
        or numerator < 0.0
    ):
        adjustment = 1.0
    else:
        adjustment = float(
            math.sqrt(
                numerator
                / denominator
            )
        )

    return (
        float(
            rho_bar
        ),
        float(
            bmp_full
            * adjustment
        ),
        float(
            bmp_day0
            * adjustment
        ),
    )
