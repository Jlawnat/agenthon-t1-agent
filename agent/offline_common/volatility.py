from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


TRADING_DAYS = 252.0


@dataclass(frozen=True)
class OhlcVarianceEstimate:
    close_to_close: float
    parkinson: float
    garman_klass: float
    rogers_satchell: float
    yang_zhang: float


def validate_ohlc(
    frame: pd.DataFrame,
) -> bool:
    by_lower = {
        str(column).lower(): str(column)
        for column
        in frame.columns
    }

    required = {
        "open",
        "high",
        "low",
        "close",
    }

    if not required.issubset(
        by_lower
    ):
        raise RuntimeError(
            "OHLC data requires open, high, low, and close columns."
        )

    open_ = pd.to_numeric(
        frame[
            by_lower[
                "open"
            ]
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    high = pd.to_numeric(
        frame[
            by_lower[
                "high"
            ]
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    low = pd.to_numeric(
        frame[
            by_lower[
                "low"
            ]
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    close = pd.to_numeric(
        frame[
            by_lower[
                "close"
            ]
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    finite = (
        np.isfinite(
            open_
        )
        & np.isfinite(
            high
        )
        & np.isfinite(
            low
        )
        & np.isfinite(
            close
        )
    )

    positive = (
        (
            open_
            > 0.0
        )
        & (
            high
            > 0.0
        )
        & (
            low
            > 0.0
        )
        & (
            close
            > 0.0
        )
    )

    ordering = (
        (
            high
            >= np.maximum(
                open_,
                close,
            )
        )
        & (
            low
            <= np.minimum(
                open_,
                close,
            )
        )
    )

    return bool(
        np.all(
            finite
            & positive
            & ordering
        )
    )


def _numeric_ohlc(
    frame: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    by_lower = {
        str(column).lower(): str(column)
        for column
        in frame.columns
    }

    arrays = []

    for name in (
        "open",
        "high",
        "low",
        "close",
    ):
        if name not in by_lower:
            raise RuntimeError(
                f"OHLC data is missing {name}."
            )

        values = pd.to_numeric(
            frame[
                by_lower[
                    name
                ]
            ],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        if (
            not np.all(
                np.isfinite(
                    values
                )
            )
            or np.any(
                values
                <= 0.0
            )
        ):
            raise RuntimeError(
                f"OHLC column {name} contains invalid values."
            )

        arrays.append(
            values
        )

    return tuple(
        arrays
    )


def _sample_variance(
    values: np.ndarray,
) -> float:
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(
        values
    ) < 2:
        raise RuntimeError(
            "At least two observations are required for sample variance."
        )

    return float(
        np.var(
            values,
            ddof=1,
        )
    )


def ohlc_variance_estimators(
    frame: pd.DataFrame,
    *,
    prior_close: float | None = None,
) -> OhlcVarianceEstimate:
    open_, high, low, close = (
        _numeric_ohlc(
            frame
        )
    )

    n = len(
        close
    )

    if n < 2:
        raise RuntimeError(
            "At least two OHLC observations are required."
        )

    if prior_close is None:
        close_returns = np.log(
            close[
                1:
            ]
            / close[
                :-1
            ]
        )

        yz_open = open_[
            1:
        ]
        yz_close = close[
            1:
        ]
        yz_high = high[
            1:
        ]
        yz_low = low[
            1:
        ]
        yz_previous_close = close[
            :-1
        ]
    else:
        prior_close = float(
            prior_close
        )

        if (
            not math.isfinite(
                prior_close
            )
            or prior_close <= 0.0
        ):
            raise RuntimeError(
                "prior_close must be finite and positive."
            )

        close_returns = np.log(
            close
            / np.concatenate(
                (
                    np.asarray(
                        [
                            prior_close
                        ]
                    ),
                    close[
                        :-1
                    ],
                )
            )
        )

        yz_open = open_
        yz_close = close
        yz_high = high
        yz_low = low
        yz_previous_close = np.concatenate(
            (
                np.asarray(
                    [
                        prior_close
                    ]
                ),
                close[
                    :-1
                ],
            )
        )

    cc_variance = (
        _sample_variance(
            close_returns
        )
    )

    log_high_low = np.log(
        high
        / low
    )

    log_close_open = np.log(
        close
        / open_
    )

    parkinson_variance = float(
        np.mean(
            log_high_low
            * log_high_low
        )
        / (
            4.0
            * math.log(
                2.0
            )
        )
    )

    garman_klass_daily = (
        0.5
        * log_high_low
        * log_high_low
        - (
            2.0
            * math.log(
                2.0
            )
            - 1.0
        )
        * log_close_open
        * log_close_open
    )

    garman_klass_variance = float(
        np.mean(
            garman_klass_daily
        )
    )

    rogers_satchell_daily = (
        np.log(
            high
            / open_
        )
        * np.log(
            high
            / close
        )
        + np.log(
            low
            / open_
        )
        * np.log(
            low
            / close
        )
    )

    rogers_satchell_variance = float(
        np.mean(
            rogers_satchell_daily
        )
    )

    overnight_returns = np.log(
        yz_open
        / yz_previous_close
    )

    open_to_close_returns = np.log(
        yz_close
        / yz_open
    )

    yz_rs_daily = (
        np.log(
            yz_high
            / yz_open
        )
        * np.log(
            yz_high
            / yz_close
        )
        + np.log(
            yz_low
            / yz_open
        )
        * np.log(
            yz_low
            / yz_close
        )
    )

    yz_n = len(
        overnight_returns
    )

    if yz_n < 2:
        raise RuntimeError(
            "Yang-Zhang estimator requires at least two prior-close observations."
        )

    k = (
        0.34
        / (
            1.34
            + (
                yz_n
                + 1.0
            )
            / (
                yz_n
                - 1.0
            )
        )
    )

    yang_zhang_variance = float(
        _sample_variance(
            overnight_returns
        )
        + k
        * _sample_variance(
            open_to_close_returns
        )
        + (
            1.0
            - k
        )
        * float(
            np.mean(
                yz_rs_daily
            )
        )
    )

    return OhlcVarianceEstimate(
        close_to_close=max(
            cc_variance,
            0.0,
        ),
        parkinson=max(
            parkinson_variance,
            0.0,
        ),
        garman_klass=max(
            garman_klass_variance,
            0.0,
        ),
        rogers_satchell=max(
            rogers_satchell_variance,
            0.0,
        ),
        yang_zhang=max(
            yang_zhang_variance,
            0.0,
        ),
    )


def annualized_volatility(
    daily_variance: float,
    *,
    trading_days: float = TRADING_DAYS,
) -> float:
    return float(
        math.sqrt(
            max(
                float(
                    daily_variance
                ),
                0.0,
            )
            * float(
                trading_days
            )
        )
    )


def rolling_ohlc_estimators(
    frame: pd.DataFrame,
    *,
    window: int,
) -> list[
    OhlcVarianceEstimate
]:
    if window < 2:
        raise RuntimeError(
            "Rolling OHLC window must be at least two days."
        )

    if len(
        frame
    ) <= window:
        return []

    by_lower = {
        str(column).lower(): str(column)
        for column
        in frame.columns
    }

    close_column = (
        by_lower.get(
            "close"
        )
    )

    if close_column is None:
        raise RuntimeError(
            "OHLC data requires close column."
        )

    close_values = pd.to_numeric(
        frame[
            close_column
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    estimates = []

    # Starting at `end == window` intentionally gives n_days - window
    # windows. Each window contains exactly `window` return periods for
    # estimators that require a prior close.
    for end in range(
        window,
        len(
            frame
        ),
    ):
        start = (
            end
            - window
            + 1
        )

        subset = frame.iloc[
            start:
            end
            + 1
        ]

        prior_close = float(
            close_values[
                start
                - 1
            ]
        )

        estimates.append(
            ohlc_variance_estimators(
                subset,
                prior_close=prior_close,
            )
        )

    return estimates


def estimator_arrays(
    estimates: list[
        OhlcVarianceEstimate
    ],
) -> dict[
    str,
    np.ndarray,
]:
    names = (
        "close_to_close",
        "parkinson",
        "garman_klass",
        "rogers_satchell",
        "yang_zhang",
    )

    return {
        name: np.asarray(
            [
                getattr(
                    estimate,
                    name,
                )
                for estimate
                in estimates
            ],
            dtype=float,
        )
        for name
        in names
    }


def efficiency_ratios(
    variance_series: dict[
        str,
        np.ndarray,
    ],
) -> dict[
    str,
    float,
]:
    baseline = np.asarray(
        variance_series[
            "close_to_close"
        ],
        dtype=float,
    )

    baseline_variance = float(
        np.var(
            baseline,
            ddof=1,
        )
    )

    result = {
        "close_to_close": 1.0
    }

    for name, values in variance_series.items():
        if name == "close_to_close":
            continue

        estimator_variance = float(
            np.var(
                np.asarray(
                    values,
                    dtype=float,
                ),
                ddof=1,
            )
        )

        if estimator_variance <= 0.0:
            result[
                name
            ] = float(
                "inf"
            )
        else:
            result[
                name
            ] = float(
                baseline_variance
                / estimator_variance
            )

    return result
