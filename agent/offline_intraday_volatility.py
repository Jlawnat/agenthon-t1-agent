from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


_FREQUENCIES = (
    1,
    2,
    5,
    15,
    30,
)


def _find_intraday_csv(
    task_dir: Path,
) -> Path:
    candidates = []

    for path in sorted(
        task_dir.rglob("*.csv")
    ):
        if "checks" in path.parts:
            continue

        try:
            sample = pd.read_csv(
                path,
                nrows=5,
            )
        except Exception:
            continue

        lower = {
            str(column).lower()
            for column
            in sample.columns
        }

        if {
            "timestamp",
            "mid_price",
        }.issubset(
            lower
        ):
            candidates.append(
                path
            )

    if not candidates:
        raise RuntimeError(
            "No timestamp/mid_price intraday CSV was found."
        )

    return candidates[0]


def clean_intraday_quotes(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    by_lower = {
        str(column).lower(): str(column)
        for column
        in frame.columns
    }

    if not {
        "timestamp",
        "mid_price",
    }.issubset(
        by_lower
    ):
        raise RuntimeError(
            "Intraday data requires timestamp and mid_price columns."
        )

    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                frame[
                    by_lower[
                        "timestamp"
                    ]
                ],
                errors="raise",
            ),
            "mid_price": pd.to_numeric(
                frame[
                    by_lower[
                        "mid_price"
                    ]
                ],
                errors="raise",
            ).astype(float),
        }
    )

    if (
        not np.all(
            np.isfinite(
                data[
                    "mid_price"
                ].to_numpy()
            )
        )
        or (
            data[
                "mid_price"
            ]
            <= 0.0
        ).any()
    ):
        raise RuntimeError(
            "Intraday mid prices must be finite and positive."
        )

    data = (
        data.groupby(
            "timestamp",
            as_index=False,
            sort=True,
        )[
            "mid_price"
        ]
        .median()
    )

    seconds = (
        data[
            "timestamp"
        ].dt.hour
        * 3600
        + data[
            "timestamp"
        ].dt.minute
        * 60
        + data[
            "timestamp"
        ].dt.second
    )

    open_seconds = (
        9
        * 3600
        + 30
        * 60
    )
    close_seconds = (
        16
        * 3600
    )

    data = data.loc[
        (
            seconds
            >= open_seconds
        )
        & (
            seconds
            <= close_seconds
        )
    ].copy()

    data = (
        data.sort_values(
            "timestamp",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    data[
        "date"
    ] = (
        data[
            "timestamp"
        ].dt.date
    )

    return data


def _sample_frequency(
    day: pd.DataFrame,
    frequency: int,
) -> pd.DataFrame:
    minute_of_day = (
        day[
            "timestamp"
        ].dt.hour
        * 60
        + day[
            "timestamp"
        ].dt.minute
    )

    offset = (
        minute_of_day
        - (
            9
            * 60
            + 30
        )
    )

    return day.loc[
        (
            offset
            % int(
                frequency
            )
        )
        == 0
    ].copy()


def _log_returns(
    sampled: pd.DataFrame,
) -> np.ndarray:
    prices = sampled[
        "mid_price"
    ].to_numpy(
        dtype=float
    )

    if len(
        prices
    ) < 2:
        return np.asarray(
            [],
            dtype=float,
        )

    return np.log(
        prices[
            1:
        ]
        / prices[
            :-1
        ]
    )


def realized_variance(
    returns: np.ndarray,
) -> float:
    returns = np.asarray(
        returns,
        dtype=float,
    )

    return float(
        np.sum(
            returns
            * returns
        )
    )


def bandi_russell_noise_variance(
    returns_1min: np.ndarray,
) -> float:
    returns_1min = np.asarray(
        returns_1min,
        dtype=float,
    )

    n = len(
        returns_1min
    )

    if n <= 0:
        raise RuntimeError(
            "Noise-variance estimation requires at least one return."
        )

    return float(
        realized_variance(
            returns_1min
        )
        / (
            2.0
            * n
        )
    )


def bipower_variation(
    returns: np.ndarray,
) -> float:
    returns = np.asarray(
        returns,
        dtype=float,
    )

    if len(
        returns
    ) < 2:
        return 0.0

    return float(
        (
            math.pi
            / 2.0
        )
        * np.sum(
            np.abs(
                returns[
                    1:
                ]
            )
            * np.abs(
                returns[
                    :-1
                ]
            )
        )
    )


def analyze_intraday_day(
    day: pd.DataFrame,
) -> dict[str, object]:
    result: dict[
        str,
        object,
    ] = {}

    return_cache: dict[
        int,
        np.ndarray,
    ] = {}

    for frequency in _FREQUENCIES:
        sampled = (
            _sample_frequency(
                day,
                frequency,
            )
        )

        returns = _log_returns(
            sampled
        )

        return_cache[
            frequency
        ] = returns

        result[
            f"RV_{frequency}min"
        ] = realized_variance(
            returns
        )

        result[
            f"n_{frequency}min"
        ] = int(
            len(
                returns
            )
        )

    result[
        "noise_var_est"
    ] = bandi_russell_noise_variance(
        return_cache[
            1
        ]
    )

    result[
        "BV_5min"
    ] = bipower_variation(
        return_cache[
            5
        ]
    )

    return result


@dataclass(frozen=True)
class IntradayRealizedVolatilitySkill:
    name: str = "intraday-realized-volatility-domain"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        lowered = (
            instruction.lower()
        )

        return (
            (
                "realized volatility" in lowered
                or "realised volatility" in lowered
                or "realized variance" in lowered
            )
            and (
                "minute-level" in lowered
                or "minute level" in lowered
                or "intraday" in lowered
            )
            and (
                "microstructure" in lowered
                or "bandi-russell" in lowered
                or "bandi russell" in lowered
            )
            and (
                "bipower" in lowered
            )
        )

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        del instruction, seed

        path = _find_intraday_csv(
            task_dir
        )

        raw = pd.read_csv(
            path
        )

        data = (
            clean_intraday_quotes(
                raw
            )
        )

        day_results = []

        for date_value, day in data.groupby(
            "date",
            sort=True,
        ):
            analyzed = (
                analyze_intraday_day(
                    day
                )
            )

            analyzed[
                "date"
            ] = str(
                date_value
            )

            day_results.append(
                analyzed
            )

        if not day_results:
            raise RuntimeError(
                "No regular-hours intraday days were found."
            )

        grand_mean_noise = float(
            np.mean(
                [
                    float(
                        row[
                            "noise_var_est"
                        ]
                    )
                    for row
                    in day_results
                ]
            )
        )

        corrected_values = []

        for row in day_results:
            corrected = float(
                float(
                    row[
                        "RV_5min"
                    ]
                )
                - 2.0
                * int(
                    row[
                        "n_5min"
                    ]
                )
                * grand_mean_noise
            )

            row[
                "RV_5min_corrected"
            ] = corrected

            row[
                "RV_5min_ann_vol"
            ] = float(
                math.sqrt(
                    max(
                        corrected,
                        0.0,
                    )
                    * 252.0
                )
            )

            corrected_values.append(
                corrected
            )

        mean_corrected = float(
            np.mean(
                corrected_values
            )
        )

        summary = {
            "num_days": int(
                len(
                    day_results
                )
            ),
            "grand_mean_noise_var": (
                grand_mean_noise
            ),
            "mean_RV_5min_corrected": (
                mean_corrected
            ),
            "grand_ann_vol": float(
                math.sqrt(
                    max(
                        mean_corrected,
                        0.0,
                    )
                    * 252.0
                )
            ),
        }

        ordered_days = []

        for row in day_results:
            ordered_days.append(
                {
                    "date": row[
                        "date"
                    ],
                    "RV_1min": float(
                        row[
                            "RV_1min"
                        ]
                    ),
                    "n_1min": int(
                        row[
                            "n_1min"
                        ]
                    ),
                    "RV_2min": float(
                        row[
                            "RV_2min"
                        ]
                    ),
                    "n_2min": int(
                        row[
                            "n_2min"
                        ]
                    ),
                    "RV_5min": float(
                        row[
                            "RV_5min"
                        ]
                    ),
                    "n_5min": int(
                        row[
                            "n_5min"
                        ]
                    ),
                    "RV_15min": float(
                        row[
                            "RV_15min"
                        ]
                    ),
                    "n_15min": int(
                        row[
                            "n_15min"
                        ]
                    ),
                    "RV_30min": float(
                        row[
                            "RV_30min"
                        ]
                    ),
                    "n_30min": int(
                        row[
                            "n_30min"
                        ]
                    ),
                    "noise_var_est": float(
                        row[
                            "noise_var_est"
                        ]
                    ),
                    "BV_5min": float(
                        row[
                            "BV_5min"
                        ]
                    ),
                    "RV_5min_corrected": float(
                        row[
                            "RV_5min_corrected"
                        ]
                    ),
                    "RV_5min_ann_vol": float(
                        row[
                            "RV_5min_ann_vol"
                        ]
                    ),
                }
            )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            out_dir
            / "results.json"
        ).write_text(
            json.dumps(
                {
                    "days": (
                        ordered_days
                    ),
                    "summary": (
                        summary
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
