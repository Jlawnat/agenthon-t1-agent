from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MomentumSpec:
    lookback_start: int
    skip_recent: int
    valid_start: int
    long_count: int
    short_count: int
    periods_per_year: int


def clean_price_panel(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
) -> tuple[pd.DataFrame, list[str]]:
    """Clean a cross-sectional price panel deterministically.

    The operation order is intentionally explicit because many finance
    benchmark tasks make cleaning order part of the contract.
    """
    if date_column not in frame.columns:
        raise RuntimeError(
            f"Price panel requires a {date_column!r} column."
        )

    asset_columns = [
        str(column)
        for column in frame.columns
        if str(column) != date_column
    ]

    if len(asset_columns) < 2:
        raise RuntimeError(
            "Price panel requires at least two asset columns."
        )

    cleaned = frame.copy()
    cleaned[date_column] = pd.to_datetime(
        cleaned[date_column],
        errors="raise",
    )

    cleaned = cleaned.sort_values(
        date_column,
        kind="stable",
    )

    cleaned = cleaned.drop_duplicates(
        subset=[date_column],
        keep="last",
    )

    for column in asset_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    cleaned[asset_columns] = (
        cleaned[
            asset_columns
        ].ffill()
    )

    cleaned = cleaned.dropna(
        subset=asset_columns,
    ).reset_index(
        drop=True
    )

    return (
        cleaned,
        asset_columns,
    )


def simple_returns_from_prices(
    prices: pd.DataFrame,
    *,
    asset_columns: list[str],
    date_column: str = "date",
) -> pd.DataFrame:
    values = (
        prices[
            asset_columns
        ]
        .astype(float)
    )

    returns = (
        values
        / values.shift(1)
        - 1.0
    )

    result = returns.iloc[
        1:
    ].reset_index(
        drop=True
    )

    result.insert(
        0,
        date_column,
        prices[
            date_column
        ].iloc[
            1:
        ].reset_index(
            drop=True
        ),
    )

    return result


def parse_cross_sectional_momentum_spec(
    instruction: str,
) -> MomentumSpec:
    lowered = instruction.lower()

    window_match = re.search(
        r"t\s*-\s*(\d+)\s+through\s+t\s*-\s*(\d+)",
        lowered,
    )

    if window_match:
        lookback_start = int(
            window_match.group(1)
        )
        skip_recent = int(
            window_match.group(2)
        )
    else:
        # Common 12-1 momentum convention: use the prior
        # 11 observations and skip the most recent one.
        lookback_start = 12
        skip_recent = 2

    valid_match = re.search(
        r"t\s*>=\s*(\d+)",
        lowered,
    )

    valid_start = (
        int(
            valid_match.group(1)
        )
        if valid_match
        else lookback_start
    )

    top_match = re.search(
        r"top\s+(\d+)",
        lowered,
    )
    bottom_match = re.search(
        r"bottom\s+(\d+)",
        lowered,
    )

    if not top_match or not bottom_match:
        raise RuntimeError(
            "Could not infer long/short counts from the instruction."
        )

    periods_per_year = (
        12
        if "monthly" in lowered
        else 252
    )

    return MomentumSpec(
        lookback_start=lookback_start,
        skip_recent=skip_recent,
        valid_start=valid_start,
        long_count=int(
            top_match.group(1)
        ),
        short_count=int(
            bottom_match.group(1)
        ),
        periods_per_year=periods_per_year,
    )


def cross_sectional_momentum_path(
    returns: pd.DataFrame,
    *,
    asset_columns: list[str],
    spec: MomentumSpec,
    date_column: str = "date",
) -> pd.DataFrame:
    """Build a generic long-short cross-sectional momentum path."""
    original_order = {
        asset: index
        for index, asset
        in enumerate(
            asset_columns
        )
    }

    rows: list[
        dict[str, object]
    ] = []

    wealth = 1.0

    for t in range(
        spec.valid_start,
        len(
            returns
        ),
    ):
        start = (
            t
            - spec.lookback_start
        )

        # If skip_recent=2, the final included index is t-2,
        # so the exclusive slice endpoint is t-1.
        stop = (
            t
            - spec.skip_recent
            + 1
        )

        if start < 0 or stop <= start:
            continue

        signal = (
            returns[
                asset_columns
            ]
            .iloc[
                start:stop
            ]
            .sum(
                axis=0
            )
        )

        long_names = sorted(
            asset_columns,
            key=lambda asset: (
                -float(
                    signal[
                        asset
                    ]
                ),
                original_order[
                    asset
                ],
            ),
        )[
            :spec.long_count
        ]

        short_names = sorted(
            asset_columns,
            key=lambda asset: (
                float(
                    signal[
                        asset
                    ]
                ),
                original_order[
                    asset
                ],
            ),
        )[
            :spec.short_count
        ]

        realized = returns.iloc[
            t
        ]

        long_return = float(
            np.mean(
                [
                    float(
                        realized[
                            asset
                        ]
                    )
                    for asset
                    in long_names
                ]
            )
        )

        short_return = float(
            np.mean(
                [
                    float(
                        realized[
                            asset
                        ]
                    )
                    for asset
                    in short_names
                ]
            )
        )

        signal_return = (
            long_return
            - short_return
        )

        wealth *= (
            1.0
            + signal_return
        )

        rows.append(
            {
                "date": (
                    pd.Timestamp(
                        realized[
                            date_column
                        ]
                    )
                    .strftime(
                        "%Y-%m-%d"
                    )
                ),
                "signal_return": float(
                    signal_return
                ),
                "cumulative_return": float(
                    wealth
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "signal_return",
            "cumulative_return",
        ],
    )


def summarize_return_path(
    path: pd.DataFrame,
    *,
    periods_per_year: int,
) -> dict[str, float]:
    if path.empty:
        raise RuntimeError(
            "Portfolio path contains no realized returns."
        )

    returns = path[
        "signal_return"
    ].to_numpy(
        dtype=float
    )

    total_return = float(
        np.prod(
            1.0
            + returns
        )
        - 1.0
    )

    annualized_return = float(
        (
            1.0
            + total_return
        )
        ** (
            float(
                periods_per_year
            )
            / len(
                returns
            )
        )
        - 1.0
    )

    annualized_vol = float(
        np.std(
            returns,
            ddof=1,
        )
        * math.sqrt(
            float(
                periods_per_year
            )
        )
    )

    sharpe = (
        float(
            annualized_return
            / annualized_vol
        )
        if annualized_vol > 0.0
        else 0.0
    )

    wealth = path[
        "cumulative_return"
    ].to_numpy(
        dtype=float
    )

    running_peak = (
        np.maximum.accumulate(
            wealth
        )
    )

    drawdown = (
        running_peak
        - wealth
    ) / running_peak

    max_drawdown = float(
        np.max(
            drawdown
        )
    )

    return {
        "total_return": (
            total_return
        ),
        "annualized_return": (
            annualized_return
        ),
        "annualized_vol": (
            annualized_vol
        ),
        "sharpe": (
            sharpe
        ),
        "max_drawdown": (
            max_drawdown
        ),
    }


def find_price_panel_csv(
    task_dir: Path,
) -> Path:
    candidates: list[
        tuple[
            int,
            Path,
        ]
    ] = []

    for path in sorted(
        task_dir.rglob(
            "*.csv"
        )
    ):
        if "checks" in path.parts:
            continue

        try:
            sample = pd.read_csv(
                path,
                nrows=4,
            )
        except Exception:
            continue

        lower_columns = [
            str(column).lower()
            for column
            in sample.columns
        ]

        if "date" not in lower_columns:
            continue

        non_date_count = (
            len(
                lower_columns
            )
            - 1
        )

        if non_date_count >= 2:
            candidates.append(
                (
                    non_date_count,
                    path,
                )
            )

    if not candidates:
        raise RuntimeError(
            "No cross-sectional price-panel CSV was found."
        )

    candidates.sort(
        key=lambda item: (
            -item[0],
            str(
                item[1]
            ),
        )
    )

    return candidates[
        0
    ][
        1
    ]
