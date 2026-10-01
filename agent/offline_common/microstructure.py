from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence

import numpy as np
import pandas as pd


def _finite_float(
    value: float,
    *,
    name: str,
) -> float:
    result = float(value)

    if not np.isfinite(result):
        raise ValueError(
            f"{name} must be finite."
        )

    return result


def _validated_sorted_times(
    timestamps: Sequence[int | float],
) -> list[int | float]:
    values = list(timestamps)

    for value in values:
        _finite_float(
            value,
            name="timestamp",
        )

    if any(
        values[index] > values[index + 1]
        for index in range(
            len(values) - 1
        )
    ):
        raise ValueError(
            "timestamps must be sorted ascending."
        )

    return values


def first_index_at_or_after(
    timestamps: Sequence[int | float],
    target: int | float,
) -> int:
    """Index of the first timestamp greater than or equal to target."""
    values = _validated_sorted_times(
        timestamps
    )

    target_value = _finite_float(
        target,
        name="target",
    )

    index = bisect_left(
        values,
        target_value,
    )

    if index >= len(values):
        raise ValueError(
            "no timestamp exists at or after target."
        )

    return int(index)


def half_open_window_indices(
    timestamps: Sequence[int | float],
    *,
    start: int | float,
    end: int | float,
) -> tuple[int, int]:
    """Slice bounds for start <= timestamp < end."""
    values = _validated_sorted_times(
        timestamps
    )

    start_value = _finite_float(
        start,
        name="start",
    )
    end_value = _finite_float(
        end,
        name="end",
    )

    if end_value < start_value:
        raise ValueError(
            "end must be greater than or equal to start."
        )

    return (
        int(
            bisect_left(
                values,
                start_value,
            )
        ),
        int(
            bisect_left(
                values,
                end_value,
            )
        ),
    )


def participation_capped_quantity(
    remaining_quantity: float,
    market_volume: float,
    participation_cap: float,
    max_child_quantity: float,
) -> float:
    """Child quantity bounded by remaining size, participation and child cap."""
    remaining = _finite_float(
        remaining_quantity,
        name="remaining_quantity",
    )
    volume = _finite_float(
        market_volume,
        name="market_volume",
    )
    cap = _finite_float(
        participation_cap,
        name="participation_cap",
    )
    maximum = _finite_float(
        max_child_quantity,
        name="max_child_quantity",
    )

    if (
        remaining < 0.0
        or volume < 0.0
        or cap < 0.0
        or maximum < 0.0
    ):
        raise ValueError(
            "quantities, volume, and participation cap "
            "must be non-negative."
        )

    return float(
        max(
            0.0,
            min(
                remaining,
                cap * volume,
                maximum,
            ),
        )
    )


def exact_synchronized_venue_rows(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    venue_column: str,
    required_venues: Sequence[str],
) -> pd.DataFrame:
    """Keep groups containing exactly one row for each required venue."""
    groups = [
        str(column)
        for column in group_columns
    ]

    venues = [
        str(venue)
        for venue in required_venues
    ]

    if not groups:
        raise ValueError(
            "group_columns must not be empty."
        )

    if not venues:
        raise ValueError(
            "required_venues must not be empty."
        )

    if len(set(venues)) != len(venues):
        raise ValueError(
            "required_venues must be unique."
        )

    needed = set(groups)
    needed.add(
        venue_column
    )

    missing = sorted(
        needed
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "frame is missing required columns: "
            + ", ".join(missing)
        )

    working = frame.copy()

    working[
        venue_column
    ] = (
        working[
            venue_column
        ]
        .astype(str)
    )

    required_sorted = sorted(
        venues
    )

    pieces: list[pd.DataFrame] = []

    for _, group in working.groupby(
        groups,
        sort=True,
        dropna=False,
    ):
        observed = sorted(
            group[
                venue_column
            ].tolist()
        )

        if observed == required_sorted:
            pieces.append(
                group.copy()
            )

    if not pieces:
        return (
            working
            .iloc[0:0]
            .copy()
            .reset_index(
                drop=True
            )
        )

    result = pd.concat(
        pieces,
        ignore_index=True,
    )

    return (
        result
        .sort_values(
            [
                *groups,
                venue_column,
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )
