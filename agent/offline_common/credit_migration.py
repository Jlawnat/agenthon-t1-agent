from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import linalg
from scipy.stats import chi2


@dataclass(frozen=True)
class TransitionHomogeneityResult:
    chi2_stat: float
    degrees_of_freedom: int
    p_value: float
    reject_null: bool


def pooled_transition_matrix(
    cohort_counts: Sequence[np.ndarray],
    *,
    absorbing_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool cohort count matrices and row-normalize the resulting transition matrix."""
    matrices = [
        np.asarray(
            counts,
            dtype=float,
        )
        for counts in cohort_counts
    ]

    if not matrices:
        raise ValueError(
            "at least one cohort count matrix is required."
        )

    shape = matrices[0].shape

    if (
        len(shape) != 2
        or shape[0] != shape[1]
    ):
        raise ValueError(
            "cohort count matrices must be square."
        )

    for matrix in matrices:
        if matrix.shape != shape:
            raise ValueError(
                "all cohort count matrices must have the same shape."
            )

        if (
            not np.all(np.isfinite(matrix))
            or np.any(matrix < 0.0)
        ):
            raise ValueError(
                "cohort counts must be finite and non-negative."
            )

    total_counts = np.sum(
        matrices,
        axis=0,
    )

    transition = np.zeros_like(
        total_counts,
        dtype=float,
    )

    for row in range(shape[0]):
        row_total = float(
            total_counts[row].sum()
        )

        if row_total > 0.0:
            transition[row] = (
                total_counts[row]
                / row_total
            )
        else:
            transition[row, row] = 1.0

    if absorbing_index is not None:
        index = int(absorbing_index)

        if not 0 <= index < shape[0]:
            raise ValueError(
                "absorbing_index is out of range."
            )

        transition[index] = 0.0
        transition[index, index] = 1.0

    return (
        transition,
        total_counts,
    )


def cumulative_target_probabilities(
    transition_matrix: np.ndarray,
    *,
    target_index: int,
    horizons: Sequence[int],
    source_indices: Sequence[int] | None = None,
) -> dict[int, np.ndarray]:
    """Return probabilities of reaching a target state at integer horizons."""
    matrix = np.asarray(
        transition_matrix,
        dtype=float,
    )

    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
    ):
        raise ValueError(
            "transition_matrix must be square."
        )

    n_states = matrix.shape[0]
    target = int(target_index)

    if not 0 <= target < n_states:
        raise ValueError(
            "target_index is out of range."
        )

    if source_indices is None:
        sources = np.arange(
            n_states,
            dtype=int,
        )
    else:
        sources = np.asarray(
            source_indices,
            dtype=int,
        )

        if (
            sources.ndim != 1
            or np.any(sources < 0)
            or np.any(sources >= n_states)
        ):
            raise ValueError(
                "source_indices contain an invalid state index."
            )

    result: dict[int, np.ndarray] = {}

    for horizon in horizons:
        h = int(horizon)

        if h < 1 or h != horizon:
            raise ValueError(
                "horizons must contain positive integers."
            )

        powered = np.linalg.matrix_power(
            matrix,
            h,
        )

        result[h] = powered[
            sources,
            target,
        ].copy()

    return result


def transition_homogeneity_test(
    cohort_counts: Sequence[np.ndarray],
    *,
    origin_index: int,
    min_destination_total: float = 5.0,
    alpha: float = 0.05,
) -> TransitionHomogeneityResult:
    """Chi-square homogeneity test for one origin state's transition distribution."""
    matrices = [
        np.asarray(
            counts,
            dtype=float,
        )
        for counts in cohort_counts
    ]

    if len(matrices) < 2:
        raise ValueError(
            "at least two cohorts are required."
        )

    shape = matrices[0].shape

    if (
        len(shape) != 2
        or shape[0] != shape[1]
    ):
        raise ValueError(
            "cohort count matrices must be square."
        )

    for matrix in matrices:
        if (
            matrix.shape != shape
            or not np.all(np.isfinite(matrix))
            or np.any(matrix < 0.0)
        ):
            raise ValueError(
                "cohort count matrices must share a valid non-negative shape."
            )

    origin = int(origin_index)

    if not 0 <= origin < shape[0]:
        raise ValueError(
            "origin_index is out of range."
        )

    threshold = float(
        min_destination_total
    )

    significance = float(alpha)

    if threshold < 0.0:
        raise ValueError(
            "min_destination_total must be non-negative."
        )

    if not 0.0 < significance < 1.0:
        raise ValueError(
            "alpha must lie strictly between 0 and 1."
        )

    table = np.stack(
        [
            matrix[origin]
            for matrix in matrices
        ],
        axis=0,
    )

    valid_columns = (
        table.sum(axis=0)
        >= threshold
    )

    filtered = table[
        :,
        valid_columns,
    ]

    valid_rows = (
        filtered.sum(axis=1)
        > 0.0
    )

    filtered = filtered[
        valid_rows
    ]

    n_rows, n_columns = filtered.shape

    if (
        n_rows < 2
        or n_columns < 2
    ):
        statistic = 0.0
        degrees_of_freedom = 0
        p_value = 1.0
    else:
        row_totals = filtered.sum(
            axis=1
        )

        column_totals = filtered.sum(
            axis=0
        )

        grand_total = float(
            filtered.sum()
        )

        expected = np.outer(
            row_totals,
            column_totals,
        ) / grand_total

        mask = expected > 0.0

        statistic = float(
            np.sum(
                (
                    filtered[mask]
                    - expected[mask]
                )
                ** 2
                / expected[mask]
            )
        )

        degrees_of_freedom = int(
            (
                n_rows - 1
            )
            * (
                n_columns - 1
            )
        )

        p_value = (
            float(
                1.0
                - chi2.cdf(
                    statistic,
                    degrees_of_freedom,
                )
            )
            if degrees_of_freedom > 0
            else 1.0
        )

    return TransitionHomogeneityResult(
        chi2_stat=statistic,
        degrees_of_freedom=degrees_of_freedom,
        p_value=p_value,
        reject_null=bool(
            p_value < significance
        ),
    )


def continuous_time_generator(
    transition_matrix: np.ndarray,
    *,
    absorbing_index: int | None = None,
) -> np.ndarray:
    """Construct a repaired continuous-time generator from a transition matrix."""
    matrix = np.asarray(
        transition_matrix,
        dtype=float,
    )

    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
    ):
        raise ValueError(
            "transition_matrix must be square."
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "transition_matrix must contain finite values."
        )

    generator = linalg.logm(
        matrix
    ).real

    n_states = matrix.shape[0]

    for i in range(n_states):
        for j in range(n_states):
            if (
                i != j
                and generator[i, j] < 0.0
            ):
                generator[i, j] = 0.0

    for i in range(n_states):
        generator[i, i] = 0.0
        generator[i, i] = -float(
            generator[i].sum()
        )

    if absorbing_index is not None:
        index = int(absorbing_index)

        if not 0 <= index < n_states:
            raise ValueError(
                "absorbing_index is out of range."
            )

        generator[index] = 0.0

    return generator


__all__ = (
    "TransitionHomogeneityResult",
    "pooled_transition_matrix",
    "cumulative_target_probabilities",
    "transition_homogeneity_test",
    "continuous_time_generator",
)
