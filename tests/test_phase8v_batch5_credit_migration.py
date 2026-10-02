from __future__ import annotations

import numpy as np
import pytest

from agent import offline_credit_migration as mature
from agent.offline_common import credit_migration as extracted


def _cohorts() -> dict[int, np.ndarray]:
    first = np.array(
        [
            [80, 15, 3, 1, 0, 0, 0, 1],
            [5, 75, 14, 3, 1, 0, 0, 2],
            [1, 6, 72, 15, 3, 1, 0, 2],
            [0, 1, 7, 70, 14, 4, 1, 3],
            [0, 0, 1, 6, 68, 17, 4, 4],
            [0, 0, 0, 1, 8, 65, 18, 8],
            [0, 0, 0, 0, 2, 10, 68, 20],
            [0, 0, 0, 0, 0, 0, 0, 100],
        ],
        dtype=float,
    )

    second = np.array(
        [
            [78, 16, 4, 1, 0, 0, 0, 1],
            [4, 74, 15, 4, 1, 0, 0, 2],
            [1, 5, 70, 17, 4, 1, 0, 2],
            [0, 1, 6, 68, 16, 5, 1, 3],
            [0, 0, 1, 5, 65, 19, 5, 5],
            [0, 0, 0, 1, 7, 61, 20, 11],
            [0, 0, 0, 0, 2, 8, 64, 26],
            [0, 0, 0, 0, 0, 0, 0, 100],
        ],
        dtype=float,
    )

    return {
        2023: first,
        2024: second,
    }


def test_pooled_transition_matrix_matches_mature() -> None:
    cohorts = _cohorts()

    expected_p, expected_counts = (
        mature._average_matrix(cohorts)
    )

    actual_p, actual_counts = (
        extracted.pooled_transition_matrix(
            list(cohorts.values()),
            absorbing_index=7,
        )
    )

    assert np.allclose(
        actual_counts,
        expected_counts,
    )
    assert np.allclose(
        actual_p,
        expected_p,
    )


def test_zero_count_row_becomes_identity() -> None:
    counts = np.zeros(
        (3, 3),
        dtype=float,
    )
    counts[0] = [8, 2, 0]
    counts[2] = [0, 0, 10]

    transition, _ = (
        extracted.pooled_transition_matrix(
            [counts],
            absorbing_index=2,
        )
    )

    assert np.allclose(
        transition[1],
        [0.0, 1.0, 0.0],
    )
    assert np.allclose(
        transition[2],
        [0.0, 0.0, 1.0],
    )


def test_cumulative_default_probabilities_match_mature() -> None:
    cohorts = _cohorts()
    transition, _ = mature._average_matrix(
        cohorts
    )

    horizons = [1, 2, 3, 5]

    expected = mature._cumulative_default(
        transition,
        horizons,
    )

    actual = (
        extracted.cumulative_target_probabilities(
            transition,
            target_index=7,
            horizons=horizons,
            source_indices=range(7),
        )
    )

    for horizon in horizons:
        assert np.allclose(
            actual[horizon],
            expected[horizon],
        )


def test_transition_homogeneity_matches_mature() -> None:
    cohorts = _cohorts()
    years = sorted(cohorts)

    expected = mature._markov_test(
        years,
        cohorts,
        0.05,
    )[3]

    actual = (
        extracted.transition_homogeneity_test(
            list(
                cohorts.values()
            ),
            origin_index=3,
            min_destination_total=5.0,
            alpha=0.05,
        )
    )

    assert np.isclose(
        actual.chi2_stat,
        expected["chi2_stat"],
    )
    assert (
        actual.degrees_of_freedom
        == expected["df"]
    )
    assert np.isclose(
        actual.p_value,
        expected["p_value"],
    )
    assert (
        actual.reject_null
        == expected["reject_H0"]
    )


def test_sparse_homogeneity_returns_neutral_result() -> None:
    first = np.zeros((3, 3))
    second = np.zeros((3, 3))

    first[0, 0] = 3
    second[0, 0] = 4

    result = (
        extracted.transition_homogeneity_test(
            [first, second],
            origin_index=0,
            min_destination_total=5.0,
        )
    )

    assert result.chi2_stat == 0.0
    assert result.degrees_of_freedom == 0
    assert result.p_value == 1.0
    assert result.reject_null is False


def test_generator_matches_mature() -> None:
    cohorts = _cohorts()

    transition, _ = mature._average_matrix(
        cohorts
    )

    expected = mature._generator(
        transition
    )

    actual = (
        extracted.continuous_time_generator(
            transition,
            absorbing_index=7,
        )
    )

    assert np.allclose(
        actual,
        expected,
    )


def test_generator_invariants() -> None:
    cohorts = _cohorts()

    transition, _ = (
        extracted.pooled_transition_matrix(
            list(cohorts.values()),
            absorbing_index=7,
        )
    )

    generator = (
        extracted.continuous_time_generator(
            transition,
            absorbing_index=7,
        )
    )

    off_diagonal = generator[
        ~np.eye(
            generator.shape[0],
            dtype=bool,
        )
    ]

    assert np.all(
        off_diagonal >= -1e-12
    )
    assert np.allclose(
        generator.sum(axis=1),
        0.0,
        atol=1e-12,
    )
    assert np.allclose(
        generator[7],
        0.0,
    )


def test_invalid_cohort_shapes_rejected() -> None:
    with pytest.raises(ValueError):
        extracted.pooled_transition_matrix(
            [
                np.ones((3, 3)),
                np.ones((4, 4)),
            ]
        )


def test_invalid_horizon_rejected() -> None:
    transition = np.eye(3)

    with pytest.raises(
        ValueError,
        match="positive integers",
    ):
        extracted.cumulative_target_probabilities(
            transition,
            target_index=2,
            horizons=[0],
        )


from pathlib import Path

from agent.capability_bridge import rank_capabilities


def _task(
    tmp_path: Path,
    instruction: str,
) -> Path:
    task = tmp_path / "task"
    (task / "environment" / "data").mkdir(
        parents=True,
    )
    (task / "instruction.md").write_text(
        instruction,
        encoding="utf-8",
    )
    return task


@pytest.mark.parametrize(
    "instruction",
    (
        (
            "Analyze credit rating migration and build "
            "the cohort transition matrix."
        ),
        (
            "Compute cumulative default probabilities and "
            "a continuous-time generator matrix."
        ),
    ),
)
def test_credit_migration_language_selects_capability(
    tmp_path: Path,
    instruction: str,
) -> None:
    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "credit-migration-transition-matrix"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_generic_markov_language_does_not_select_credit_migration(
    tmp_path: Path,
) -> None:
    instruction = (
        "Estimate a generic Markov chain transition probability."
    )

    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "credit-migration-transition-matrix"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
