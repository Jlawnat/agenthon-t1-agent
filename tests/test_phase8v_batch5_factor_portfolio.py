from __future__ import annotations

import math

import numpy as np
import pytest

from agent.offline_common import factor_portfolio as extracted


def _inputs():
    components = np.array(
        [
            [0.50, 0.40, -0.30, -0.20],
            [0.10, -0.60, 0.50, 0.20],
        ],
        dtype=float,
    )

    weights = np.array(
        [0.40, 0.30, 0.20, 0.10],
        dtype=float,
    )

    return components, weights


def test_factor_exposures_match_matrix_product() -> None:
    components, weights = _inputs()

    actual = extracted.factor_exposures(
        components,
        weights,
    )

    expected = components @ weights

    assert np.allclose(
        actual,
        expected,
    )


def test_factor_neutral_hedge_matches_mature_formula() -> None:
    components, weights = _inputs()

    expected_a = np.vstack(
        [
            components,
            np.ones(
                (
                    1,
                    len(weights),
                )
            ),
        ]
    )

    expected_b = np.append(
        -(components @ weights),
        0.0,
    )

    expected_hedge, *_ = np.linalg.lstsq(
        expected_a,
        expected_b,
        rcond=None,
    )

    actual = (
        extracted.minimum_norm_factor_neutral_hedge(
            components,
            weights,
        )
    )

    assert np.allclose(
        actual.target_exposures,
        components @ weights,
    )

    assert np.allclose(
        actual.hedge_weights,
        expected_hedge,
    )

    assert np.allclose(
        actual.neutral_weights,
        weights + expected_hedge,
    )

    assert np.allclose(
        actual.neutral_exposures,
        components
        @ (
            weights
            + expected_hedge
        ),
    )


def test_factor_neutral_constraints() -> None:
    components, weights = _inputs()

    result = (
        extracted.minimum_norm_factor_neutral_hedge(
            components,
            weights,
        )
    )

    assert np.allclose(
        components
        @ result.neutral_weights,
        np.zeros(
            components.shape[0]
        ),
        atol=1e-12,
    )

    assert np.isclose(
        np.sum(
            result.hedge_weights
        ),
        0.0,
        atol=1e-12,
    )


def test_annualized_sharpe_matches_mature_formula() -> None:
    returns = np.array(
        [
            0.01,
            0.02,
            -0.01,
            0.015,
            0.005,
            0.012,
        ],
        dtype=float,
    )

    expected = float(
        np.mean(returns)
        / np.std(
            returns,
            ddof=1,
        )
        * math.sqrt(12.0)
    )

    actual = (
        extracted.annualized_sharpe_ratio(
            returns,
            periods_per_year=12.0,
        )
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_zero_volatility_sharpe_is_zero() -> None:
    actual = (
        extracted.annualized_sharpe_ratio(
            [
                0.01,
                0.01,
                0.01,
                0.01,
            ],
            periods_per_year=12.0,
        )
    )

    assert actual == 0.0


def test_residual_variance_r_squared_matches_definition() -> None:
    benchmark = np.array(
        [
            0.02,
            -0.01,
            0.015,
            -0.005,
            0.012,
        ],
        dtype=float,
    )

    residual = np.array(
        [
            0.005,
            -0.003,
            0.004,
            -0.002,
            0.003,
        ],
        dtype=float,
    )

    expected = (
        1.0
        - np.var(residual)
        / np.var(benchmark)
    )

    actual = (
        extracted.residual_variance_r_squared(
            residual,
            benchmark,
        )
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_invalid_dimensions_rejected() -> None:
    with pytest.raises(ValueError):
        extracted.factor_exposures(
            np.eye(3),
            [0.5, 0.5],
        )

    with pytest.raises(ValueError):
        extracted.minimum_norm_factor_neutral_hedge(
            np.eye(3),
            [0.5, 0.5],
        )

    with pytest.raises(ValueError):
        extracted.residual_variance_r_squared(
            [0.1, 0.2],
            [0.1, 0.2, 0.3],
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
            "Perform PCA Factor Portfolio Construction "
            "and build a factor-neutral portfolio."
        ),
        (
            "Use PCA loadings to construct a factor neutral "
            "portfolio and report target_factor_exposures."
        ),
    ),
)
def test_pca_factor_language_selects_capability(
    tmp_path: Path,
    instruction: str,
) -> None:
    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "pca-factor-neutral-portfolio"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_generic_pca_language_does_not_select_portfolio_capability(
    tmp_path: Path,
) -> None:
    instruction = (
        "Run PCA on a matrix and inspect "
        "the principal components."
    )

    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "pca-factor-neutral-portfolio"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
