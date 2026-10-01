from __future__ import annotations

import numpy as np
import pytest

from agent import finance_derivatives_mc as mature
from agent.offline_common import monte_carlo_greeks as extracted


def _normals() -> np.ndarray:
    rng = np.random.default_rng(123456)
    return rng.standard_normal(
        (
            2048,
            12,
        )
    )


def _kwargs(
    *,
    asian: bool = False,
) -> dict:
    return {
        "spot": 100.0,
        "strike": 105.0,
        "rate": 0.04,
        "dividend_yield": 0.01,
        "volatility": 0.22,
        "maturity": 1.0,
        "option_type": "call",
        "asian": asian,
    }


def _assert_estimate_equal(
    actual,
    expected,
) -> None:
    if expected is None:
        assert actual is None
        return

    assert np.isclose(
        actual.value,
        expected.value,
    )
    assert np.isclose(
        actual.se,
        expected.se,
    )


def _assert_greek_dict_equal(
    actual,
    expected,
) -> None:
    assert actual.keys() == expected.keys()

    for greek in actual:
        _assert_estimate_equal(
            actual[greek],
            expected[greek],
        )


def test_mc_estimate_matches_mature() -> None:
    samples = np.array(
        [
            1.2,
            2.4,
            0.8,
            3.1,
            1.7,
        ],
        dtype=float,
    )

    expected = mature.mc_estimate(
        samples
    )
    actual = extracted.mc_estimate(
        samples
    )

    _assert_estimate_equal(
        actual,
        expected,
    )


def test_gbm_paths_match_mature() -> None:
    normals = _normals()

    kwargs = {
        "spot": 100.0,
        "rate": 0.04,
        "dividend_yield": 0.01,
        "volatility": 0.22,
        "maturity": 1.0,
    }

    expected = mature.gbm_paths_from_normals(
        normals,
        **kwargs,
    )

    actual = extracted.gbm_paths_from_normals(
        normals,
        **kwargs,
    )

    assert np.allclose(
        actual,
        expected,
    )


@pytest.mark.parametrize(
    "asian",
    (
        False,
        True,
    ),
)
def test_finite_difference_greeks_match_mature(
    asian: bool,
) -> None:
    normals = _normals()
    kwargs = _kwargs(
        asian=asian,
    )

    expected = mature.finite_difference_greeks(
        normals,
        **kwargs,
    )

    actual = extracted.finite_difference_greeks(
        normals,
        **kwargs,
    )

    _assert_greek_dict_equal(
        actual,
        expected,
    )


@pytest.mark.parametrize(
    "asian",
    (
        False,
        True,
    ),
)
def test_pathwise_greeks_match_mature(
    asian: bool,
) -> None:
    normals = _normals()
    kwargs = _kwargs(
        asian=asian,
    )

    expected = mature.pathwise_greeks(
        normals,
        **kwargs,
    )

    actual = extracted.pathwise_greeks(
        normals,
        **kwargs,
    )

    _assert_greek_dict_equal(
        actual,
        expected,
    )


@pytest.mark.parametrize(
    "asian",
    (
        False,
        True,
    ),
)
def test_likelihood_ratio_greeks_match_mature(
    asian: bool,
) -> None:
    normals = _normals()
    kwargs = _kwargs(
        asian=asian,
    )

    expected = mature.likelihood_ratio_greeks(
        normals,
        **kwargs,
    )

    actual = extracted.likelihood_ratio_greeks(
        normals,
        **kwargs,
    )

    _assert_greek_dict_equal(
        actual,
        expected,
    )


def test_invalid_gbm_shape_rejected() -> None:
    normals = np.ones(
        (
            1,
            5,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="N x M normal matrix",
    ):
        extracted.gbm_paths_from_normals(
            normals,
            spot=100.0,
            rate=0.04,
            dividend_yield=0.01,
            volatility=0.20,
            maturity=1.0,
        )


from pathlib import Path

from agent.capability_bridge import (
    rank_capabilities,
)


def _task(
    tmp_path: Path,
    instruction: str,
) -> Path:
    task = tmp_path / "task"

    (
        task
        / "environment"
        / "data"
    ).mkdir(
        parents=True,
    )

    (
        task
        / "instruction.md"
    ).write_text(
        instruction,
        encoding="utf-8",
    )

    return task


@pytest.mark.parametrize(
    "instruction",
    (
        (
            "Estimate the option Greeks using pathwise estimators "
            "and Monte Carlo Greeks."
        ),
        (
            "Compute delta and gamma using the likelihood ratio method."
        ),
    ),
)
def test_specific_mc_greek_language_selects_capability(
    tmp_path: Path,
    instruction: str,
) -> None:
    task = _task(
        tmp_path,
        instruction,
    )

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "monte-carlo-greeks"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_generic_monte_carlo_does_not_select_mc_greeks(
    tmp_path: Path,
) -> None:
    instruction = (
        "Price an Asian option using Monte Carlo simulation "
        "and report the estimated option value."
    )

    task = _task(
        tmp_path,
        instruction,
    )

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "monte-carlo-greeks"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
