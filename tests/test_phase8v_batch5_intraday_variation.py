from __future__ import annotations

import math

import numpy as np
import pytest

from agent import offline_intraday_volatility as mature
from agent.offline_common import intraday_variation as extracted


def test_realized_variance_matches_mature() -> None:
    returns = np.array(
        [
            0.001,
            -0.002,
            0.0015,
            0.0007,
            -0.0011,
        ],
        dtype=float,
    )

    assert np.isclose(
        extracted.realized_variance(returns),
        mature.realized_variance(returns),
    )


def test_bandi_russell_matches_mature() -> None:
    returns = np.array(
        [
            0.001,
            -0.002,
            0.0015,
            0.0007,
            -0.0011,
        ],
        dtype=float,
    )

    assert np.isclose(
        extracted.bandi_russell_noise_variance(
            returns
        ),
        mature.bandi_russell_noise_variance(
            returns
        ),
    )


def test_bipower_variation_matches_mature() -> None:
    returns = np.array(
        [
            0.001,
            -0.002,
            0.0015,
            0.0007,
            -0.0011,
        ],
        dtype=float,
    )

    assert np.isclose(
        extracted.bipower_variation(
            returns
        ),
        mature.bipower_variation(
            returns
        ),
    )


def test_additive_noise_correction_matches_formula() -> None:
    rv = 0.0025
    n_returns = 78
    noise = 1.2e-6

    expected = (
        rv
        - 2.0
        * n_returns
        * noise
    )

    actual = (
        extracted.additive_noise_corrected_variance(
            rv,
            n_returns=n_returns,
            noise_variance=noise,
        )
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_periodic_sampling_mask() -> None:
    offsets = np.array(
        [
            0,
            1,
            2,
            5,
            10,
            11,
            15,
        ]
    )

    actual = (
        extracted.periodic_sampling_mask(
            offsets,
            frequency=5,
        )
    )

    expected = np.array(
        [
            True,
            False,
            False,
            True,
            True,
            False,
            True,
        ]
    )

    assert np.array_equal(
        actual,
        expected,
    )


def test_annualized_volatility_matches_formula() -> None:
    variance = 0.0004

    expected = math.sqrt(
        variance * 252.0
    )

    actual = (
        extracted.annualized_volatility_from_variance(
            variance,
            periods_per_year=252.0,
        )
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_negative_corrected_variance_clips_only_for_volatility() -> None:
    corrected = (
        extracted.additive_noise_corrected_variance(
            0.0001,
            n_returns=100,
            noise_variance=1e-6,
        )
    )

    assert corrected < 0.0

    annualized = (
        extracted.annualized_volatility_from_variance(
            corrected,
            periods_per_year=252.0,
            clip_negative=True,
        )
    )

    assert annualized == 0.0


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        extracted.bandi_russell_noise_variance(
            []
        )

    with pytest.raises(ValueError):
        extracted.additive_noise_corrected_variance(
            0.001,
            n_returns=-1,
            noise_variance=1e-6,
        )

    with pytest.raises(ValueError):
        extracted.periodic_sampling_mask(
            [0, 1, 2],
            frequency=0,
        )

    with pytest.raises(ValueError):
        extracted.annualized_volatility_from_variance(
            -0.001,
            periods_per_year=252.0,
            clip_negative=False,
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
            "Estimate Bandi-Russell microstructure noise "
            "and compute bipower variation."
        ),
        (
            "Build the volatility signature and apply "
            "microstructure-noise correction."
        ),
    ),
)
def test_intraday_variation_language_selects_capability(
    tmp_path: Path,
    instruction: str,
) -> None:
    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "intraday-realized-variation"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_generic_realized_volatility_does_not_select_capability(
    tmp_path: Path,
) -> None:
    instruction = (
        "Compute daily realized volatility "
        "from closing prices."
    )

    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "intraday-realized-variation"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
