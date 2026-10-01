from __future__ import annotations

import numpy as np
import pytest

from agent import offline_asian_options as mature
from agent.offline_common import asian_options as extracted


def test_monitoring_times_match_mature() -> None:
    actual = extracted.monitoring_times(
        1.0,
        12,
    )

    expected = mature.monitoring_times(
        1.0,
        12,
    )

    assert np.allclose(
        actual,
        expected,
    )


def test_geometric_asian_call_matches_mature() -> None:
    kwargs = {
        "S0": 100.0,
        "K": 105.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.20,
        "n_monitoring": 12,
    }

    assert np.isclose(
        extracted.geometric_asian_call(
            **kwargs,
        ),
        mature.geometric_asian_call(
            **kwargs,
        ),
    )


def test_arithmetic_moments_match_mature() -> None:
    kwargs = {
        "S0": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.20,
        "n_monitoring": 12,
    }

    actual = extracted.arithmetic_moments(
        **kwargs,
    )

    expected = mature.arithmetic_moments(
        **kwargs,
    )

    assert np.allclose(
        actual,
        expected,
    )


def test_levy_asian_call_matches_mature() -> None:
    kwargs = {
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.25,
        "n_monitoring": 12,
    }

    assert np.isclose(
        extracted.levy_asian_call(
            **kwargs,
        ),
        mature.levy_asian_call(
            **kwargs,
        ),
    )


def test_curran_asian_call_matches_mature() -> None:
    kwargs = {
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.25,
        "n_monitoring": 12,
    }

    assert np.isclose(
        extracted.curran_asian_call(
            **kwargs,
        ),
        mature.curran_asian_call(
            **kwargs,
        ),
    )


def test_zero_volatility_prices_match_mature() -> None:
    kwargs = {
        "S0": 100.0,
        "K": 95.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.0,
        "n_monitoring": 12,
    }

    assert np.isclose(
        extracted.geometric_asian_call(
            **kwargs,
        ),
        mature.geometric_asian_call(
            **kwargs,
        ),
    )

    assert np.isclose(
        extracted.levy_asian_call(
            **kwargs,
        ),
        mature.levy_asian_call(
            **kwargs,
        ),
    )

    assert np.isclose(
        extracted.curran_asian_call(
            **kwargs,
        ),
        mature.curran_asian_call(
            **kwargs,
        ),
    )


def test_monte_carlo_asian_matches_mature() -> None:
    kwargs = {
        "S0": 100.0,
        "strikes": [
            90.0,
            100.0,
            110.0,
        ],
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.20,
        "n_monitoring": 12,
        "n_paths": 5000,
    }

    actual = extracted.monte_carlo_asian(
        **kwargs,
        rng=np.random.default_rng(
            123456,
        ),
    )

    expected = mature.monte_carlo_asian(
        **kwargs,
        rng=np.random.default_rng(
            123456,
        ),
    )

    assert np.allclose(
        np.asarray(actual),
        np.asarray(expected),
    )


def test_prices_are_nonnegative() -> None:
    common = {
        "S0": 100.0,
        "K": 120.0,
        "T": 1.5,
        "r": 0.04,
        "sigma": 0.30,
        "n_monitoring": 24,
    }

    assert (
        extracted.geometric_asian_call(
            **common
        )
        >= 0.0
    )

    assert (
        extracted.levy_asian_call(
            **common
        )
        >= 0.0
    )

    assert (
        extracted.curran_asian_call(
            **common
        )
        >= 0.0
    )


from pathlib import Path

from agent.capability_bridge import rank_capabilities


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
            "Price the arithmetic Asian call using "
            "the Levy and Curran approximations."
        ),
        (
            "Compare the geometric Asian benchmark "
            "with the Curran approximation."
        ),
    ),
)
def test_specific_asian_language_selects_capability(
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
        "asian-option-approximations"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_generic_monte_carlo_option_does_not_select_asian(
    tmp_path: Path,
) -> None:
    instruction = (
        "Price a European call using Monte Carlo simulation."
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
        "asian-option-approximations"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
