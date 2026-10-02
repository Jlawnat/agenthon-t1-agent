from __future__ import annotations

import math

import numpy as np
import pytest

from agent.offline_common import first_passage as extracted
from agent.qf_primitives import (
    brownian_running_max_hit_probability,
    brownian_running_min_hit_probability,
)


def test_upper_first_passage_matches_running_max_primitive() -> None:
    mu = 0.06
    sigma = 0.22
    horizon = 0.75
    barrier = math.log(1.10)

    expected = brownian_running_max_hit_probability(
        mu,
        sigma,
        horizon,
        barrier,
    )

    actual = extracted.upper_first_passage_cdf(
        mu,
        sigma,
        horizon,
        barrier,
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_lower_first_passage_matches_running_min_primitive() -> None:
    mu = 0.04
    sigma = 0.25
    horizon = 0.5
    barrier = math.log(0.90)

    expected = brownian_running_min_hit_probability(
        mu,
        sigma,
        horizon,
        barrier,
    )

    actual = extracted.lower_first_passage_cdf(
        mu,
        sigma,
        horizon,
        barrier,
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_zero_time_cdf_is_zero() -> None:
    assert (
        extracted.upper_first_passage_cdf(
            0.05,
            0.20,
            0.0,
            0.10,
        )
        == 0.0
    )

    assert (
        extracted.lower_first_passage_cdf(
            0.05,
            0.20,
            0.0,
            -0.10,
        )
        == 0.0
    )


def test_expected_upper_first_passage_time() -> None:
    mu = 0.08
    barrier = 0.12

    assert np.isclose(
        extracted.expected_upper_first_passage_time(
            mu,
            barrier,
        ),
        barrier / mu,
    )


def test_expected_lower_first_passage_time() -> None:
    mu = -0.05
    barrier = -0.15

    assert np.isclose(
        extracted.expected_lower_first_passage_time(
            mu,
            barrier,
        ),
        barrier / mu,
    )


def test_expected_time_is_infinite_when_drift_points_away() -> None:
    assert math.isinf(
        extracted.expected_upper_first_passage_time(
            0.0,
            0.10,
        )
    )

    assert math.isinf(
        extracted.expected_upper_first_passage_time(
            -0.02,
            0.10,
        )
    )

    assert math.isinf(
        extracted.expected_lower_first_passage_time(
            0.02,
            -0.10,
        )
    )


@pytest.mark.parametrize(
    "fn,args",
    (
        (
            extracted.upper_first_passage_cdf,
            (
                0.05,
                0.0,
                1.0,
                0.10,
            ),
        ),
        (
            extracted.upper_first_passage_cdf,
            (
                0.05,
                0.20,
                -1.0,
                0.10,
            ),
        ),
        (
            extracted.upper_first_passage_cdf,
            (
                0.05,
                0.20,
                1.0,
                -0.10,
            ),
        ),
        (
            extracted.lower_first_passage_cdf,
            (
                0.05,
                0.20,
                1.0,
                0.10,
            ),
        ),
    ),
)
def test_invalid_first_passage_inputs_rejected(
    fn,
    args,
) -> None:
    with pytest.raises(ValueError):
        fn(*args)


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
            "Compute the first passage time distribution "
            "for a drifted Brownian motion."
        ),
        (
            "Evaluate first-passage time probabilities "
            "for upper and lower barriers."
        ),
    ),
)
def test_first_passage_language_selects_capability(
    tmp_path: Path,
    instruction: str,
) -> None:
    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "brownian-first-passage"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_reflection_principle_alone_does_not_select_first_passage(
    tmp_path: Path,
) -> None:
    instruction = (
        "Price a digital barrier option using the reflection principle."
    )

    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "brownian-first-passage"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
