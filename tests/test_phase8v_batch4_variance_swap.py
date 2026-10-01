from __future__ import annotations

import math

import numpy as np
import pytest

from agent import offline_variance_swap as mature
from agent.offline_common import variance_swap as extracted


def test_trapezoidal_widths_match_mature() -> None:
    strikes = np.array(
        [
            80.0,
            90.0,
            100.0,
            115.0,
            130.0,
        ]
    )

    actual = extracted.trapezoidal_strike_widths(
        strikes
    )

    expected = mature.trapezoidal_strike_widths(
        strikes
    )

    assert np.allclose(
        actual,
        expected,
    )


def test_nonuniform_trapezoidal_widths() -> None:
    strikes = np.array(
        [
            90.0,
            95.0,
            105.0,
            120.0,
        ]
    )

    expected = np.array(
        [
            5.0,
            7.5,
            12.5,
            15.0,
        ]
    )

    assert np.allclose(
        extracted.trapezoidal_strike_widths(
            strikes
        ),
        expected,
    )


def test_fair_variance_formula() -> None:
    strikes = np.array(
        [
            80.0,
            90.0,
            100.0,
            110.0,
            125.0,
        ]
    )

    prices = np.array(
        [
            0.70,
            1.20,
            2.30,
            1.40,
            0.55,
        ]
    )

    maturity = 0.5
    rate = 0.04

    widths = (
        extracted.trapezoidal_strike_widths(
            strikes
        )
    )

    expected = (
        2.0
        * math.exp(
            rate * maturity
        )
        / maturity
        * np.sum(
            widths
            * prices
            / strikes**2
        )
    )

    actual = (
        extracted.variance_swap_fair_variance(
            strikes,
            prices,
            risk_free_rate=rate,
            maturity=maturity,
        )
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_forward_interpolation() -> None:
    actual = extracted.interpolate_at_forward(
        100.0,
        110.0,
        0.20,
        0.30,
        104.0,
    )

    assert np.isclose(
        actual,
        0.24,
    )


def test_variance_swap_long_pnl() -> None:
    actual = extracted.variance_swap_pnl(
        0.30,
        0.25**2,
        variance_notional=10_000.0,
    )

    expected = (
        10_000.0
        * (
            0.30**2
            - 0.25**2
        )
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_fair_variance_is_nonnegative() -> None:
    value = extracted.variance_swap_fair_variance(
        [
            90.0,
            100.0,
            115.0,
        ],
        [
            1.0,
            2.0,
            0.8,
        ],
        risk_free_rate=0.03,
        maturity=0.75,
    )

    assert value >= 0.0


def test_unsorted_strikes_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        extracted.trapezoidal_strike_widths(
            [
                100.0,
                90.0,
                110.0,
            ]
        )


def test_invalid_replication_shapes_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="equal-length",
    ):
        extracted.variance_swap_fair_variance(
            [
                90.0,
                100.0,
            ],
            [
                1.0,
            ],
            risk_free_rate=0.03,
            maturity=1.0,
        )


from pathlib import Path

from agent.capability_bridge import rank_capabilities


def _task(
    tmp_path: Path,
    instruction: str,
) -> Path:
    task = tmp_path / "task"
    (task / "environment" / "data").mkdir(parents=True)
    (task / "instruction.md").write_text(
        instruction,
        encoding="utf-8",
    )
    return task


@pytest.mark.parametrize(
    "instruction",
    (
        "Compute the fair variance swap strike using log-contract replication.",
        "Use trapezoidal strike weights on the non-uniform strike grid.",
    ),
)
def test_specific_variance_swap_language_selects_capability(
    tmp_path: Path,
    instruction: str,
) -> None:
    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "variance-swap-replication"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_generic_variance_language_does_not_select_variance_swap(
    tmp_path: Path,
) -> None:
    instruction = (
        "Estimate the variance of daily portfolio returns."
    )

    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "variance-swap-replication"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
