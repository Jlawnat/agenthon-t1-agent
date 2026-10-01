from __future__ import annotations

import numpy as np
import pytest

from agent import offline_lookback_options as mature
from agent.offline_common import lookback_options as extracted


def test_floating_lookback_call_matches_mature() -> None:
    kwargs = {
        "spot": 100.0,
        "running_min": 90.0,
        "maturity": 1.0,
        "risk_free_rate": 0.05,
        "dividend_yield": 0.01,
        "volatility": 0.20,
    }

    assert np.isclose(
        extracted.floating_lookback_call(**kwargs),
        mature.floating_lookback_call(**kwargs),
    )


def test_floating_lookback_put_matches_mature() -> None:
    kwargs = {
        "spot": 100.0,
        "running_max": 115.0,
        "maturity": 1.0,
        "risk_free_rate": 0.05,
        "dividend_yield": 0.01,
        "volatility": 0.20,
    }

    assert np.isclose(
        extracted.floating_lookback_put(**kwargs),
        mature.floating_lookback_put(**kwargs),
    )


@pytest.mark.parametrize(
    "strike,running_max",
    (
        (110.0, 105.0),
        (95.0, 110.0),
    ),
)
def test_fixed_strike_lookback_call_matches_mature(
    strike: float,
    running_max: float,
) -> None:
    kwargs = {
        "spot": 100.0,
        "running_max": running_max,
        "strike": strike,
        "maturity": 1.0,
        "risk_free_rate": 0.05,
        "dividend_yield": 0.01,
        "volatility": 0.20,
    }

    assert np.isclose(
        extracted.fixed_strike_lookback_call(**kwargs),
        mature.fixed_strike_lookback_call(**kwargs),
    )


def test_prices_are_nonnegative() -> None:
    assert (
        extracted.floating_lookback_call(
            spot=100.0,
            running_min=90.0,
            maturity=0.5,
            risk_free_rate=0.05,
            dividend_yield=0.01,
            volatility=0.25,
        )
        >= 0.0
    )

    assert (
        extracted.floating_lookback_put(
            spot=100.0,
            running_max=110.0,
            maturity=0.5,
            risk_free_rate=0.05,
            dividend_yield=0.01,
            volatility=0.25,
        )
        >= 0.0
    )

    assert (
        extracted.fixed_strike_lookback_call(
            spot=100.0,
            running_max=105.0,
            strike=100.0,
            maturity=0.5,
            risk_free_rate=0.05,
            dividend_yield=0.01,
            volatility=0.25,
        )
        >= 0.0
    )


@pytest.mark.parametrize(
    "fn,kwargs",
    (
        (
            extracted.floating_lookback_call,
            {
                "spot": 100.0,
                "running_min": 90.0,
                "maturity": 1.0,
                "risk_free_rate": 0.05,
                "dividend_yield": 0.05,
                "volatility": 0.20,
            },
        ),
        (
            extracted.floating_lookback_put,
            {
                "spot": 100.0,
                "running_max": 110.0,
                "maturity": 1.0,
                "risk_free_rate": 0.05,
                "dividend_yield": 0.05,
                "volatility": 0.20,
            },
        ),
        (
            extracted.fixed_strike_lookback_call,
            {
                "spot": 100.0,
                "running_max": 110.0,
                "strike": 100.0,
                "maturity": 1.0,
                "risk_free_rate": 0.05,
                "dividend_yield": 0.05,
                "volatility": 0.20,
            },
        ),
    ),
)
def test_zero_cost_of_carry_limit_rejected(
    fn,
    kwargs,
) -> None:
    with pytest.raises(
        RuntimeError,
        match="Zero cost-of-carry",
    ):
        fn(**kwargs)


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
            "Price fixed-strike lookback calls and "
            "floating-strike lookback calls and puts."
        ),
        (
            "Value the lookback options using closed-form formulas."
        ),
    ),
)
def test_specific_lookback_language_selects_capability(
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
        "lookback-option-pricing"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_generic_running_maximum_does_not_select_lookback(
    tmp_path: Path,
) -> None:
    instruction = (
        "Calculate the running maximum of the portfolio loss series."
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
        "lookback-option-pricing"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
