from __future__ import annotations

import numpy as np
import pytest

from agent import finance_derivatives_mc as mature
from agent.offline_common import cliquet as extracted


def test_forward_start_price_matches_mature() -> None:
    kwargs = {
        "spot": 100.0,
        "rate": 0.04,
        "dividend_yield": 0.01,
        "volatility": 0.22,
        "start": 0.5,
        "end": 1.0,
    }

    actual = extracted.forward_start_atm_call_price(
        **kwargs,
    )

    expected = mature.forward_start_atm_call_price(
        **kwargs,
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_zero_vol_forward_start_matches_mature() -> None:
    kwargs = {
        "spot": 100.0,
        "rate": 0.05,
        "dividend_yield": 0.01,
        "volatility": 0.0,
        "start": 0.25,
        "end": 0.75,
    }

    assert np.isclose(
        extracted.forward_start_atm_call_price(
            **kwargs,
        ),
        mature.forward_start_atm_call_price(
            **kwargs,
        ),
    )


def test_expired_forward_start_is_zero() -> None:
    assert (
        extracted.forward_start_atm_call_price(
            spot=100.0,
            rate=0.04,
            dividend_yield=0.01,
            volatility=0.20,
            start=1.0,
            end=1.0,
        )
        == 0.0
    )


def test_cliquet_prices_match_mature() -> None:
    kwargs = {
        "spot": 100.0,
        "rate": 0.04,
        "dividend_yield": 0.01,
        "volatility": 0.22,
        "maturity": 2.0,
        "resets": 8,
    }

    actual = extracted.cliquet_forward_start_prices(
        **kwargs,
    )

    expected = mature.cliquet_forward_start_prices(
        **kwargs,
    )

    assert np.allclose(
        actual,
        expected,
    )


def test_cliquet_entries_equal_forward_start_pieces() -> None:
    maturity = 1.0
    resets = 4
    dt = maturity / resets

    prices = extracted.cliquet_forward_start_prices(
        spot=100.0,
        rate=0.04,
        dividend_yield=0.01,
        volatility=0.20,
        maturity=maturity,
        resets=resets,
    )

    expected = np.asarray(
        [
            extracted.forward_start_atm_call_price(
                spot=100.0,
                rate=0.04,
                dividend_yield=0.01,
                volatility=0.20,
                start=i * dt,
                end=(i + 1) * dt,
            )
            for i in range(resets)
        ]
    )

    assert np.allclose(
        prices,
        expected,
    )


@pytest.mark.parametrize(
    "maturity,resets",
    (
        (0.0, 4),
        (1.0, 0),
        (-1.0, 4),
    ),
)
def test_invalid_cliquet_inputs_rejected(
    maturity: float,
    resets: int,
) -> None:
    with pytest.raises(
        RuntimeError,
        match="positive",
    ):
        extracted.cliquet_forward_start_prices(
            spot=100.0,
            rate=0.04,
            dividend_yield=0.01,
            volatility=0.20,
            maturity=maturity,
            resets=resets,
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
        "Price the cliquet and report each reset-period component.",
        "Value the forward-start options underlying the contract.",
    ),
)
def test_specific_cliquet_language_selects_capability(
    tmp_path: Path,
    instruction: str,
) -> None:
    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "cliquet-forward-start-pricing"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_generic_forward_language_does_not_select_cliquet(
    tmp_path: Path,
) -> None:
    instruction = (
        "Calculate the one-year forward rate from the yield curve."
    )

    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "cliquet-forward-start-pricing"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
