from __future__ import annotations

import numpy as np
import pytest

from agent import finance_fd as mature
from agent.offline_common import finite_difference as extracted


def _spec(
    cls,
    *,
    stock_steps: int = 60,
    time_steps: int = 80,
    dividends=(),
):
    return cls(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        dividend_yield=0.0,
        volatility=0.20,
        maturity=1.0,
        s_max=300.0,
        stock_steps=stock_steps,
        time_steps=time_steps,
        dividends=tuple(dividends),
        psor_omega=1.2,
        psor_tolerance=1e-8,
        psor_max_iterations=10000,
    )


def test_european_call_matches_mature() -> None:
    mature_spec = _spec(
        mature.FiniteDifferenceOptionSpec,
    )
    extracted_spec = _spec(
        extracted.FiniteDifferenceOptionSpec,
    )

    expected = mature.crank_nicolson_option(
        mature_spec,
        option_type="call",
        exercise_type="european",
        return_grid=True,
    )

    actual = extracted.crank_nicolson_option(
        extracted_spec,
        option_type="call",
        exercise_type="european",
        return_grid=True,
    )

    assert np.isclose(
        actual.value,
        expected.value,
    )
    assert np.isclose(
        actual.delta,
        expected.delta,
    )
    assert np.allclose(
        actual.stock_grid,
        expected.stock_grid,
    )
    assert np.allclose(
        actual.time_grid,
        expected.time_grid,
    )
    assert np.allclose(
        actual.value_grid,
        expected.value_grid,
    )
    assert (
        actual.psor_iterations_max
        == expected.psor_iterations_max
    )


def test_american_put_with_cash_dividend_matches_mature() -> None:
    dividends = (
        (
            0.5,
            2.0,
        ),
    )

    mature_spec = _spec(
        mature.FiniteDifferenceOptionSpec,
        dividends=dividends,
    )
    extracted_spec = _spec(
        extracted.FiniteDifferenceOptionSpec,
        dividends=dividends,
    )

    expected = mature.crank_nicolson_option(
        mature_spec,
        option_type="put",
        exercise_type="american",
        return_boundary=True,
    )

    actual = extracted.crank_nicolson_option(
        extracted_spec,
        option_type="put",
        exercise_type="american",
        return_boundary=True,
    )

    assert np.isclose(
        actual.value,
        expected.value,
    )
    assert np.isclose(
        actual.delta,
        expected.delta,
    )
    assert np.allclose(
        actual.exercise_boundary,
        expected.exercise_boundary,
    )
    assert (
        actual.psor_iterations_max
        == expected.psor_iterations_max
    )


def test_dividend_override_matches_mature() -> None:
    mature_spec = _spec(
        mature.FiniteDifferenceOptionSpec,
    )
    extracted_spec = _spec(
        extracted.FiniteDifferenceOptionSpec,
    )

    override = (
        (
            0.25,
            1.5,
        ),
        (
            0.75,
            1.0,
        ),
    )

    expected = mature.crank_nicolson_option(
        mature_spec,
        option_type="call",
        exercise_type="european",
        dividends=override,
    )

    actual = extracted.crank_nicolson_option(
        extracted_spec,
        option_type="call",
        exercise_type="european",
        dividends=override,
    )

    assert np.isclose(
        actual.value,
        expected.value,
    )
    assert np.isclose(
        actual.delta,
        expected.delta,
    )


def test_with_grid_size_matches_mature() -> None:
    mature_spec = _spec(
        mature.FiniteDifferenceOptionSpec,
    )
    extracted_spec = _spec(
        extracted.FiniteDifferenceOptionSpec,
    )

    expected = mature.with_grid_size(
        mature_spec,
        stock_steps=120,
        time_steps=160,
    )

    actual = extracted.with_grid_size(
        extracted_spec,
        stock_steps=120,
        time_steps=160,
    )

    assert actual.spot == expected.spot
    assert actual.strike == expected.strike
    assert actual.rate == expected.rate
    assert actual.dividend_yield == expected.dividend_yield
    assert actual.volatility == expected.volatility
    assert actual.maturity == expected.maturity
    assert actual.s_max == expected.s_max
    assert actual.stock_steps == expected.stock_steps
    assert actual.time_steps == expected.time_steps
    assert actual.dividends == expected.dividends
    assert actual.psor_omega == expected.psor_omega
    assert (
        actual.psor_tolerance
        == expected.psor_tolerance
    )
    assert (
        actual.psor_max_iterations
        == expected.psor_max_iterations
    )


def test_richardson_matches_mature() -> None:
    fine = 10.25
    coarse = 10.10

    assert np.isclose(
        extracted.richardson_second_order(
            fine,
            coarse,
        ),
        mature.richardson_second_order(
            fine,
            coarse,
        ),
    )


def test_invalid_exercise_type_rejected() -> None:
    spec = _spec(
        extracted.FiniteDifferenceOptionSpec,
    )

    with pytest.raises(
        RuntimeError,
        match="exercise_type",
    ):
        extracted.crank_nicolson_option(
            spec,
            option_type="put",
            exercise_type="bermudan",
        )


def test_grid_too_small_rejected() -> None:
    spec = _spec(
        extracted.FiniteDifferenceOptionSpec,
        stock_steps=2,
        time_steps=1,
    )

    with pytest.raises(
        RuntimeError,
        match="grid is too small",
    ):
        extracted.crank_nicolson_option(
            spec,
            option_type="call",
            exercise_type="european",
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
            "Price the American option using a Crank-Nicolson "
            "scheme with PSOR."
        ),
        (
            "Apply Richardson extrapolation to the two grid estimates."
        ),
    ),
)
def test_specific_fd_language_selects_capability(
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
        "finite-difference-options"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_generic_black_scholes_option_does_not_select_fd(
    tmp_path: Path,
) -> None:
    instruction = (
        "Price a European call with Black-Scholes "
        "and report delta and gamma."
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
        "finite-difference-options"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
