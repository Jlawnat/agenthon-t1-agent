from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from agent.offline_common import cap_floor as extracted


def test_black_caplet_formula() -> None:
    forward = 0.055
    strike = 0.05
    volatility = 0.20
    fixing_time = 1.0
    discount_factor = 0.95
    accrual = 0.25
    notional = 1_000_000.0

    root_t = math.sqrt(fixing_time)

    d1 = (
        math.log(forward / strike)
        + 0.5 * volatility**2 * fixing_time
    ) / (
        volatility * root_t
    )

    d2 = d1 - volatility * root_t

    expected = (
        notional
        * accrual
        * discount_factor
        * (
            forward * norm.cdf(d1)
            - strike * norm.cdf(d2)
        )
    )

    actual = extracted.black_caplet(
        forward_rate=forward,
        strike=strike,
        volatility=volatility,
        fixing_time=fixing_time,
        discount_factor=discount_factor,
        accrual=accrual,
        notional=notional,
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_black_floorlet_formula() -> None:
    forward = 0.045
    strike = 0.05
    volatility = 0.20
    fixing_time = 1.0
    discount_factor = 0.95
    accrual = 0.25
    notional = 1_000_000.0

    root_t = math.sqrt(fixing_time)

    d1 = (
        math.log(forward / strike)
        + 0.5 * volatility**2 * fixing_time
    ) / (
        volatility * root_t
    )

    d2 = d1 - volatility * root_t

    expected = (
        notional
        * accrual
        * discount_factor
        * (
            strike * norm.cdf(-d2)
            - forward * norm.cdf(-d1)
        )
    )

    actual = extracted.black_floorlet(
        forward_rate=forward,
        strike=strike,
        volatility=volatility,
        fixing_time=fixing_time,
        discount_factor=discount_factor,
        accrual=accrual,
        notional=notional,
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_immediate_fixing_uses_intrinsic_value() -> None:
    cap = extracted.black_caplet(
        forward_rate=0.06,
        strike=0.05,
        volatility=0.20,
        fixing_time=0.0,
        discount_factor=0.99,
        accrual=0.25,
        notional=1000.0,
    )

    floor = extracted.black_floorlet(
        forward_rate=0.04,
        strike=0.05,
        volatility=0.20,
        fixing_time=0.0,
        discount_factor=0.99,
        accrual=0.25,
        notional=1000.0,
    )

    expected = (
        1000.0
        * 0.25
        * 0.99
        * 0.01
    )

    assert np.isclose(
        cap,
        expected,
    )
    assert np.isclose(
        floor,
        expected,
    )


def test_zero_volatility_uses_intrinsic_value() -> None:
    actual = extracted.black_caplet(
        forward_rate=0.06,
        strike=0.05,
        volatility=0.0,
        fixing_time=1.0,
        discount_factor=0.95,
        accrual=0.25,
        notional=1000.0,
    )

    expected = (
        1000.0
        * 0.25
        * 0.95
        * 0.01
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_strip_components_match_single_period_functions() -> None:
    forwards = np.array(
        [
            0.045,
            0.050,
            0.055,
            0.060,
        ]
    )

    dfs = np.array(
        [
            0.99,
            0.98,
            0.97,
            0.96,
        ]
    )

    result = extracted.black_cap_floor_strip(
        forwards,
        dfs,
        strike=0.05,
        volatility=0.20,
        accrual=0.25,
        notional=1_000_000.0,
    )

    expected_caplets = np.asarray(
        [
            extracted.black_caplet(
                forward_rate=float(fwd),
                strike=0.05,
                volatility=0.20,
                fixing_time=i * 0.25,
                discount_factor=float(df),
                accrual=0.25,
                notional=1_000_000.0,
            )
            for i, (fwd, df)
            in enumerate(zip(forwards, dfs))
        ]
    )

    expected_floorlets = np.asarray(
        [
            extracted.black_floorlet(
                forward_rate=float(fwd),
                strike=0.05,
                volatility=0.20,
                fixing_time=i * 0.25,
                discount_factor=float(df),
                accrual=0.25,
                notional=1_000_000.0,
            )
            for i, (fwd, df)
            in enumerate(zip(forwards, dfs))
        ]
    )

    assert np.allclose(
        result.caplets,
        expected_caplets,
    )

    assert np.allclose(
        result.floorlets,
        expected_floorlets,
    )


def test_cap_floor_parity() -> None:
    result = extracted.black_cap_floor_strip(
        [
            0.045,
            0.050,
            0.055,
            0.060,
        ],
        [
            0.99,
            0.98,
            0.97,
            0.96,
        ],
        strike=0.05,
        volatility=0.20,
        accrual=0.25,
        notional=1_000_000.0,
    )

    assert abs(
        result.parity_error
    ) < 1e-8

    assert np.isclose(
        result.cap_price
        - result.floor_price,
        result.swap_value,
    )


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="identical shape",
    ):
        extracted.black_cap_floor_strip(
            [0.04, 0.05],
            [0.99],
            strike=0.05,
            volatility=0.20,
            accrual=0.25,
        )


def test_invalid_single_period_inputs_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        extracted.black_caplet(
            forward_rate=-0.01,
            strike=0.05,
            volatility=0.20,
            fixing_time=1.0,
            discount_factor=0.95,
            accrual=0.25,
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
            "Price the interest rate cap and floor "
            "using Black's model."
        ),
        (
            "Calculate each floorlet and verify cap-floor parity."
        ),
    ),
)
def test_specific_cap_floor_language_selects_capability(
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
        "interest-rate-cap-floor"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_generic_caplet_language_does_not_force_cap_floor(
    tmp_path: Path,
) -> None:
    instruction = (
        "Calibrate a short-rate model using caplet prices."
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
        "interest-rate-cap-floor"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
