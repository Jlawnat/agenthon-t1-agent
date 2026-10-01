from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from agent.offline_common import rate_curves as extracted
from agent.offline_dual_curve import (
    act_360,
    act_365_25,
    fixed_leg_annuity as mature_fixed_leg_annuity,
    floating_leg_pv_per_unit as mature_floating_leg_pv,
    generate_schedule,
    log_linear_df,
    model_par_swap_rate as mature_par_swap_rate,
    thirty_360,
)


def test_simple_forward_rate_formula() -> None:
    start_df = 0.99
    end_df = 0.98
    accrual = 0.25

    expected = (
        start_df
        / end_df
        - 1.0
    ) / accrual

    assert np.isclose(
        extracted.simple_forward_rate(
            start_df,
            end_df,
            accrual,
        ),
        expected,
    )


def test_forward_discount_factor() -> None:
    assert np.isclose(
        extracted.forward_discount_factor(
            0.97,
            0.94,
        ),
        0.94 / 0.97,
    )


def test_fixed_leg_annuity_matches_mature() -> None:
    valuation = date(
        2025,
        1,
        1,
    )

    effective = date(
        2025,
        1,
        3,
    )

    maturity = date(
        2027,
        1,
        3,
    )

    curve = [
        (0.25, 0.99),
        (0.50, 0.98),
        (1.00, 0.96),
        (2.00, 0.92),
        (3.00, 0.88),
    ]

    schedule = generate_schedule(
        effective,
        maturity,
        6,
    )

    accruals = []
    dfs = []

    previous = effective

    for payment_date in schedule:
        accruals.append(
            thirty_360(
                previous,
                payment_date,
            )
        )

        dfs.append(
            log_linear_df(
                curve,
                act_365_25(
                    valuation,
                    payment_date,
                ),
            )
        )

        previous = payment_date

    expected = mature_fixed_leg_annuity(
        effective,
        maturity,
        curve,
        valuation,
        frequency_months=6,
    )

    actual = extracted.fixed_leg_annuity(
        accruals,
        dfs,
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_floating_leg_pv_matches_mature() -> None:
    valuation = date(
        2025,
        1,
        1,
    )

    effective = date(
        2025,
        1,
        3,
    )

    maturity = date(
        2026,
        1,
        3,
    )

    ois_curve = [
        (0.25, 0.99),
        (0.50, 0.98),
        (1.00, 0.96),
        (2.00, 0.92),
    ]

    projection_curve = [
        (0.25, 0.988),
        (0.50, 0.975),
        (1.00, 0.948),
        (2.00, 0.900),
    ]

    schedule = generate_schedule(
        effective,
        maturity,
        3,
    )

    forwards = []
    accruals = []
    dfs = []

    previous = effective

    for payment_date in schedule:
        accrual = act_360(
            previous,
            payment_date,
        )

        start_time = act_365_25(
            valuation,
            previous,
        )

        end_time = act_365_25(
            valuation,
            payment_date,
        )

        start_df = log_linear_df(
            projection_curve,
            start_time,
        )

        end_df = log_linear_df(
            projection_curve,
            end_time,
        )

        forwards.append(
            extracted.simple_forward_rate(
                start_df,
                end_df,
                accrual,
            )
        )

        accruals.append(
            accrual
        )

        dfs.append(
            log_linear_df(
                ois_curve,
                end_time,
            )
        )

        previous = payment_date

    expected = mature_floating_leg_pv(
        effective,
        maturity,
        ois_curve,
        projection_curve,
        valuation,
        frequency_months=3,
    )

    actual = (
        extracted.floating_leg_pv_per_unit(
            forwards,
            accruals,
            dfs,
        )
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_par_swap_rate_matches_mature() -> None:
    valuation = date(
        2025,
        1,
        1,
    )

    effective = date(
        2025,
        1,
        3,
    )

    maturity = date(
        2027,
        1,
        3,
    )

    ois_curve = [
        (0.25, 0.99),
        (0.50, 0.98),
        (1.00, 0.96),
        (2.00, 0.92),
        (3.00, 0.88),
    ]

    projection_curve = [
        (0.25, 0.988),
        (0.50, 0.975),
        (1.00, 0.948),
        (2.00, 0.900),
        (3.00, 0.850),
    ]

    fixed_dates = generate_schedule(
        effective,
        maturity,
        6,
    )

    fixed_accruals = []
    fixed_dfs = []

    previous = effective

    for payment_date in fixed_dates:
        fixed_accruals.append(
            thirty_360(
                previous,
                payment_date,
            )
        )

        fixed_dfs.append(
            log_linear_df(
                ois_curve,
                act_365_25(
                    valuation,
                    payment_date,
                ),
            )
        )

        previous = payment_date

    float_dates = generate_schedule(
        effective,
        maturity,
        3,
    )

    float_forwards = []
    float_accruals = []
    float_dfs = []

    previous = effective

    for payment_date in float_dates:
        accrual = act_360(
            previous,
            payment_date,
        )

        start_time = act_365_25(
            valuation,
            previous,
        )

        end_time = act_365_25(
            valuation,
            payment_date,
        )

        start_df = log_linear_df(
            projection_curve,
            start_time,
        )

        end_df = log_linear_df(
            projection_curve,
            end_time,
        )

        float_forwards.append(
            extracted.simple_forward_rate(
                start_df,
                end_df,
                accrual,
            )
        )

        float_accruals.append(
            accrual
        )

        float_dfs.append(
            log_linear_df(
                ois_curve,
                end_time,
            )
        )

        previous = payment_date

    expected = mature_par_swap_rate(
        effective,
        maturity,
        ois_curve,
        projection_curve,
        valuation,
    )

    actual = extracted.par_swap_rate(
        fixed_accruals=fixed_accruals,
        fixed_discount_factors=fixed_dfs,
        floating_forward_rates=float_forwards,
        floating_accruals=float_accruals,
        floating_discount_factors=float_dfs,
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        extracted.fixed_leg_annuity(
            [0.5, 0.5],
            [0.99],
        )


def test_invalid_accrual_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="accrual",
    ):
        extracted.simple_forward_rate(
            1.0,
            0.99,
            0.0,
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
            "Bootstrap the projection curve and value the swap "
            "using a dual-curve framework."
        ),
        (
            "Roll the valuation date and use forward discount factors "
            "from the frozen curves."
        ),
    ),
)
def test_specific_dual_curve_language_selects_capability(
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
        "dual-curve-rate-mechanics"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_generic_forward_rate_language_does_not_select_capability(
    tmp_path: Path,
) -> None:
    instruction = (
        "Calculate a one-year forward rate from the yield curve."
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
        "dual-curve-rate-mechanics"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
