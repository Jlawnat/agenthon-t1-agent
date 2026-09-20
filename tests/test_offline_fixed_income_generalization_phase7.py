from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agent.offline_common.fixed_income import (
    bootstrap_par_curve,
    reprice_par_bond,
)
from agent.offline_fixed_income import (
    FixedIncomeCurveSkill,
)


def test_bootstrap_reprices_par_bonds() -> None:
    maturities = [
        1.0,
        2.0,
        3.0,
        5.0,
    ]

    par_rates = [
        0.04,
        0.043,
        0.046,
        0.05,
    ]

    curve = bootstrap_par_curve(
        maturities,
        par_rates,
        coupon_frequency=2,
    )

    assert len(
        curve.zero_rates
    ) == len(
        maturities
    )

    assert all(
        np.isfinite(
            curve.zero_rates
        )
    )

    assert all(
        0.0
        < discount_factor
        <= 1.0
        for discount_factor
        in curve.discount_factors
    )

    for maturity, par_rate in zip(
        maturities,
        par_rates,
    ):
        price = reprice_par_bond(
            maturity=maturity,
            par_rate=par_rate,
            coupon_frequency=2,
            curve=curve,
        )

        assert np.isclose(
            price,
            1.0,
            atol=1e-10,
        )


def _write_task(
    root: Path,
) -> Path:
    task = root / "task"
    data_dir = (
        task
        / "environment"
        / "data"
    )
    data_dir.mkdir(
        parents=True
    )

    (
        task
        / "instruction.md"
    ).write_text(
        """
Bootstrap a zero-coupon yield curve from par coupon rates.
Use discount factor interpolation through zero rates and
calculate each forward rate. Write zero_rates,
discount_factors, and forward_rates to results.json.
""".strip(),
        encoding="utf-8",
    )

    (
        data_dir
        / "curve_data.json"
    ).write_text(
        json.dumps(
            {
                "maturities": [
                    1,
                    2,
                    3,
                    5,
                ],
                "par_rates": [
                    0.04,
                    0.043,
                    0.046,
                    0.05,
                ],
                "coupon_freq": 2,
            }
        ),
        encoding="utf-8",
    )

    return task


def test_generic_fixed_income_skill(
    tmp_path: Path,
) -> None:
    task = _write_task(
        tmp_path
    )
    out = tmp_path / "out"

    instruction = (
        task
        / "instruction.md"
    ).read_text(
        encoding="utf-8"
    )

    skill = FixedIncomeCurveSkill()

    assert skill.matches(
        instruction=instruction,
        task_dir=task,
    )

    skill.solve(
        instruction=instruction,
        task_dir=task,
        out_dir=out,
        seed=42,
    )

    result = json.loads(
        (
            out
            / "results.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert set(
        result
    ) == {
        "zero_rates",
        "discount_factors",
        "forward_rates",
    }

    assert set(
        result[
            "zero_rates"
        ]
    ) == {
        "1",
        "2",
        "3",
        "5",
    }

    for maturity in (
        "1",
        "2",
        "3",
        "5",
    ):
        assert (
            result[
                "discount_factors"
            ][
                maturity
            ]
            > 0.0
        )
