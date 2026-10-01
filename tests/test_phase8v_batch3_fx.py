from __future__ import annotations

import numpy as np
import pytest

from agent.finance_xccy import (
    interp_log as mature_interp_log,
)
from agent.offline_common import fx as extracted


def test_covered_interest_forward_formula() -> None:
    spot = 1.08
    rb = 0.04
    rq = 0.05
    tb = 90.0 / 360.0
    tq = 90.0 / 365.0

    expected = (
        spot
        * (
            1.0
            + rq
            * tq
        )
        / (
            1.0
            + rb
            * tb
        )
    )

    actual = (
        extracted.covered_interest_forward(
            spot,
            base_rate=rb,
            quote_rate=rq,
            base_year_fraction=tb,
            quote_year_fraction=tq,
        )
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_implied_rate_recovers_input_rate() -> None:
    spot = 1.25
    rb = 0.035
    rq = 0.0475
    tb = 0.25
    tq = 0.25

    forward = (
        extracted.covered_interest_forward(
            spot,
            base_rate=rb,
            quote_rate=rq,
            base_year_fraction=tb,
            quote_year_fraction=tq,
        )
    )

    implied = (
        extracted.implied_quote_rate_from_forward(
            spot,
            forward,
            base_rate=rb,
            base_year_fraction=tb,
            quote_year_fraction=tq,
        )
    )

    assert np.isclose(
        implied,
        rq,
    )


def test_invert_bid_ask_flips_sides() -> None:
    bid, ask = (
        extracted.invert_bid_ask(
            150.0,
            150.2,
        )
    )

    assert np.isclose(
        bid,
        1.0 / 150.2,
    )

    assert np.isclose(
        ask,
        1.0 / 150.0,
    )

    assert bid <= ask


def test_synthetic_cross_bid_ask() -> None:
    bid, ask, mid = (
        extracted.synthetic_cross_bid_ask(
            base_usd_bid=1.2500,
            base_usd_ask=1.2510,
            quote_usd_bid=1.0800,
            quote_usd_ask=1.0810,
        )
    )

    assert np.isclose(
        bid,
        1.2500 / 1.0810,
    )

    assert np.isclose(
        ask,
        1.2510 / 1.0800,
    )

    assert np.isclose(
        mid,
        (
            bid
            + ask
        )
        / 2.0,
    )


def test_forward_points() -> None:
    assert np.isclose(
        extracted.forward_points(
            1.1000,
            1.1025,
            pip_multiplier=10000.0,
        ),
        25.0,
    )


def test_zero_basis_when_forward_equals_cip() -> None:
    assert np.isclose(
        extracted.continuous_basis_bps(
            1.25,
            1.25,
            0.5,
        ),
        0.0,
    )


def test_basis_round_trip() -> None:
    cip = 1.30
    basis = -18.5
    tau = 2.0

    quoted = (
        extracted.forward_from_continuous_basis(
            cip,
            basis,
            tau,
        )
    )

    recovered = (
        extracted.continuous_basis_bps(
            quoted,
            cip,
            tau,
        )
    )

    assert np.isclose(
        recovered,
        basis,
    )


def test_log_linear_forward_matches_mature_interpolation() -> None:
    spot = 1.20

    times = [
        0.25,
        0.5,
        1.0,
    ]

    forwards = [
        1.21,
        1.22,
        1.24,
    ]

    target = 0.375

    expected = mature_interp_log(
        target,
        [
            0.0,
            *times,
        ],
        [
            spot,
            *forwards,
        ],
    )

    actual = (
        extracted.log_linear_forward(
            spot,
            times,
            forwards,
            target,
        )
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_invalid_bid_ask_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="bid",
    ):
        extracted.invert_bid_ask(
            1.2,
            1.1,
        )


def test_zero_quote_year_fraction_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="quote_year_fraction",
    ):
        extracted.implied_quote_rate_from_forward(
            1.0,
            1.01,
            base_rate=0.03,
            base_year_fraction=0.25,
            quote_year_fraction=0.0,
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
            "Use covered interest rate parity to construct FX forwards "
            "and synthetic cross rates."
        ),
        (
            "Recover implied foreign yields from market forward quotes."
        ),
        (
            "Reconcile the cross-currency basis and broken-date FX forwards."
        ),
    ),
)
def test_specific_fx_language_selects_capability(
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
        "fx-forward-cross-currency"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_generic_option_fx_language_does_not_select_capability(
    tmp_path: Path,
) -> None:
    instruction = (
        "Price an FX option with the Garman-Kohlhagen model "
        "and calculate delta and vega."
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
        "fx-forward-cross-currency"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
