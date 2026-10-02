from __future__ import annotations

import math

import numpy as np
import pytest

from agent import offline_merton_jump_diffusion as mature
from agent.offline_common import merton_jump_diffusion as extracted
from agent.qf_primitives import black_scholes_price


def _returns() -> np.ndarray:
    rng = np.random.default_rng(314159)
    return rng.normal(
        loc=0.0003,
        scale=0.012,
        size=180,
    )


def test_negative_log_likelihood_matches_mature() -> None:
    values = _returns()

    dt = 1.0 / 252.0
    drift = float(
        np.mean(values)
        / dt
    )

    sigma = 0.18
    intensity = 2.4
    mu_jump = -0.025
    sigma_jump = 0.045

    expected = mature._negative_log_likelihood(
        (
            sigma,
            intensity,
            mu_jump,
            sigma_jump,
        ),
        values,
        dt,
        drift,
        15,
    )

    actual = (
        extracted.merton_jump_negative_log_likelihood(
            values,
            dt=dt,
            drift=drift,
            diffusion_volatility=sigma,
            jump_intensity=intensity,
            jump_mean=mu_jump,
            jump_volatility=sigma_jump,
            max_jumps=15,
        )
    )

    assert np.isclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_merton_call_price_matches_mature() -> None:
    parameters = dict(
        spot=100.0,
        strike=105.0,
        rate=0.04,
        maturity=0.75,
        diffusion_volatility=0.20,
        jump_intensity=1.8,
        jump_mean=-0.03,
        jump_volatility=0.08,
    )

    expected = mature._merton_call(
        parameters["spot"],
        parameters["strike"],
        parameters["rate"],
        parameters["maturity"],
        parameters["diffusion_volatility"],
        parameters["jump_intensity"],
        parameters["jump_mean"],
        parameters["jump_volatility"],
        50,
    )

    actual = extracted.merton_call_price(
        **parameters,
        n_terms=50,
    )

    assert np.isclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_zero_jump_intensity_reduces_to_black_scholes() -> None:
    spot = 100.0
    strike = 95.0
    rate = 0.03
    maturity = 0.8
    sigma = 0.24

    actual = extracted.merton_call_price(
        spot=spot,
        strike=strike,
        rate=rate,
        maturity=maturity,
        diffusion_volatility=sigma,
        jump_intensity=0.0,
        jump_mean=-0.04,
        jump_volatility=0.10,
    )

    expected = black_scholes_price(
        spot,
        strike,
        rate,
        0.0,
        sigma,
        maturity,
        "call",
    )

    assert np.isclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_total_volatility_formula() -> None:
    sigma = 0.20
    intensity = 3.0
    mu_jump = -0.02
    sigma_jump = 0.06

    expected = math.sqrt(
        sigma**2
        + intensity
        * (
            mu_jump**2
            + sigma_jump**2
        )
    )

    actual = (
        extracted.merton_total_volatility(
            sigma,
            intensity,
            mu_jump,
            sigma_jump,
        )
    )

    assert np.isclose(
        actual,
        expected,
    )


def test_calibration_returns_valid_parameters() -> None:
    values = _returns()
    dt = 1.0 / 252.0

    result = (
        extracted.calibrate_merton_jump_diffusion(
            values,
            dt=dt,
            drift=float(
                np.mean(values)
                / dt
            ),
            bounds=(
                (0.05, 0.50),
                (0.10, 6.00),
                (-0.10, 0.10),
                (0.005, 0.20),
            ),
            restarts=2,
            seed=42,
            max_jumps=10,
        )
    )

    assert 0.05 <= result.diffusion_volatility <= 0.50
    assert 0.10 <= result.jump_intensity <= 6.00
    assert -0.10 <= result.jump_mean <= 0.10
    assert 0.005 <= result.jump_volatility <= 0.20
    assert np.isfinite(
        result.log_likelihood
    )


def test_merton_call_is_nonnegative() -> None:
    value = extracted.merton_call_price(
        spot=100.0,
        strike=120.0,
        rate=0.03,
        maturity=1.0,
        diffusion_volatility=0.20,
        jump_intensity=2.0,
        jump_mean=-0.04,
        jump_volatility=0.09,
    )

    assert value >= 0.0


def test_invalid_likelihood_parameters_rejected() -> None:
    with pytest.raises(ValueError):
        extracted.merton_jump_negative_log_likelihood(
            [0.01, -0.01],
            dt=1.0 / 252.0,
            drift=0.05,
            diffusion_volatility=0.0,
            jump_intensity=1.0,
            jump_mean=0.0,
            jump_volatility=0.05,
        )


def test_invalid_calibration_bounds_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="four parameter",
    ):
        extracted.calibrate_merton_jump_diffusion(
            _returns(),
            dt=1.0 / 252.0,
            drift=0.05,
            bounds=(
                (0.05, 0.50),
                (0.10, 6.00),
            ),
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
        (
            "Calibrate a Merton jump-diffusion model "
            "and price calls using the Merton series formula."
        ),
        (
            "Fit the Poisson-weighted mixture for a "
            "Merton jump diffusion model."
        ),
    ),
)
def test_merton_language_selects_capability(
    tmp_path: Path,
    instruction: str,
) -> None:
    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "merton-jump-diffusion"
        in {
            item.descriptor.capability_id
            for item in selected
        }
    )


def test_generic_jump_language_does_not_select_merton(
    tmp_path: Path,
) -> None:
    instruction = (
        "Simulate a generic jump process with Poisson arrivals."
    )

    task = _task(tmp_path, instruction)

    selected = rank_capabilities(
        instruction=instruction,
        task_dir=task,
    )

    assert (
        "merton-jump-diffusion"
        not in {
            item.descriptor.capability_id
            for item in selected
        }
    )
