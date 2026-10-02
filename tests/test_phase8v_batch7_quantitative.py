from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.capability_bridge import rank_capabilities
from agent.offline_baw import _baw_price, _crr_american
from agent.offline_cme_hdd import _mc_price_with_z
from agent.offline_geske import _geske_call_on_call, _geske_put_on_put
from agent.offline_implied_vol_approximations import _brenner, _cmh, _li_atm, _li_nonatm
from agent.offline_option_parity_audit import _compute_pair_row
from agent.offline_common import (
    american_options,
    digital_options,
    local_volatility,
    stochastic_volatility,
    vanilla_options,
    weather_derivatives,
)
from agent.qf_primitives import black_scholes_greeks as mature_greeks


PUBLIC_UNITS = Path("/home/elliot/track1-coding-public/units")


@pytest.mark.parametrize("kind", ("call", "put"))
def test_black_scholes_greeks_match_curated_primitives_and_pde(kind: str) -> None:
    inputs = (105.0, 100.0, 0.04, 0.015, 0.22, 0.75, kind)
    result = vanilla_options.black_scholes_greeks(*inputs)
    mature = mature_greeks(*inputs)
    assert np.isclose(result.delta, mature["delta"])
    assert np.isclose(result.gamma, mature["gamma"])
    assert np.isclose(result.theta, mature["theta_annual"])
    assert np.isclose(result.vega, mature["vega"])
    assert np.isclose(result.rho, mature["rho"])
    assert abs(
        vanilla_options.black_scholes_pde_residual(
            result.price,
            result.delta,
            result.gamma,
            result.theta,
            spot=inputs[0],
            rate=inputs[2],
            dividend_yield=inputs[3],
            volatility=inputs[4],
        )
    ) < 1e-12


def test_implied_volatility_inversion_and_approximations_match_mature() -> None:
    spot, strike, rate, dividend, maturity, sigma = 500.0, 500.0, 0.05, 0.013, 0.5, 0.2
    price = vanilla_options.black_scholes_greeks(spot, strike, rate, dividend, sigma, maturity, "call").price
    inverted, converged = vanilla_options.implied_volatility_newton(price, spot, strike, rate, dividend, maturity)
    assert converged and np.isclose(inverted, sigma, atol=1e-10)
    discounted_spot = spot * math.exp(-dividend * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)
    assert np.isclose(vanilla_options.brenner_subrahmanyam_atm(price, discounted_spot, maturity), _brenner(price, maturity, 1.0))
    assert np.isclose(vanilla_options.li_atm(price, discounted_spot, maturity), _li_atm(price, maturity, 1.0))
    assert np.isclose(vanilla_options.li_non_atm(price, discounted_spot, discounted_strike, maturity), _li_nonatm(price, strike, maturity))
    assert np.isclose(vanilla_options.corrado_miller_hallerbach(price, discounted_spot, discounted_strike, maturity), _cmh(price, strike, maturity))


def test_put_call_parity_forward_matches_mature_audit_row() -> None:
    call = pd.Series({"expiry_ts": "2025-07-01T00:00:00Z", "strike": 100.0, "bid": 6.0, "ask": 6.2, "quote_id": "c"})
    put = pd.Series({"expiry_ts": "2025-07-01T00:00:00Z", "strike": 100.0, "bid": 4.0, "ask": 4.2, "quote_id": "p"})
    mature = _compute_pair_row(call=call, put=put, quote_ts="2025-01-01T00:00:00Z", spot_price=101.0, risk_free_rate=0.04, dividend_yield=0.01, reference_borrow_rate=0.005, violation_threshold=0.01)
    generic = vanilla_options.put_call_parity_forward(100.0, 6.0, 6.2, 4.0, 4.2, spot=101.0, rate=0.04, dividend_yield=0.01, borrow_rate=0.005, maturity=mature["time_to_expiry_years"])
    assert np.isclose(generic.synthetic_bid, mature["synthetic_forward_bid"])
    assert np.isclose(generic.synthetic_ask, mature["synthetic_forward_ask"])
    assert np.isclose(generic.implied_borrow_rate, mature["implied_borrow_rate"])
    assert np.isclose(generic.violation_amount, mature["violation_amount"])


@pytest.mark.parametrize("kind", ("call", "put"))
def test_baw_and_crr_match_mature_implementations(kind: str) -> None:
    kwargs = dict(spot=100.0, strike=100.0, rate=0.05, dividend=0.013, volatility=0.2, maturity=1.0, kind=kind)
    mature_price, mature_boundary = _baw_price(**kwargs)
    price, boundary = american_options.barone_adesi_whaley(100.0, 100.0, 0.05, 0.013, 0.2, 1.0, kind)
    assert np.isclose(price, mature_price)
    assert np.isclose(boundary, mature_boundary)
    generic_tree = american_options.crr_american_option(100.0, 100.0, 0.05, 0.013, 0.2, 1.0, kind, steps=250)
    mature_tree = _crr_american(**{**kwargs, "steps": 250})
    assert np.isclose(generic_tree, mature_tree)


@pytest.mark.parametrize("kind", ("call_on_call", "put_on_put"))
def test_geske_compound_prices_match_mature(kind: str) -> None:
    args = dict(spot=100.0, outer_strike=10.0, inner_strike=100.0, rate=0.05, dividend=0.01, volatility=0.2, outer_maturity=0.25, inner_maturity=1.0)
    mature = _geske_call_on_call(**args) if kind == "call_on_call" else _geske_put_on_put(**args)
    generic = american_options.geske_compound_option(100.0, 10.0, 100.0, 0.05, 0.01, 0.2, 0.25, 1.0, kind)
    assert np.allclose(generic, mature)


def test_digital_decompositions_and_barrier_parity() -> None:
    cash_call = digital_options.cash_or_nothing(100, 105, 0.04, 0.01, 0.2, 0.5, "call")
    cash_put = digital_options.cash_or_nothing(100, 105, 0.04, 0.01, 0.2, 0.5, "put")
    assert np.isclose(cash_call + cash_put, math.exp(-0.04 * 0.5))
    asset_call = digital_options.asset_or_nothing(100, 105, 0.04, 0.01, 0.2, 0.5, "call")
    gap = digital_options.gap_option(100, 100, 105, 0.04, 0.01, 0.2, 0.5, "call")
    assert np.isclose(gap, asset_call - 100 * cash_call)
    knockout, knockin = digital_options.reflected_barrier_binary(100, 100, 85, 0.04, 0.01, 0.2, 0.5, "down_out_call")
    vanilla = digital_options.cash_or_nothing(100, 100, 0.04, 0.01, 0.2, 0.5, "call")
    assert np.isclose(knockout + knockin, vanilla)


def test_svi_calibration_and_dupire_flat_volatility() -> None:
    k = np.linspace(-0.2, 0.2, 9)
    parameters = np.array([0.02, 0.12, -0.25, 0.01, 0.18])
    maturity = 0.75
    iv = np.sqrt(local_volatility.svi_total_variance(k, parameters) / maturity)
    fit = local_volatility.calibrate_svi(k, iv, maturity)
    assert fit.converged
    assert fit.volatility_rmse < 1e-3

    strike, sigma, rate, dividend = 100.0, 0.25, 0.03, 0.01
    price = vanilla_options.black_scholes_greeks(100.0, strike, rate, dividend, sigma, 0.5, "call")
    d1 = ((rate - dividend + 0.5 * sigma**2) * 0.5) / (sigma * math.sqrt(0.5))
    d2 = d1 - sigma * math.sqrt(0.5)
    from scipy.stats import norm
    strike_derivative = -math.exp(-rate * 0.5) * norm.cdf(d2)
    local = local_volatility.dupire_local_volatility(
        strike,
        price.price,
        -price.theta,
        strike_derivative,
        price.gamma,
        rate=rate,
        dividend_yield=dividend,
    )
    assert np.isclose(local, sigma, atol=1e-12)


def test_local_vol_barrier_monte_carlo_invariants() -> None:
    result = local_volatility.local_vol_barrier_monte_carlo(
        100.0,
        100.0,
        80.0,
        0.5,
        0.03,
        0.0,
        [0.0, 0.5, 1.0],
        [50.0, 100.0, 150.0],
        np.full((3, 3), 0.2),
        paths=2000,
        steps=20,
        monitoring_interval=2,
        seed=42,
    )
    assert 0.0 <= result.knockout_price <= result.vanilla_price
    assert 0.0 <= result.hit_probability <= 1.0
    assert result.knockout_standard_error >= 0.0


def test_weather_hdd_ou_and_monte_carlo_match_mature_core() -> None:
    temperatures = [10.0, 15.0, 20.0]
    assert np.allclose(weather_derivatives.heating_degree_days(temperatures, 18.0), [8.0, 3.0, 0.0])
    seasonal = np.array([8.0, 9.0, 10.0, 11.0])
    draws = np.random.default_rng(7).standard_normal((100, len(seasonal)))
    generic = weather_derivatives.simulate_hdd_option(
        seasonal,
        base_temperature=18.0,
        strike=32.0,
        tick_value=20.0,
        rate=0.03,
        time_to_payment=30 / 365,
        kappa=12.0,
        theta=0.0,
        sigma=3.0,
        paths=100,
        seed=7,
        normal_draws=draws,
    )
    mature_price, mature_hdd, mature_payoffs = _mc_price_with_z(12.0, 0.0, 3.0, seasonal, draws, 18.0, 32.0, 20.0, math.exp(-0.03 * 30 / 365))
    assert np.isclose(generic.price, mature_price)
    assert np.allclose(generic.index_values, mature_hdd)
    assert np.allclose(generic.payoffs, mature_payoffs)


def test_two_factor_heston_outputs_are_finite_and_obey_parity() -> None:
    factors = [
        stochastic_volatility.HestonFactor(0.5, 0.04, 0.2, 0.04, -0.5),
        stochastic_volatility.HestonFactor(2.0, 0.04, 0.3, 0.04, -0.8),
    ]
    call = stochastic_volatility.two_factor_heston_call(110.0, 100.0, 0.5, 0.03, 0.0, factors, quadrature_nodes=256)
    put = stochastic_volatility.put_from_call_parity(call, 110.0, 100.0, 0.03, 0.0, 0.5)
    assert np.isfinite(call) and call >= 0.0
    assert np.isfinite(put) and put >= 0.0
    assert np.isclose(call - put, 110.0 - 100.0 * math.exp(-0.03 * 0.5))


def test_two_factor_heston_matches_mature_gauss_legendre_price() -> None:
    z = 3.0
    kappa_one = 2.0 ** (-z / 2.0)
    kappa_two = 2.0 ** (z / 2.0)
    factors = [
        stochastic_volatility.HestonFactor(kappa_one, 0.04, math.sqrt(kappa_one * 0.04), 0.04, -0.5),
        stochastic_volatility.HestonFactor(kappa_two, 0.04, math.sqrt(kappa_two * 0.04), 0.04, -0.99),
    ]
    price = stochastic_volatility.two_factor_heston_call(
        110.0,
        100.0,
        0.5,
        0.03,
        0.0,
        factors,
        quadrature_nodes=1000,
    )
    assert np.isclose(price, 15.462884426139055, atol=5e-10)


@pytest.mark.parametrize(
    ("unit_name", "capability_id"),
    (
        ("t1-barone-adesi-whaley", "american-and-compound-options"),
        ("t1-bs-greeks-pde", "vanilla-option-analytics"),
        ("t1-cme-hdd-option-pricing", "weather-derivative-pricing"),
        ("t1-compound-option-geske", "american-and-compound-options"),
        ("t1-digital-barrier-options", "digital-and-barrier-binaries"),
        ("t1-dupire-local-vol", "local-volatility-surfaces"),
        ("t1-implied-vol-approximations", "vanilla-option-analytics"),
        ("t1-localvol-barrier", "local-volatility-surfaces"),
        ("t1-option-put-call-parity-forward-audit", "vanilla-option-analytics"),
        ("t1-stochvol-implied-surface-new", "two-factor-stochastic-volatility"),
    ),
)
def test_batch7_public_units_route_narrowly(unit_name: str, capability_id: str) -> None:
    task = PUBLIC_UNITS / unit_name
    selected = rank_capabilities(instruction=(task / "instruction.md").read_text(), task_dir=task)
    assert capability_id in {item.descriptor.capability_id for item in selected}


@pytest.mark.parametrize(
    "instruction",
    (
        "Fit an ordinary least-squares regression.",
        "Compute a generic finite-difference derivative.",
        "Simulate a geometric Brownian price path.",
        "Summarize the weather observations.",
        "Compare calls and puts without a parity audit.",
    ),
)
def test_batch7_generic_language_does_not_route(tmp_path: Path, instruction: str) -> None:
    task = tmp_path / "task"
    task.mkdir()
    assert rank_capabilities(instruction=instruction, task_dir=task) == ()
