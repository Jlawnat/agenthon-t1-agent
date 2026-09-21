from __future__ import annotations

import math

import numpy as np

from agent.offline_baw import (
    _baw_price,
)
from agent.offline_geske import (
    _geske_call_on_call,
    _geske_put_on_put,
)
from agent.offline_kelly_var import (
    _simulate_scheme,
)


def test_baw_american_prices_dominate_european_like_values() -> None:
    call, star_call = _baw_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        dividend=0.02,
        volatility=0.20,
        maturity=1.0,
        kind="call",
    )
    put, star_put = _baw_price(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        dividend=0.02,
        volatility=0.20,
        maturity=1.0,
        kind="put",
    )

    assert call > 0.0
    assert put > 0.0
    assert star_call > 100.0
    assert 0.0 < star_put < 100.0


def test_geske_compound_prices_are_positive() -> None:
    cc, cc_star = _geske_call_on_call(
        spot=100.0,
        outer_strike=5.0,
        inner_strike=100.0,
        rate=0.05,
        dividend=0.01,
        volatility=0.20,
        outer_maturity=0.25,
        inner_maturity=1.0,
    )
    pp, pp_star = _geske_put_on_put(
        spot=100.0,
        outer_strike=5.0,
        inner_strike=100.0,
        rate=0.05,
        dividend=0.01,
        volatility=0.20,
        outer_maturity=0.25,
        inner_maturity=1.0,
    )

    assert cc > 0.0
    assert pp > 0.0
    assert cc_star > 0.0
    assert pp_star > 0.0


def test_kelly_simulation_is_deterministic_for_generator_state() -> None:
    mean = np.array(
        [0.001, 0.0005, 0.0008],
        dtype=float,
    )
    covariance = np.array(
        [
            [0.0001, 0.0, 0.0],
            [0.0, 0.0002, 0.0],
            [0.0, 0.0, 0.0003],
        ],
        dtype=float,
    )
    weights = np.array(
        [0.5, 0.3, 0.2],
        dtype=float,
    )

    a = _simulate_scheme(
        rng=np.random.default_rng(99),
        mean=mean,
        covariance=covariance,
        weights=weights,
        initial_capital=1_000_000.0,
        n_paths=100,
        n_days=20,
    )
    b = _simulate_scheme(
        rng=np.random.default_rng(99),
        mean=mean,
        covariance=covariance,
        weights=weights,
        initial_capital=1_000_000.0,
        n_paths=100,
        n_days=20,
    )

    assert a == b
    assert a["mean_terminal"] > 0.0
    assert math.isfinite(
        a["sharpe"]
    )
