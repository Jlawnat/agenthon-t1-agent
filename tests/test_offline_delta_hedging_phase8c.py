from __future__ import annotations

import numpy as np
import pandas as pd

from agent.offline_delta_hedging import (
    DeltaHedgingPnlSkill,
    _escrowed_call_price_delta,
    _pv_remaining_dividends,
)


def test_delta_hedging_skill_matches() -> None:
    skill = DeltaHedgingPnlSkill()

    assert skill.matches(
        instruction=(
            "Run a discrete delta-hedging P&L simulation with "
            "transaction costs, discrete dividends, and time-varying "
            "implied volatility."
        ),
        task_dir=None,
    )


def test_t0_escrowed_dividend_values() -> None:
    dividends = pd.DataFrame(
        {
            "ex_div_day": [
                60,
                160,
            ],
            "amount": [
                0.55,
                0.60,
            ],
        }
    )

    pv = _pv_remaining_dividends(
        day=0,
        dividends=dividends,
        risk_free_rate=0.03,
        trading_days=252,
    )

    price, delta, effective = (
        _escrowed_call_price_delta(
            spot=100.0,
            strike=100.0,
            tau=1.0,
            risk_free_rate=0.03,
            volatility=0.20,
            pv_dividends=pv,
        )
    )

    assert np.isclose(
        pv,
        1.134765,
        atol=1e-4,
    )

    assert np.isclose(
        effective,
        98.865235,
        atol=1e-4,
    )

    assert np.isclose(
        price,
        8.746564,
        rtol=1e-4,
    )

    assert np.isclose(
        delta,
        0.576496,
        rtol=1e-4,
    )


def test_parse_markdown_wrapped_contract_parameters() -> None:
    from agent.offline_delta_hedging import _parse_contract_parameters

    instruction = """
| Symbol | Value |
| Strike K | 100.0 |
| Maturity | `T = 1.0` year = 252 trading days, `dt = 1/252` |
| Risk-free rate r | 0.03 (continuously compounded) |

Run four simulations at `freq_days ∈ {1, 5, 10, 22}`.
"""

    params = _parse_contract_parameters(instruction)

    assert params["strike"] == 100.0
    assert params["risk_free_rate"] == 0.03
    assert params["maturity_years"] == 1.0
    assert params["trading_days"] == 252
    assert params["frequencies"] == [1, 5, 10, 22]
