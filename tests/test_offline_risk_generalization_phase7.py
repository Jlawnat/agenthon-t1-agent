from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_common.risk import (
    expanding_historical_backtest,
    historical_var_es,
    normal_var_es,
    parse_risk_spec,
)
from agent.offline_risk import (
    MarketRiskSkill,
)


def _write_task(
    root: Path,
) -> Path:
    task = root / "task"
    data = (
        task
        / "environment"
        / "data"
    )
    data.mkdir(
        parents=True
    )

    (
        task
        / "instruction.md"
    ).write_text(
        """
Compare Value-at-Risk and Expected Shortfall using historical
simulation, Normal, Student-t, age-weighted historical simulation,
and EWMA. Use a portfolio notional value: $1,000,000.
The age-weighted decay parameter lambda = 0.98.
Use EWMA with lambda = 0.94.
Compute the 10-day horizon and a Kupiec backtest with a
minimum of 20 observations. Write var_1day.csv,
var_10day.csv, backtest.json, data_summary.json and summary.json.
Use 95% and 99% confidence.
""".strip(),
        encoding="utf-8",
    )

    dates = pd.bdate_range(
        "2020-01-02",
        periods=80,
    )

    rows = []

    for index, date in enumerate(
        dates
    ):
        for symbol, drift in (
            (
                "AAA",
                0.001,
            ),
            (
                "BBB",
                0.0005,
            ),
        ):
            price = (
                100.0
                * np.exp(
                    drift
                    * index
                    + 0.02
                    * np.sin(
                        index
                        / (
                            4.0
                            if symbol
                            == "AAA"
                            else 7.0
                        )
                    )
                )
            )

            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": price,
                    "high": (
                        price
                        * 1.01
                    ),
                    "low": (
                        price
                        * 0.99
                    ),
                    "close": price,
                    "volume": 1000,
                }
            )

    pd.DataFrame(
        rows
    ).to_csv(
        data
        / "market.csv",
        index=False,
    )

    return task


def test_generic_risk_primitives() -> None:
    instruction = """
    Portfolio notional value: $2,000,000.
    Decay parameter lambda = 0.97.
    EWMA with lambda = 0.93.
    5-day horizon.
    Minimum of 30 observations.
    Confidence levels 95% and 99%.
    """

    spec = parse_risk_spec(
        instruction
    )

    assert spec.notional == 2_000_000.0
    assert spec.age_lambda == 0.97
    assert spec.ewma_lambda == 0.93
    assert spec.horizon_days == 5
    assert spec.backtest_min_obs == 30
    assert spec.confidence_levels == (
        0.95,
        0.99,
    )

    sample = np.asarray(
        [
            -0.03,
            -0.02,
            -0.01,
            0.0,
            0.01,
            0.02,
        ]
    )

    hs_var, hs_es = historical_var_es(
        sample,
        alpha=0.95,
        notional=1_000_000.0,
    )

    assert hs_var > 0.0
    assert hs_es >= hs_var * 0.95

    normal_var, normal_es = normal_var_es(
        sample,
        alpha=0.99,
        notional=1_000_000.0,
    )

    assert normal_var > 0.0
    assert normal_es > normal_var

    (
        n_days,
        n_exceedances,
        rate,
        pvalue,
    ) = expanding_historical_backtest(
        np.linspace(
            -0.04,
            0.04,
            100,
        ),
        alpha=0.99,
        minimum_observations=20,
    )

    assert n_days == 80
    assert n_exceedances >= 0
    assert 0.0 <= rate <= 1.0
    assert 0.0 <= pvalue <= 1.0


def test_generic_market_risk_skill(
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

    skill = MarketRiskSkill()

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

    assert {
        path.name
        for path in out.iterdir()
    } == {
        "data_summary.json",
        "var_1day.csv",
        "var_10day.csv",
        "backtest.json",
        "summary.json",
    }

    one_day = pd.read_csv(
        out
        / "var_1day.csv"
    )

    assert one_day.shape == (
        10,
        4,
    )

    assert set(
        one_day[
            "method"
        ]
    ) == {
        "HS",
        "Normal",
        "StudentT",
        "AgeWeightedHS",
        "EWMA",
    }

    assert (
        one_day[
            "var_dollar"
        ]
        > 0.0
    ).all()
