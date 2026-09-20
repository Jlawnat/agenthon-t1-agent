from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_alpha import (
    AlphaHedgeStrategySkill,
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
Construct a cross-sectional alpha signal with Fama-French
factor exposure analysis and a long-short backtest.
Report annualized_alpha and market_beta_residual in
results.json and solution.json.
""".strip(),
        encoding="utf-8",
    )

    params = {
        "lookback_alpha": 5,
        "lookback_risk": 10,
        "max_gross_leverage": 2.0,
        "max_position_pct": 0.5,
        "transaction_cost_bps": 5,
        "rebalance_frequency": "monthly",
        "annualization_factor": 252,
        "long_top_n": 1,
        "short_bottom_n": 1,
        "long_weight_per_stock": 0.5,
        "short_weight_per_stock": 0.5,
        "signal_start_day": 5,
        "trade_start_day": 6,
        "factor_names": [
            "Mkt_RF",
            "SMB",
            "HML",
        ],
        "seed": 2030,
    }

    (
        data
        / "params.json"
    ).write_text(
        json.dumps(
            params
        ),
        encoding="utf-8",
    )

    dates = pd.bdate_range(
        "2020-01-02",
        periods=70,
    )

    base = np.linspace(
        -0.005,
        0.005,
        len(
            dates
        ),
    )

    frame = {
        "date": dates,
        "STOCK_A": base + 0.002,
        "STOCK_B": -base - 0.001,
        "STOCK_C": np.sin(
            np.arange(
                len(
                    dates
                )
            )
        )
        * 0.001,
        "STOCK_D": np.cos(
            np.arange(
                len(
                    dates
                )
            )
        )
        * 0.001,
    }

    returns = pd.DataFrame(
        frame
    )

    returns.loc[
        10,
        "STOCK_C",
    ] = np.nan

    returns.to_csv(
        data
        / "returns.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "date": dates,
            "Mkt_RF": np.linspace(
                -0.002,
                0.002,
                len(
                    dates
                ),
            ),
            "SMB": np.zeros(
                len(
                    dates
                )
            ),
            "HML": np.zeros(
                len(
                    dates
                )
            ),
        }
    ).to_csv(
        data
        / "factors.csv",
        index=False,
    )

    return task


def test_alpha_skill_writes_contract(
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

    skill = (
        AlphaHedgeStrategySkill()
    )

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
        "results.json",
        "solution.json",
    }

    results = json.loads(
        (
            out
            / "results.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    solution = json.loads(
        (
            out
            / "solution.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert set(
        results
    ) == {
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "annualized_alpha",
        "market_beta_residual",
        "n_rebalances",
    }

    intermediates = solution[
        "intermediates"
    ]

    assert intermediates[
        "nan_count"
    ][
        "value"
    ] == 1

    assert (
        intermediates[
            "n_rebalances"
        ][
            "value"
        ]
        == results[
            "n_rebalances"
        ]
    )

    assert np.isclose(
        intermediates[
            "annualized_alpha"
        ][
            "value"
        ],
        results[
            "annualized_alpha"
        ],
    )

    assert np.isfinite(
        results[
            "sharpe_ratio"
        ]
    )

    assert (
        results[
            "max_drawdown"
        ]
        >= 0.0
    )
