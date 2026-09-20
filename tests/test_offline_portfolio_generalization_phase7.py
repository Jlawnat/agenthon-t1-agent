from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_common.portfolio import (
    clean_price_panel,
    parse_cross_sectional_momentum_spec,
)
from agent.offline_portfolio import (
    PortfolioStrategySkill,
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
Build a cross-sectional momentum long-short portfolio
from monthly prices. Rank using returns from t-4 through
t-2 inclusive, with valid portfolio months t >= 4.
Go long the top 1 and short the bottom 1.
Write results.json and portfolio.csv.
""".strip(),
        encoding="utf-8",
    )

    dates = pd.date_range(
        "2020-01-31",
        periods=12,
        freq="ME",
    )

    frame = pd.DataFrame(
        {
            "date": dates,
            "A": 100.0
            * np.cumprod(
                np.full(
                    12,
                    1.02,
                )
            ),
            "B": 100.0
            * np.cumprod(
                np.full(
                    12,
                    0.99,
                )
            ),
            "C": 100.0
            * np.cumprod(
                np.full(
                    12,
                    1.005,
                )
            ),
        }
    )

    # Exercise the generic cleaning sequence:
    # duplicate date (keep last) + forward fill.
    duplicate = (
        frame.iloc[
            [5]
        ].copy()
    )
    duplicate[
        "A"
    ] *= 1.001

    frame = pd.concat(
        [
            frame.iloc[
                :6
            ],
            duplicate,
            frame.iloc[
                6:
            ],
        ],
        ignore_index=True,
    )

    frame.loc[
        8,
        "C",
    ] = np.nan

    frame.to_csv(
        data
        / "prices.csv",
        index=False,
    )

    return task


def test_generic_portfolio_skill(
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

    spec = (
        parse_cross_sectional_momentum_spec(
            instruction
        )
    )

    assert spec.lookback_start == 4
    assert spec.skip_recent == 2
    assert spec.valid_start == 4
    assert spec.long_count == 1
    assert spec.short_count == 1
    assert spec.periods_per_year == 12

    raw = pd.read_csv(
        task
        / "environment"
        / "data"
        / "prices.csv"
    )

    cleaned, assets = (
        clean_price_panel(
            raw
        )
    )

    assert len(
        cleaned
    ) == 12
    assert assets == [
        "A",
        "B",
        "C",
    ]

    skill = (
        PortfolioStrategySkill()
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

    assert (
        out
        / "results.json"
    ).is_file()

    portfolio = pd.read_csv(
        out
        / "portfolio.csv"
    )

    assert not portfolio.empty
    assert list(
        portfolio.columns
    ) == [
        "date",
        "signal_return",
        "cumulative_return",
    ]

    assert (
        portfolio[
            "cumulative_return"
        ]
        > 0.0
    ).all()
