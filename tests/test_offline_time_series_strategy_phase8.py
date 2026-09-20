from __future__ import annotations

from pathlib import Path

import pandas as pd

from agent.offline_time_series_strategy import (
    TimeSeriesStrategySkill,
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
Task: EMA Crossover Momentum Backtest (TEST)

Use EMA(3) and EMA(6) on adj_close.
A golden cross buys at the next trading day's adj_open.
A death cross sells at the next trading day's adj_open.
Long-only backtest. Initial capital: $10,000.
Write results.json, trades.csv, daily_portfolio.csv and
cumulative_returns.html.
""".strip(),
        encoding="utf-8",
    )

    dates = pd.bdate_range(
        "2024-01-02",
        periods=30,
    )

    prices = [
        10, 10, 10, 10, 10,
        11, 12, 13, 14, 15,
        14, 13, 12, 11, 10,
        9, 8, 7, 8, 9,
        10, 11, 12, 13, 14,
        13, 12, 11, 10, 9,
    ]

    pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": [
                value + 1
                for value
                in prices
            ],
            "low": [
                value - 1
                for value
                in prices
            ],
            "close": prices,
            "adj_close": prices,
            "adj_open": prices,
            "volume": 1000,
        }
    ).to_csv(
        data
        / "test_prices.csv",
        index=False,
    )

    return task


def test_time_series_strategy_matches() -> None:
    skill = TimeSeriesStrategySkill()

    assert skill.matches(
        instruction=(
            "EMA crossover momentum backtest "
            "with a golden cross and death cross."
        ),
        task_dir=Path("."),
    )


def test_time_series_strategy_outputs(
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

    skill = TimeSeriesStrategySkill()

    skill.solve(
        instruction=instruction,
        task_dir=task,
        out_dir=out,
        seed=42,
    )

    assert {
        path.name
        for path
        in out.iterdir()
    } == {
        "results.json",
        "trades.csv",
        "daily_portfolio.csv",
        "cumulative_returns.html",
    }

    trades = pd.read_csv(
        out
        / "trades.csv"
    )

    portfolio = pd.read_csv(
        out
        / "daily_portfolio.csv"
    )

    assert len(
        portfolio
    ) == 30

    assert {
        "ticker",
        "type",
        "signal_date",
        "exec_date",
        "price",
        "shares",
        "pnl",
    }.issubset(
        trades.columns
    )

    html = (
        out
        / "cumulative_returns.html"
    ).read_text(
        encoding="utf-8"
    ).lower()

    assert "plotly" in html
