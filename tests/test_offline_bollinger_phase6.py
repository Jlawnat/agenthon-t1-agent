from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_bollinger import (
    BollingerBacktestSkill,
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
# Bollinger Band Mean Reversion Backtest
Use adj_close and adj_open with risk-based position sizing
and a trailing stop.
Write daily_portfolio.csv and trade_analysis.json.
""".strip(),
        encoding="utf-8",
    )

    dates = pd.bdate_range(
        "2026-01-02",
        periods=40,
    )

    prices = np.full(
        len(dates),
        100.0,
    )

    pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "adj_close": prices,
            "volume": np.full(
                len(dates),
                1_000_000,
            ),
            "adj_open": prices,
        }
    ).to_csv(
        data_dir
        / "aapl_prices.csv",
        index=False,
    )

    return task


def test_bollinger_skill_writes_required_outputs(
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
        BollingerBacktestSkill()
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

    expected = {
        "results.json",
        "trades.csv",
        "daily_portfolio.csv",
        "cumulative_returns.html",
        "trade_analysis.json",
    }

    assert {
        path.name
        for path in out.iterdir()
        if path.is_file()
    } == expected

    results = json.loads(
        (
            out
            / "results.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert results[
        "AAPL"
    ][
        "num_trading_days"
    ] == 40

    assert results[
        "AAPL"
    ][
        "final_capital"
    ] == 100000.0

    assert results[
        "AAPL"
    ][
        "num_trades"
    ] == 0

    portfolio = pd.read_csv(
        out
        / "daily_portfolio.csv"
    )

    assert len(
        portfolio
    ) == 40

    assert np.isclose(
        portfolio[
            "cumulative_return"
        ].iloc[0],
        1.0,
    )

    assert np.isclose(
        portfolio[
            "cumulative_return"
        ].iloc[-1],
        1.0,
    )

    trades = pd.read_csv(
        out
        / "trades.csv"
    )

    assert list(
        trades.columns
    ) == [
        "ticker",
        "type",
        "signal_date",
        "exec_date",
        "price",
        "shares",
        "pnl",
    ]

    analysis = json.loads(
        (
            out
            / "trade_analysis.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert analysis == {
        "trades": []
    }

    html = (
        out
        / "cumulative_returns.html"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "Plotly.newPlot"
        in html
    )
    assert (
        "AAPL"
        in html
    )
