from __future__ import annotations

from pathlib import Path

import pandas as pd

from agent.offline_time_series_strategy import (
    TimeSeriesStrategySkill,
    _parse_ma_config,
)


def test_parse_ema_and_sma_configs() -> None:
    assert _parse_ma_config(
        "Use EMA(50) and EMA(200) for the crossover."
    ) == (
        "EMA",
        50,
        200,
    )

    assert _parse_ma_config(
        "Use SMA(50) and SMA(200) for the crossover."
    ) == (
        "SMA",
        50,
        200,
    )


def _write_task(
    root: Path,
    *,
    kind: str,
) -> Path:
    task = root / kind.lower()
    data = task / "environment" / "data"
    data.mkdir(parents=True)

    (task / "instruction.md").write_text(
        f"""
Task: {kind} Crossover Momentum Backtest (TEST)

Use {kind}(3) and {kind}(6) on adj_close.
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
            "high": [x + 1 for x in prices],
            "low": [x - 1 for x in prices],
            "close": prices,
            "adj_close": prices,
            "adj_open": prices,
            "volume": 1000,
        }
    ).to_csv(
        data / "test_prices.csv",
        index=False,
    )

    return task


def test_skill_matches_both_ma_families() -> None:
    skill = TimeSeriesStrategySkill()

    assert skill.matches(
        instruction=(
            "EMA crossover momentum backtest "
            "with a golden cross and death cross."
        ),
        task_dir=Path("."),
    )

    assert skill.matches(
        instruction=(
            "SMA crossover momentum backtest "
            "with a golden cross and death cross."
        ),
        task_dir=Path("."),
    )


def test_ema_and_sma_both_produce_outputs(
    tmp_path: Path,
) -> None:
    for kind in ("EMA", "SMA"):
        task = _write_task(
            tmp_path,
            kind=kind,
        )

        out = tmp_path / f"out-{kind.lower()}"

        instruction = (
            task / "instruction.md"
        ).read_text(
            encoding="utf-8"
        )

        TimeSeriesStrategySkill().solve(
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
            "trades.csv",
            "daily_portfolio.csv",
            "cumulative_returns.html",
        }

        html = (
            out / "cumulative_returns.html"
        ).read_text(
            encoding="utf-8"
        )

        assert (
            f"{kind} Crossover Momentum"
            in html
        )
