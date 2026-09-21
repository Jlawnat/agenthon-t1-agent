from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_double_sort import DoubleSortCornerSkill
from agent.offline_historical_var import HistoricalVarDataPrepSkill


def test_historical_var_cleaning_pipeline(tmp_path: Path) -> None:
    task = tmp_path / "unit"
    data = task / "environment" / "data"
    out = tmp_path / "out"
    data.mkdir(parents=True)

    frame = pd.DataFrame(
        {
            " date ": [
                "2024-01-02",
                "01/03/2024",
                "03-Jan-24",
                "2024-01-04",
                "2024-01-05",
                "2024-01-06",
                "2024-01-08",
            ],
            "SPY": [0.01, 0.02, 0.03, 0.04, 0.30, 0.01, np.nan],
            "TLT": [0.01, 0.02, 0.04, 0.04, 0.01, 0.01, 0.01],
            "GLD": [0.01, 0.02, 0.04, 0.04, 0.01, 0.01, 0.01],
            "IEF": [0.01, 0.02, 0.04, 0.04, 0.01, 0.01, 0.01],
            "EFA": [0.01, 0.02, 0.04, 0.04, 0.01, 0.01, 0.01],
        }
    )
    frame.to_csv(data / "returns_raw.csv", index=False)
    pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
            ]
        }
    ).to_csv(data / "trading_calendar.csv", index=False)

    HistoricalVarDataPrepSkill().solve(
        instruction="",
        task_dir=task,
        out_dir=out,
        seed=42,
    )

    results = json.loads((out / "results.json").read_text())
    solution = json.loads((out / "solution.json").read_text())
    report = solution["cleaning_report"]

    assert report == {
        "n_rows_raw": 7,
        "n_non_trading_days_removed": 1,
        "n_duplicates_removed": 1,
        "n_outliers_removed": 1,
        "n_missing_rows_removed": 1,
        "n_rows_after_cleaning": 3,
    }
    assert results["n_observations_clean"] == 3
    assert results["worst_day_date"] == "2024-01-02"


def test_double_sort_exact_turnover_convention(tmp_path: Path) -> None:
    task = tmp_path / "unit"
    data = task / "environment" / "data"
    out = tmp_path / "out"
    data.mkdir(parents=True)

    rows = []
    dates = ["2024-01-31", "2024-02-29"]
    for date_index, date in enumerate(dates):
        for identifier in range(25):
            beta = float(identifier)
            momentum = float(identifier if date_index == 0 else -identifier)
            ret = 0.10 + 0.001 * identifier + 0.01 * date_index
            rows.append(
                {
                    "identifier": identifier,
                    "date": date,
                    "betabab_1260d": beta,
                    "ret_12_7": momentum,
                    "ret_lead1m": ret,
                }
            )

    pd.DataFrame(rows).to_parquet(
        data / "stock_data.parquet",
        index=False,
    )

    DoubleSortCornerSkill().solve(
        instruction="",
        task_dir=task,
        out_dir=out,
        seed=42,
    )

    result = pd.read_csv(out / "strategy_returns.csv")
    assert result["date"].tolist() == dates
    assert np.isclose(
        result.loc[0, "net_return"],
        0.104 - 0.0015,
    )
    assert np.isclose(
        result.loc[1, "net_return"],
        0.110 - 0.0015 * 2.0,
    )
