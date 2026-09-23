from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from agent.offline_runtime import solve_offline


def test_current_bs_greeks_runs_without_checks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = tmp_path / "task"
    data = task / "environment" / "data"
    out = tmp_path / "output"

    data.mkdir(parents=True)

    (task / "instruction.md").write_text(
        """
# Task: Black-Scholes Greeks via Finite-Difference PDE Solver

Read /input/environment/data/options.parquet.

Compute Black-Scholes option price and Greeks.

Write /output/results.parquet with:
option_id, price, delta, gamma, vega, theta.

### `/output/reward.json`

Written automatically by checks/test.sh — do not write this file.
""".strip()
    )

    options = pd.DataFrame(
        [
            {
                "option_id": "pair_call",
                "S": 100.0,
                "K": 100.0,
                "T": 1.0,
                "r": 0.05,
                "sigma": 0.20,
                "option_type": "call",
            },
            {
                "option_id": "pair_put",
                "S": 100.0,
                "K": 100.0,
                "T": 1.0,
                "r": 0.05,
                "sigma": 0.20,
                "option_type": "put",
            },
        ]
    )
    options.to_parquet(
        data / "options.parquet",
        index=False,
    )

    monkeypatch.setenv(
        "AGENT_WORK_ROOT",
        str(tmp_path / "work"),
    )

    skill_name = solve_offline(
        task_dir=task,
        out_dir=out,
        seed=42,
    )

    assert (
        skill_name
        == "black-scholes-greeks-current-results-parquet"
    )

    assert not (task / "checks").exists()
    assert (out / "results.parquet").exists()

    result = pd.read_parquet(
        out / "results.parquet"
    )

    assert list(result.columns) == [
        "option_id",
        "price",
        "delta",
        "gamma",
        "vega",
        "theta",
    ]

    assert result["option_id"].tolist() == [
        "pair_call",
        "pair_put",
    ]

    call = result.iloc[0]
    put = result.iloc[1]

    parity_rhs = (
        100.0
        - 100.0 * math.exp(-0.05)
    )

    assert abs(
        (call["price"] - put["price"])
        - parity_rhs
    ) < 1e-10

    assert 0.0 < call["delta"] < 1.0
    assert -1.0 < put["delta"] < 0.0
    assert (result["gamma"] >= 0.0).all()
    assert (result["vega"] >= 0.0).all()


def test_corporate_action_runs_without_checks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = tmp_path / "task"
    data = task / "environment" / "data"
    out = tmp_path / "output"

    data.mkdir(parents=True)

    (task / "instruction.md").write_text(
        """
# Task: Corporate Action Price Adjustment

Use prices.csv and corporate_actions.json.
Write adjusted_prices.csv and results.json.
""".strip()
    )

    pd.DataFrame(
        [
            {
                "date": "2023-01-02",
                "close": 100.0,
                "volume": 1000,
            },
            {
                "date": "2023-01-03",
                "close": 102.0,
                "volume": 1100,
            },
            {
                "date": "2023-01-04",
                "close": 25.0,
                "volume": 4400,
            },
            {
                "date": "2023-01-05",
                "close": 24.0,
                "volume": 4500,
            },
        ]
    ).to_csv(
        data / "prices.csv",
        index=False,
    )

    actions = {
        "corporate_actions": [
            {
                "date": "2023-01-04",
                "type": "split",
                "ratio": "4:1",
            },
            {
                "date": "2023-01-05",
                "type": "dividend",
                "amount": 1.0,
            },
        ]
    }

    (
        data / "corporate_actions.json"
    ).write_text(
        json.dumps(actions)
    )

    monkeypatch.setenv(
        "AGENT_WORK_ROOT",
        str(tmp_path / "work"),
    )

    skill_name = solve_offline(
        task_dir=task,
        out_dir=out,
        seed=42,
    )

    assert (
        skill_name
        == "corporate-action-price-adjustment"
    )

    assert not (task / "checks").exists()
    assert (
        out / "adjusted_prices.csv"
    ).exists()
    assert (out / "results.json").exists()

    adjusted = pd.read_csv(
        out / "adjusted_prices.csv"
    )
    results = json.loads(
        (
            out / "results.json"
        ).read_text()
    )

    assert len(adjusted) == 4
    assert (
        adjusted["close_adjusted"].iloc[-1]
        == adjusted["close_unadjusted"].iloc[-1]
    )

    assert (
        adjusted["close_adjusted"].iloc[0]
        < adjusted["close_unadjusted"].iloc[0] * 0.5
    )

    assert (
        adjusted["volume_adjusted"].iloc[0]
        == 4000
    )

    assert results["n_actions_applied"] == 2
    assert (
        results[
            "cumulative_price_adjustment_factor"
        ]
        < 0.5
    )
    assert (
        results[
            "cumulative_volume_adjustment_factor"
        ]
        == 4.0
    )
    assert (
        results[
            "total_dividends_per_original_share"
        ]
        == 1.0
    )
