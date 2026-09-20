from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_cir import (
    CirBondPricingSkill,
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
# CIR Short-Rate Model
Use the Cox-Ingersoll-Ross model for zero-coupon bonds.
Check the Feller condition and write bond_prices.csv
and forward_rates.csv from DFF data.
""".strip(),
        encoding="utf-8",
    )

    dates = pd.date_range(
        "2016-01-01",
        periods=360,
        freq="D",
    )

    x = np.arange(
        len(dates),
        dtype=float,
    )

    rates_decimal = (
        0.035
        + 0.006
        * np.sin(
            x
            / 25.0
        )
        + 0.002
        * np.sin(
            x
            / 7.0
        )
    )

    rates_decimal = np.maximum(
        rates_decimal,
        0.005,
    )

    frame = pd.DataFrame(
        {
            "date": dates,
            "DFF": rates_decimal * 100.0,
            "DTB3": 4.0,
            "DGS1": 4.1,
            "DGS2": 4.2,
            "DGS5": 4.3,
            "DGS10": 4.4,
            "DGS30": 4.5,
        }
    )

    frame.to_csv(
        data_dir
        / "fred_rates.csv",
        index=False,
    )

    return task


def test_cir_skill_writes_consistent_outputs(
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

    skill = CirBondPricingSkill()

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
        "calibration.json",
        "bond_prices.csv",
        "forward_rates.csv",
        "summary.json",
    }

    assert {
        path.name
        for path in out.iterdir()
        if path.is_file()
    } == expected

    calibration = json.loads(
        (
            out
            / "calibration.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        0.01
        <= calibration["kappa"]
        <= 1.0
    )

    assert (
        0.001
        <= calibration["theta"]
        <= 0.10
    )

    assert (
        0.001
        <= calibration["sigma"]
        <= 0.20
    )

    assert np.isfinite(
        calibration["log_likelihood"]
    )

    bonds = pd.read_csv(
        out
        / "bond_prices.csv"
    )

    assert len(bonds) == 8

    assert (
        bonds["zcb_price"] > 0.0
    ).all()

    assert (
        bonds["zcb_price"]
        <= 1.0 + 1e-12
    ).all()

    assert np.all(
        np.diff(
            bonds["zcb_price"].to_numpy()
        )
        <= 1e-6
    )

    expected_zero = -np.log(
        bonds["zcb_price"]
    )

    assert np.allclose(
        bonds["zero_rate"],
        expected_zero,
    )

    assert np.allclose(
        bonds["model_yield"],
        expected_zero
        / bonds["tau"],
    )

    forwards = pd.read_csv(
        out
        / "forward_rates.csv"
    )

    assert len(forwards) == 7

    assert (
        forwards["forward_rate"] > 0.0
    ).all()

    summary = json.loads(
        (
            out
            / "summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert np.isclose(
        summary["kappa"],
        calibration["kappa"],
    )

    assert isinstance(
        summary["yield_curve_upward_sloping"],
        bool,
    )
