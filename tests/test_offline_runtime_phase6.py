from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.offline_runtime import (
    solve_offline,
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

    instruction = """
# Black-Scholes Greeks Surface & PDE Verification
Compute Black-Scholes Greeks and verify the PDE.
Write greeks_surface.csv and pde_verification.csv.
Also write calibration.json and summary.json.
""".strip()

    (
        task
        / "instruction.md"
    ).write_text(
        instruction,
        encoding="utf-8",
    )

    steps = np.arange(
        63,
        dtype=float,
    )

    log_returns = (
        0.0002
        + 0.0135 * np.sin(steps)
    )

    prices = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(log_returns))))

    pd.DataFrame(
        {
            "Date": pd.date_range(
                "2026-01-01",
                periods=len(prices),
            ),
            "Close": prices,
        }
    ).to_csv(
        data
        / "prices.csv",
        index=False,
    )

    return task


def test_offline_black_scholes_skill_writes_contract(
    tmp_path: Path,
) -> None:
    task = _write_task(
        tmp_path
    )
    out = tmp_path / "out"

    skill = solve_offline(
        task_dir=task,
        out_dir=out,
        seed=42,
    )

    assert skill == (
        "black-scholes-greeks-pde"
    )

    expected = {
        "calibration.json",
        "greeks_surface.csv",
        "pde_verification.csv",
        "summary.json",
    }

    assert {
        path.name
        for path in out.iterdir()
        if path.is_file()
    } == expected

    greeks = pd.read_csv(
        out
        / "greeks_surface.csv"
    )
    pde = pd.read_csv(
        out
        / "pde_verification.csv"
    )
    summary = json.loads(
        (
            out
            / "summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert len(greeks) == 27
    assert len(pde) == 27

    assert (
        pde[
            "call_pde_residual"
        ].abs().max()
        < 1e-8
    )
    assert (
        pde[
            "put_pde_residual"
        ].abs().max()
        < 1e-8
    )
    assert (
        pde[
            "parity_error"
        ].abs().max()
        < 1e-8
    )

    assert summary[
        "delta_call_range_valid"
    ] is True
    assert summary[
        "gamma_positive_everywhere"
    ] is True
    assert summary[
        "vega_positive_everywhere"
    ] is True


def test_offline_runtime_rejects_unmatched_task(
    tmp_path: Path,
) -> None:
    task = tmp_path / "task"
    task.mkdir()

    (
        task
        / "instruction.md"
    ).write_text(
        "Compute an unrelated portfolio statistic.",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="No offline solver skill matched",
    ):
        solve_offline(
            task_dir=task,
            out_dir=tmp_path / "out",
            seed=42,
        )
