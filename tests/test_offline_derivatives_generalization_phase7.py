from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_common.derivatives import (
    kirk_spread_price,
    margrabe_price,
)
from agent.offline_derivatives import (
    TwoAssetDerivativesSkill,
)


def test_kirk_k0_matches_margrabe() -> None:
    margrabe, _ = margrabe_price(
        S1=120.0,
        S2=100.0,
        sigma1=0.25,
        sigma2=0.20,
        rho=0.4,
        T=0.5,
        q1=0.01,
        q2=0.01,
    )

    kirk = kirk_spread_price(
        S1=120.0,
        S2=100.0,
        K=0.0,
        sigma1=0.25,
        sigma2=0.20,
        rho=0.4,
        T=0.5,
        r=0.05,
        q1=0.01,
        q2=0.01,
    )

    assert np.isclose(
        margrabe,
        kirk,
        rtol=1e-10,
        atol=1e-10,
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
Calibrate two correlated assets and price Margrabe exchange
options and Kirk spread options. Include correlation
sensitivity and Monte Carlo verification using 20000 correlated
paths. Write calibration.json, margrabe_prices.csv,
kirk_prices.csv, correlation_sensitivity.csv, and summary.json.
""".strip(),
        encoding="utf-8",
    )

    dates = pd.bdate_range(
        "2024-01-02",
        periods=120,
    )

    x = np.arange(
        len(
            dates
        ),
        dtype=float,
    )

    first_close = (
        100.0
        * np.exp(
            0.0005
            * x
            + 0.02
            * np.sin(
                x
                / 6.0
            )
        )
    )

    second_close = (
        95.0
        * np.exp(
            0.0004
            * x
            + 0.018
            * np.sin(
                x
                / 6.0
                + 0.5
            )
        )
    )

    for name, close in (
        (
            "asset1.csv",
            first_close,
        ),
        (
            "asset2.csv",
            second_close,
        ),
    ):
        pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": (
                    close
                    * 1.01
                ),
                "low": (
                    close
                    * 0.99
                ),
                "close": close,
                "volume": 1000,
            }
        ).to_csv(
            data
            / name,
            index=False,
        )

    return task


def test_generic_two_asset_derivatives_skill(
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
        TwoAssetDerivativesSkill()
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
        "calibration.json",
        "margrabe_prices.csv",
        "kirk_prices.csv",
        "correlation_sensitivity.csv",
        "summary.json",
    }

    assert {
        path.name
        for path in out.iterdir()
    } == expected

    kirk = pd.read_csv(
        out
        / "kirk_prices.csv"
    )

    assert len(
        kirk
    ) == 15

    assert (
        kirk[
            "kirk_price"
        ]
        >= 0.0
    ).all()

    assert (
        kirk[
            "mc_std_err"
        ]
        >= 0.0
    ).all()
