from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_brinson import (
    BrinsonSectorAttributionSkill,
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
# Brinson-Fachler Sector Attribution
Compute allocation, selection, and interaction using
benchmark_weights and portfolio_etfs.
Return weight_snapshot_march.
""".strip(),
        encoding="utf-8",
    )

    sectors = (
        "XLK",
        "XLF",
    )

    params = {
        "benchmark_weights": {
            "Q1": {
                "XLK": 0.6,
                "XLF": 0.4,
            },
            "Q2": {
                "XLK": 0.6,
                "XLF": 0.4,
            },
            "Q3": {
                "XLK": 0.6,
                "XLF": 0.4,
            },
            "Q4": {
                "XLK": 0.6,
                "XLF": 0.4,
            },
        },
        "portfolio_weights": {
            "XLK": 0.5,
            "XLF": 0.4,
            "CASH": 0.1,
        },
        "portfolio_etfs": {
            "XLK": "XLK",
            "XLF": "XLF",
        },
    }

    (
        data_dir
        / "params.json"
    ).write_text(
        json.dumps(
            params,
            indent=2,
        ),
        encoding="utf-8",
    )

    months = pd.date_range(
        "2023-12-31",
        periods=13,
        freq="ME",
    )

    xlks = 100.0 * (
        1.01
        ** np.arange(
            len(months)
        )
    )
    xlfs = 100.0 * (
        1.005
        ** np.arange(
            len(months)
        )
    )

    pd.DataFrame(
        {
            "Date": months,
            "XLK": xlks,
            "XLF": xlfs,
        }
    ).to_csv(
        data_dir
        / "sector_etfs.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "month": [
                f"2024-{month:02d}"
                for month
                in range(
                    1,
                    13,
                )
            ],
            "annual_rate": [
                0.048
            ]
            * 12,
        }
    ).to_csv(
        data_dir
        / "cash_rates.csv",
        index=False,
    )

    return task


def test_brinson_skill_writes_consistent_results(
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
        BrinsonSectorAttributionSkill()
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

    result = json.loads(
        (
            out
            / "results.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert len(
        result[
            "monthly_allocation"
        ]
    ) == 12

    assert len(
        result[
            "monthly_selection"
        ]
    ) == 12

    assert len(
        result[
            "monthly_interaction"
        ]
    ) == 12

    assert np.isclose(
        sum(
            result[
                "monthly_allocation"
            ]
        ),
        result[
            "total_allocation_effect"
        ],
    )

    assert np.isclose(
        sum(
            result[
                "monthly_selection"
            ]
        ),
        result[
            "total_selection_effect"
        ],
    )

    assert np.isclose(
        sum(
            result[
                "monthly_interaction"
            ]
        ),
        result[
            "total_interaction_effect"
        ],
    )

    assert np.isclose(
        result[
            "q1_allocation"
        ]
        + result[
            "q2_allocation"
        ]
        + result[
            "q3_allocation"
        ]
        + result[
            "q4_allocation"
        ],
        result[
            "total_allocation_effect"
        ],
    )

    # Same ETFs on portfolio and benchmark sides means no
    # selection or interaction effects in this fixture.
    assert np.isclose(
        result[
            "total_selection_effect"
        ],
        0.0,
    )

    assert np.isclose(
        result[
            "total_interaction_effect"
        ],
        0.0,
    )

    assert set(
        result[
            "weight_snapshot_march"
        ]
    ) == {
        "XLK",
        "XLF",
        "CASH",
    }

    assert np.isclose(
        sum(
            result[
                "weight_snapshot_march"
            ].values()
        ),
        1.0,
    )

    assert len(
        result[
            "monthly_sector_allocation"
        ][
            "XLK"
        ]
    ) == 12
