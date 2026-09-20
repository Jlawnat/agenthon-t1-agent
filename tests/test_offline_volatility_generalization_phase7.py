from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_common.volatility import (
    annualized_volatility,
    efficiency_ratios,
    estimator_arrays,
    ohlc_variance_estimators,
    rolling_ohlc_estimators,
    validate_ohlc,
)
from agent.offline_volatility import (
    OhlcVolatilitySkill,
)


def _make_ohlc(
    periods: int = 90,
) -> pd.DataFrame:
    dates = pd.bdate_range(
        "2024-01-02",
        periods=periods,
    )

    index = np.arange(
        periods,
        dtype=float,
    )

    close = (
        100.0
        * np.exp(
            0.0004
            * index
            + 0.01
            * np.sin(
                index
                / 5.0
            )
        )
    )

    open_ = close * (
        1.0
        + 0.002
        * np.sin(
            index
            / 3.0
        )
    )

    high = np.maximum(
        open_,
        close,
    ) * 1.012

    low = np.minimum(
        open_,
        close,
    ) * 0.988

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000,
        }
    )


def test_generic_ohlc_primitives() -> None:
    frame = _make_ohlc()

    assert validate_ohlc(
        frame
    )

    full = (
        ohlc_variance_estimators(
            frame
        )
    )

    for value in (
        full.close_to_close,
        full.parkinson,
        full.garman_klass,
        full.rogers_satchell,
        full.yang_zhang,
    ):
        assert value >= 0.0

    assert annualized_volatility(
        full.close_to_close
    ) > 0.0

    rolling = (
        rolling_ohlc_estimators(
            frame,
            window=21,
        )
    )

    assert len(
        rolling
    ) == len(
        frame
    ) - 21

    arrays = estimator_arrays(
        rolling
    )

    ratios = efficiency_ratios(
        arrays
    )

    assert ratios[
        "close_to_close"
    ] == 1.0

    assert set(
        ratios
    ) == {
        "close_to_close",
        "parkinson",
        "garman_klass",
        "rogers_satchell",
        "yang_zhang",
    }


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
Compute OHLC realized volatility estimators using Parkinson,
Garman-Klass, Rogers-Satchell, and Yang-Zhang methods.
Include rolling analysis and efficiency comparison.
Write calibration.json, full_sample_vol.json,
rolling_vol_stats.csv, vol_term_structure.csv, and summary.json.
""".strip(),
        encoding="utf-8",
    )

    _make_ohlc(
        120
    ).to_csv(
        data
        / "market_daily.csv",
        index=False,
    )

    return task


def test_generic_ohlc_volatility_skill(
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
        OhlcVolatilitySkill()
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

    assert {
        path.name
        for path in out.iterdir()
    } == {
        "calibration.json",
        "full_sample_vol.json",
        "rolling_vol_stats.csv",
        "vol_term_structure.csv",
        "summary.json",
    }

    stats = pd.read_csv(
        out
        / "rolling_vol_stats.csv"
    )

    assert len(
        stats
    ) == 5

    assert set(
        stats[
            "estimator"
        ]
    ) == {
        "close_to_close",
        "parkinson",
        "garman_klass",
        "rogers_satchell",
        "yang_zhang",
    }

    term = pd.read_csv(
        out
        / "vol_term_structure.csv"
    )

    assert set(
        term[
            "window"
        ]
    ) == {
        5,
        10,
        21,
        63,
        126,
        252,
    }
