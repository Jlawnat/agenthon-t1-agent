from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_common.event_study import (
    EventStudySpec,
    build_event_records,
    corrado_rank_statistics,
    kolari_pynnonen_statistics,
    prepare_event_study_returns,
)
from agent.offline_event_study import (
    EventStudySkill,
)


def _fixture_frames() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    dates = pd.bdate_range(
        "2020-01-02",
        periods=260,
    )

    index = np.arange(
        len(
            dates
        ),
        dtype=float,
    )

    market_returns = (
        0.0002
        + 0.008
        * np.sin(
            index
            / 7.0
        )
    )

    market_prices = (
        100.0
        * np.exp(
            np.cumsum(
                market_returns
            )
        )
    )

    stock_a_returns = (
        0.0003
        + 1.2
        * market_returns
        + 0.003
        * np.sin(
            index
            / 3.0
        )
    )

    stock_b_returns = (
        -0.0001
        + 0.8
        * market_returns
        + 0.004
        * np.cos(
            index
            / 5.0
        )
    )

    stock_prices = pd.DataFrame(
        {
            "date": dates,
            "AAA": (
                80.0
                * np.exp(
                    np.cumsum(
                        stock_a_returns
                    )
                )
            ),
            "BBB": (
                120.0
                * np.exp(
                    np.cumsum(
                        stock_b_returns
                    )
                )
            ),
        }
    )

    market_frame = pd.DataFrame(
        {
            "date": dates,
            "adj_close": (
                market_prices
            ),
        }
    )

    events = pd.DataFrame(
        {
            "ticker": [
                "AAA",
                "BBB",
                "AAA",
            ],
            "event_date": [
                dates[
                    160
                ],
                dates[
                    180
                ],
                dates[
                    220
                ],
            ],
        }
    )

    return (
        stock_prices,
        market_frame,
        events,
    )


def test_generic_event_study_primitives() -> None:
    (
        stocks,
        market,
        events,
    ) = _fixture_frames()

    merged, _ = (
        prepare_event_study_returns(
            stocks,
            market,
        )
    )

    records = build_event_records(
        merged,
        events,
        spec=EventStudySpec(),
    )

    assert len(
        records
    ) == 3

    assert all(
        60
        <= record.n_obs
        <= 120
        for record
        in records
    )

    assert all(
        len(
            record.event_abnormal_returns
        )
        == 11
        for record
        in records
    )

    corrado_full, corrado_day0 = (
        corrado_rank_statistics(
            records
        )
    )

    rho_bar, kp_full, kp_day0 = (
        kolari_pynnonen_statistics(
            records
        )
    )

    for value in (
        corrado_full,
        corrado_day0,
        rho_bar,
        kp_full,
        kp_day0,
    ):
        assert np.isfinite(
            value
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
Conduct an event study using a market model and abnormal returns.
Compute CAR and CAAR. Include the Corrado rank test and
Kolari-Pynnonen adjustment. Write market_model_params.csv,
abnormal_returns.csv, car_per_event.csv, caar_summary.json,
and statistical_tests.json.
""".strip(),
        encoding="utf-8",
    )

    (
        stocks,
        market,
        events,
    ) = _fixture_frames()

    stocks.to_csv(
        data
        / "stocks.csv",
        index=False,
    )

    market.to_csv(
        data
        / "market.csv",
        index=False,
    )

    events.to_csv(
        data
        / "events.csv",
        index=False,
    )

    return task


def test_generic_event_study_skill(
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

    skill = EventStudySkill()

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
        "market_model_params.csv",
        "abnormal_returns.csv",
        "car_per_event.csv",
        "caar_summary.json",
        "statistical_tests.json",
    }

    params = pd.read_csv(
        out
        / "market_model_params.csv"
    )

    abnormal = pd.read_csv(
        out
        / "abnormal_returns.csv"
    )

    assert len(
        params
    ) == 3

    assert len(
        abnormal
    ) == 33

    assert set(
        abnormal[
            "relative_day"
        ]
    ) == set(
        range(
            -5,
            6,
        )
    )
