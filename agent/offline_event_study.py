from __future__ import annotations

from dataclasses import dataclass
import json
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


def _discover_inputs(
    task_dir: Path,
) -> tuple[
    Path,
    Path,
    Path,
]:
    events_path = None
    stock_path = None
    market_path = None

    for path in sorted(
        task_dir.rglob(
            "*.csv"
        )
    ):
        if "checks" in path.parts:
            continue

        try:
            sample = pd.read_csv(
                path,
                nrows=5,
            )
        except Exception:
            continue

        lower = {
            str(column).lower()
            for column
            in sample.columns
        }

        if {
            "ticker",
            "event_date",
        }.issubset(
            lower
        ):
            events_path = path
            continue

        if "date" not in lower:
            continue

        non_date = [
            str(column)
            for column
            in sample.columns
            if str(
                column
            ).lower()
            != "date"
        ]

        if len(
            non_date
        ) >= 2:
            stock_path = path
        elif len(
            non_date
        ) == 1:
            market_path = path

    if (
        events_path is None
        or stock_path is None
        or market_path is None
    ):
        raise RuntimeError(
            "Could not discover event-study input files."
        )

    return (
        events_path,
        stock_path,
        market_path,
    )


def _r8(
    value: float,
) -> float:
    return float(
        round(
            float(
                value
            ),
            8,
        )
    )


@dataclass(frozen=True)
class EventStudySkill:
    """Generic deterministic market-model event-study handler."""

    name: str = "event-study-domain"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        lowered = (
            instruction.lower()
        )

        semantic_markers = (
            "event study",
            "market model",
            "abnormal returns",
            "corrado",
            "kolari",
            "caar",
        )

        output_markers = (
            "market_model_params.csv",
            "abnormal_returns.csv",
            "car_per_event.csv",
            "statistical_tests.json",
        )

        return (
            all(
                marker in lowered
                for marker
                in semantic_markers
            )
            and all(
                marker in lowered
                for marker
                in output_markers
            )
        )

    def solve(
        self,
        *,
        instruction: str,
        task_dir: Path,
        out_dir: Path,
        seed: int,
    ) -> None:
        del instruction, seed

        (
            events_path,
            stock_path,
            market_path,
        ) = _discover_inputs(
            task_dir
        )

        events = pd.read_csv(
            events_path
        )
        stocks = pd.read_csv(
            stock_path
        )
        market = pd.read_csv(
            market_path
        )

        (
            merged,
            _,
        ) = prepare_event_study_returns(
            stocks,
            market,
        )

        records = build_event_records(
            merged,
            events,
            spec=EventStudySpec(),
        )

        if not records:
            raise RuntimeError(
                "Event-study engine found no valid events."
            )

        parameter_rows = []
        abnormal_rows = []
        car_rows = []

        for record in records:
            parameter_rows.append(
                {
                    "ticker": (
                        record.ticker
                    ),
                    "event_date": (
                        record.event_date
                    ),
                    "alpha": _r8(
                        record.alpha
                    ),
                    "beta": _r8(
                        record.beta
                    ),
                    "r_squared": _r8(
                        record.r_squared
                    ),
                    "n_obs": int(
                        record.n_obs
                    ),
                }
            )

            for (
                relative_day,
                date,
                abnormal_return,
            ) in zip(
                record.event_relative_days,
                record.event_dates,
                record.event_abnormal_returns,
            ):
                abnormal_rows.append(
                    {
                        "ticker": (
                            record.ticker
                        ),
                        "event_date": (
                            record.event_date
                        ),
                        "relative_day": int(
                            relative_day
                        ),
                        "date": date,
                        "ar": _r8(
                            abnormal_return
                        ),
                    }
                )

            relative = (
                record.event_relative_days
            )
            abnormal = (
                record.event_abnormal_returns
            )

            pre_mask = (
                relative < 0
            )
            event_mask = (
                relative == 0
            )
            post_mask = (
                relative > 0
            )

            car_rows.append(
                {
                    "ticker": (
                        record.ticker
                    ),
                    "event_date": (
                        record.event_date
                    ),
                    "car_full": _r8(
                        np.sum(
                            abnormal
                        )
                    ),
                    "car_pre": _r8(
                        np.sum(
                            abnormal[
                                pre_mask
                            ]
                        )
                    ),
                    "car_event": _r8(
                        np.sum(
                            abnormal[
                                event_mask
                            ]
                        )
                    ),
                    "car_post": _r8(
                        np.sum(
                            abnormal[
                                post_mask
                            ]
                        )
                    ),
                }
            )

        abnormal_frame = pd.DataFrame(
            abnormal_rows
        )

        caar_by_day = {}

        for relative_day in range(
            -5,
            6,
        ):
            values = abnormal_frame.loc[
                abnormal_frame[
                    "relative_day"
                ]
                == relative_day,
                "ar",
            ].to_numpy(
                dtype=float
            )

            caar_by_day[
                str(
                    relative_day
                )
            ] = _r8(
                np.mean(
                    values
                )
            )

        caar_pre = float(
            sum(
                caar_by_day[
                    str(
                        day
                    )
                ]
                for day
                in range(
                    -5,
                    0,
                )
            )
        )

        caar_post = float(
            sum(
                caar_by_day[
                    str(
                        day
                    )
                ]
                for day
                in range(
                    1,
                    6,
                )
            )
        )

        caar_full = float(
            sum(
                caar_by_day.values()
            )
        )

        caar_summary = {
            "num_events": int(
                len(
                    records
                )
            ),
            "caar_by_day": (
                caar_by_day
            ),
            "caar_full": _r8(
                caar_full
            ),
            "caar_pre": _r8(
                caar_pre
            ),
            "caar_post": _r8(
                caar_post
            ),
        }

        (
            corrado_full,
            corrado_day0,
        ) = corrado_rank_statistics(
            records
        )

        (
            rho_bar,
            kp_full,
            kp_day0,
        ) = kolari_pynnonen_statistics(
            records
        )

        statistical_tests = {
            "corrado_z_full": _r8(
                corrado_full
            ),
            "corrado_z_day0": _r8(
                corrado_day0
            ),
            "kp_rho_bar": _r8(
                rho_bar
            ),
            "kp_t_full": _r8(
                kp_full
            ),
            "kp_t_day0": _r8(
                kp_day0
            ),
        }

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            parameter_rows,
            columns=[
                "ticker",
                "event_date",
                "alpha",
                "beta",
                "r_squared",
                "n_obs",
            ],
        ).to_csv(
            out_dir
            / "market_model_params.csv",
            index=False,
        )

        abnormal_frame[
            [
                "ticker",
                "event_date",
                "relative_day",
                "date",
                "ar",
            ]
        ].to_csv(
            out_dir
            / "abnormal_returns.csv",
            index=False,
        )

        pd.DataFrame(
            car_rows,
            columns=[
                "ticker",
                "event_date",
                "car_full",
                "car_pre",
                "car_event",
                "car_post",
            ],
        ).to_csv(
            out_dir
            / "car_per_event.csv",
            index=False,
        )

        (
            out_dir
            / "caar_summary.json"
        ).write_text(
            json.dumps(
                caar_summary,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        (
            out_dir
            / "statistical_tests.json"
        ).write_text(
            json.dumps(
                statistical_tests,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
