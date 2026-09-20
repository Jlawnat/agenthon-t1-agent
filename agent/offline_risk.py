from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from agent.offline_common.risk import (
    age_weighted_var_es,
    aligned_log_returns,
    conditional_normal_var_es,
    expanding_historical_backtest,
    historical_var_es,
    normal_var_es,
    overlapping_horizon_returns,
    parse_risk_spec,
    student_t_var_es,
    ewma_portfolio_volatility,
)


def _find_market_risk_csv(
    task_dir: Path,
) -> Path:
    candidates: list[
        Path
    ] = []

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
            "date",
            "symbol",
            "close",
        }.issubset(
            lower
        ):
            candidates.append(
                path
            )

    if not candidates:
        raise RuntimeError(
            "No date/symbol/close market-risk CSV was found."
        )

    return candidates[
        0
    ]


@dataclass(frozen=True)
class MarketRiskSkill:
    """Generic deterministic market-risk domain handler."""

    name: str = "market-risk-domain"

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

        required = (
            "value-at-risk",
            "expected shortfall",
            "historical simulation",
            "student-t",
            "ewma",
            "kupiec",
        )

        outputs = (
            "var_1day.csv",
            "var_10day.csv",
            "backtest.json",
        )

        return (
            all(
                token in lowered
                for token
                in required
            )
            and all(
                token in lowered
                for token
                in outputs
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
        del seed

        spec = (
            parse_risk_spec(
                instruction
            )
        )

        path = (
            _find_market_risk_csv(
                task_dir
            )
        )

        frame = pd.read_csv(
            path
        )

        by_lower = {
            str(column).lower(): str(column)
            for column
            in frame.columns
        }

        (
            prices,
            asset_returns,
            counts,
        ) = aligned_log_returns(
            frame,
            date_column=(
                by_lower[
                    "date"
                ]
            ),
            symbol_column=(
                by_lower[
                    "symbol"
                ]
            ),
            close_column=(
                by_lower[
                    "close"
                ]
            ),
        )

        symbols = list(
            asset_returns.columns
        )

        equal_weights = np.full(
            len(
                symbols
            ),
            1.0
            / len(
                symbols
            ),
            dtype=float,
        )

        return_matrix = (
            asset_returns.to_numpy(
                dtype=float
            )
        )

        portfolio_returns = (
            return_matrix
            @ equal_weights
        )

        portfolio_mean = float(
            np.mean(
                portfolio_returns
            )
        )
        portfolio_std = float(
            np.std(
                portfolio_returns,
                ddof=1,
            )
        )

        if len(
            symbols
        ) == 2:
            correlation = float(
                np.corrcoef(
                    return_matrix[
                        :,
                        0
                    ],
                    return_matrix[
                        :,
                        1
                    ],
                )[
                    0,
                    1,
                ]
            )
        else:
            correlation = float(
                np.mean(
                    np.corrcoef(
                        return_matrix,
                        rowvar=False,
                    )[
                        np.triu_indices(
                            len(
                                symbols
                            ),
                            k=1,
                        )
                    ]
                )
            )

        data_summary = {
            f"n_{symbols[0].lower()}": int(
                counts[
                    symbols[
                        0
                    ]
                ]
            ),
            f"n_{symbols[1].lower()}": int(
                counts[
                    symbols[
                        1
                    ]
                ]
            ),
            "n_aligned": int(
                len(
                    prices
                )
            ),
            "n_returns": int(
                len(
                    portfolio_returns
                )
            ),
            "portfolio_mean": (
                portfolio_mean
            ),
            "portfolio_std": (
                portfolio_std
            ),
            "correlation": (
                correlation
            ),
        }

        ewma_daily_vol = (
            ewma_portfolio_volatility(
                return_matrix,
                decay=(
                    spec.ewma_lambda
                ),
                weights=equal_weights,
            )
        )

        one_day_rows = []
        ten_day_rows = []

        horizon_returns = (
            overlapping_horizon_returns(
                portfolio_returns,
                horizon=(
                    spec.horizon_days
                ),
            )
        )

        methods = (
            "HS",
            "Normal",
            "StudentT",
            "AgeWeightedHS",
            "EWMA",
        )

        one_day_lookup: dict[
            tuple[
                str,
                float,
            ],
            tuple[
                float,
                float,
            ],
        ] = {}

        for alpha in spec.confidence_levels:
            hs = historical_var_es(
                portfolio_returns,
                alpha=alpha,
                notional=(
                    spec.notional
                ),
            )

            normal = normal_var_es(
                portfolio_returns,
                alpha=alpha,
                notional=(
                    spec.notional
                ),
            )

            student = student_t_var_es(
                portfolio_returns,
                alpha=alpha,
                notional=(
                    spec.notional
                ),
            )

            age = age_weighted_var_es(
                portfolio_returns,
                alpha=alpha,
                notional=(
                    spec.notional
                ),
                decay=(
                    spec.age_lambda
                ),
            )

            ewma = conditional_normal_var_es(
                ewma_daily_vol,
                alpha=alpha,
                notional=(
                    spec.notional
                ),
            )

            method_values = {
                "HS": hs,
                "Normal": normal,
                "StudentT": student,
                "AgeWeightedHS": age,
                "EWMA": ewma,
            }

            for method in methods:
                var_value, es_value = (
                    method_values[
                        method
                    ]
                )

                one_day_lookup[
                    (
                        method,
                        float(
                            alpha
                        ),
                    )
                ] = (
                    float(
                        var_value
                    ),
                    float(
                        es_value
                    ),
                )

                one_day_rows.append(
                    {
                        "method": (
                            method
                        ),
                        "alpha": float(
                            alpha
                        ),
                        "var_dollar": float(
                            var_value
                        ),
                        "es_dollar": float(
                            es_value
                        ),
                    }
                )

            horizon_hs = historical_var_es(
                horizon_returns,
                alpha=alpha,
                notional=(
                    spec.notional
                ),
            )

            horizon_age = age_weighted_var_es(
                horizon_returns,
                alpha=alpha,
                notional=(
                    spec.notional
                ),
                decay=(
                    spec.age_lambda
                ),
            )

            square_root_horizon = math.sqrt(
                float(
                    spec.horizon_days
                )
            )

            horizon_values = {
                "HS": horizon_hs,
                "AgeWeightedHS": horizon_age,
                "Normal": tuple(
                    value
                    * square_root_horizon
                    for value
                    in normal
                ),
                "StudentT": tuple(
                    value
                    * square_root_horizon
                    for value
                    in student
                ),
                "EWMA": tuple(
                    value
                    * square_root_horizon
                    for value
                    in ewma
                ),
            }

            for method in methods:
                var_value, es_value = (
                    horizon_values[
                        method
                    ]
                )

                ten_day_rows.append(
                    {
                        "method": (
                            method
                        ),
                        "alpha": float(
                            alpha
                        ),
                        "var_dollar": float(
                            var_value
                        ),
                        "es_dollar": float(
                            es_value
                        ),
                    }
                )

        backtest_alpha = max(
            spec.confidence_levels
        )

        (
            n_test_days,
            n_exceedances,
            exceedance_rate,
            kupiec_pvalue,
        ) = expanding_historical_backtest(
            portfolio_returns,
            alpha=backtest_alpha,
            minimum_observations=(
                spec.backtest_min_obs
            ),
        )

        backtest = {
            "n_test_days": (
                n_test_days
            ),
            "n_exceedances": (
                n_exceedances
            ),
            "exceedance_rate": (
                exceedance_rate
            ),
            "expected_rate": float(
                1.0
                - backtest_alpha
            ),
            "kupiec_pvalue": (
                kupiec_pvalue
            ),
        }

        summary_alpha = max(
            spec.confidence_levels
        )

        comparison = [
            (
                method,
                one_day_lookup[
                    (
                        method,
                        float(
                            summary_alpha
                        ),
                    )
                ][
                    0
                ],
            )
            for method in methods
        ]

        comparison.sort(
            key=lambda item: (
                item[
                    1
                ],
                item[
                    0
                ],
            )
        )

        summary = {
            "most_conservative_method": (
                comparison[
                    -1
                ][
                    0
                ]
            ),
            "least_conservative_method": (
                comparison[
                    0
                ][
                    0
                ]
            ),
            "var_range_99": [
                float(
                    comparison[
                        0
                    ][
                        1
                    ]
                ),
                float(
                    comparison[
                        -1
                    ][
                        1
                    ]
                ),
            ],
            "ewma_current_vol": float(
                ewma_daily_vol
                * math.sqrt(
                    252.0
                )
            ),
        }

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            out_dir
            / "data_summary.json"
        ).write_text(
            json.dumps(
                data_summary,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        pd.DataFrame(
            one_day_rows,
            columns=[
                "method",
                "alpha",
                "var_dollar",
                "es_dollar",
            ],
        ).to_csv(
            out_dir
            / "var_1day.csv",
            index=False,
        )

        pd.DataFrame(
            ten_day_rows,
            columns=[
                "method",
                "alpha",
                "var_dollar",
                "es_dollar",
            ],
        ).to_csv(
            out_dir
            / "var_10day.csv",
            index=False,
        )

        (
            out_dir
            / "backtest.json"
        ).write_text(
            json.dumps(
                backtest,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        (
            out_dir
            / "summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
