from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd


def _find_named_file(
    task_dir: Path,
    filename: str,
) -> Path:
    matches = [
        path
        for path in task_dir.rglob(filename)
        if (
            path.is_file()
            and "checks" not in path.parts
        )
    ]

    if not matches:
        raise RuntimeError(
            f"Offline Brinson skill could not find {filename}."
        )

    return sorted(
        matches
    )[0]


def _quarter_for_month(
    month_index: int,
) -> str:
    return (
        "Q1",
        "Q1",
        "Q1",
        "Q2",
        "Q2",
        "Q2",
        "Q3",
        "Q3",
        "Q3",
        "Q4",
        "Q4",
        "Q4",
    )[month_index]


def _compound(
    returns: list[float],
) -> float:
    value = 1.0

    for result in returns:
        value *= (
            1.0
            + float(result)
        )

    return float(
        value - 1.0
    )


@dataclass(frozen=True)
class BrinsonSectorAttributionSkill:
    name: str = "brinson-sector-attribution"

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
            "brinson-fachler",
            "allocation",
            "selection",
            "interaction",
            "benchmark_weights",
            "portfolio_etfs",
            "weight_snapshot_march",
        )

        return all(
            token in lowered
            for token in required
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

        params_path = (
            _find_named_file(
                task_dir,
                "params.json",
            )
        )
        prices_path = (
            _find_named_file(
                task_dir,
                "sector_etfs.csv",
            )
        )
        cash_path = (
            _find_named_file(
                task_dir,
                "cash_rates.csv",
            )
        )

        params = json.loads(
            params_path.read_text(
                encoding="utf-8"
            )
        )

        benchmark_weights = (
            params.get(
                "benchmark_weights"
            )
        )
        target_weights = (
            params.get(
                "portfolio_weights"
            )
        )
        portfolio_etfs = (
            params.get(
                "portfolio_etfs"
            )
        )

        if not isinstance(
            benchmark_weights,
            dict,
        ):
            raise RuntimeError(
                "benchmark_weights is missing or invalid."
            )

        if not isinstance(
            target_weights,
            dict,
        ):
            raise RuntimeError(
                "portfolio_weights is missing or invalid."
            )

        if not isinstance(
            portfolio_etfs,
            dict,
        ):
            raise RuntimeError(
                "portfolio_etfs is missing or invalid."
            )

        sectors = list(
            portfolio_etfs.keys()
        )

        if not sectors:
            raise RuntimeError(
                "portfolio_etfs contains no sectors."
            )

        if "CASH" not in target_weights:
            raise RuntimeError(
                "portfolio_weights must include CASH."
            )

        prices = pd.read_csv(
            prices_path
        )

        date_column = None

        for candidate in (
            "Date",
            "date",
        ):
            if candidate in prices.columns:
                date_column = candidate
                break

        if date_column is None:
            raise RuntimeError(
                "sector_etfs.csv requires a Date column."
            )

        prices[date_column] = (
            pd.to_datetime(
                prices[
                    date_column
                ],
                errors="raise",
            )
        )

        prices = prices.sort_values(
            date_column,
            kind="stable",
        ).reset_index(
            drop=True
        )

        needed_tickers = (
            set(sectors)
            | set(
                str(value)
                for value
                in portfolio_etfs.values()
            )
        )

        missing_tickers = sorted(
            ticker
            for ticker in needed_tickers
            if ticker not in prices.columns
        )

        if missing_tickers:
            raise RuntimeError(
                "sector_etfs.csv is missing ticker(s): "
                + ", ".join(
                    missing_tickers
                )
            )

        prices[
            "month"
        ] = (
            prices[
                date_column
            ]
            .dt.to_period(
                "M"
            )
            .astype(str)
        )

        monthly_ends = (
            prices
            .groupby(
                "month",
                sort=True,
            )
            .tail(1)
            .set_index(
                "month"
            )
        )

        months = [
            f"2024-{month:02d}"
            for month
            in range(
                1,
                13,
            )
        ]

        previous_months = [
            "2023-12",
            *months[:-1],
        ]

        required_months = (
            set(months)
            | set(
                previous_months
            )
        )

        missing_months = sorted(
            required_months
            - set(
                monthly_ends.index
            )
        )

        if missing_months:
            raise RuntimeError(
                "sector_etfs.csv is missing required month-end data: "
                + ", ".join(
                    missing_months
                )
            )

        monthly_returns: dict[
            str,
            list[float],
        ] = {}

        for ticker in sorted(
            needed_tickers
        ):
            values: list[
                float
            ] = []

            for month, previous in zip(
                months,
                previous_months,
            ):
                current_price = float(
                    monthly_ends.loc[
                        month,
                        ticker,
                    ]
                )
                previous_price = float(
                    monthly_ends.loc[
                        previous,
                        ticker,
                    ]
                )

                if previous_price == 0.0:
                    raise RuntimeError(
                        f"Zero prior month-end price for {ticker}."
                    )

                values.append(
                    (
                        current_price
                        / previous_price
                    )
                    - 1.0
                )

            monthly_returns[
                ticker
            ] = values

        cash_frame = pd.read_csv(
            cash_path
        )

        required_cash_columns = {
            "month",
            "annual_rate",
        }

        if not required_cash_columns.issubset(
            cash_frame.columns
        ):
            raise RuntimeError(
                "cash_rates.csv requires month and annual_rate."
            )

        cash_rates = {
            str(row["month"]): float(
                row[
                    "annual_rate"
                ]
            )
            for _, row
            in cash_frame.iterrows()
        }

        missing_cash_months = [
            month
            for month in months
            if month not in cash_rates
        ]

        if missing_cash_months:
            raise RuntimeError(
                "cash_rates.csv is missing month(s): "
                + ", ".join(
                    missing_cash_months
                )
            )

        portfolio_weights = {
            str(key): float(value)
            for key, value
            in target_weights.items()
        }

        monthly_portfolio_returns: list[
            float
        ] = []
        monthly_benchmark_returns: list[
            float
        ] = []

        monthly_allocation: list[
            float
        ] = []
        monthly_selection: list[
            float
        ] = []
        monthly_interaction: list[
            float
        ] = []

        sector_allocation = {
            sector: 0.0
            for sector
            in sectors
        }
        sector_allocation[
            "CASH"
        ] = 0.0

        sector_selection = {
            sector: 0.0
            for sector
            in sectors
        }

        sector_interaction = {
            sector: 0.0
            for sector
            in sectors
        }

        monthly_sector_allocation = {
            sector: []
            for sector
            in sectors
        }
        monthly_sector_allocation[
            "CASH"
        ] = []

        weight_snapshot_march: (
            dict[str, float]
            | None
        ) = None

        for month_index, month in enumerate(
            months
        ):
            quarter = (
                _quarter_for_month(
                    month_index
                )
            )

            if month_index in {
                0,
                3,
                6,
                9,
            }:
                portfolio_weights = {
                    str(key): float(value)
                    for key, value
                    in target_weights.items()
                }

            if month_index == 2:
                weight_snapshot_march = {
                    key: float(value)
                    for key, value
                    in portfolio_weights.items()
                }

            quarter_weights_raw = (
                benchmark_weights.get(
                    quarter
                )
            )

            if not isinstance(
                quarter_weights_raw,
                dict,
            ):
                raise RuntimeError(
                    f"Missing benchmark weights for {quarter}."
                )

            benchmark_month_weights = {
                str(key): float(value)
                for key, value
                in quarter_weights_raw.items()
            }

            benchmark_return = 0.0

            for sector in sectors:
                benchmark_return += (
                    benchmark_month_weights[
                        sector
                    ]
                    * monthly_returns[
                        sector
                    ][
                        month_index
                    ]
                )

            monthly_benchmark_returns.append(
                float(
                    benchmark_return
                )
            )

            cash_return = (
                float(
                    cash_rates[
                        month
                    ]
                )
                / 12.0
            )

            portfolio_return = 0.0

            for sector in sectors:
                portfolio_return += (
                    portfolio_weights[
                        sector
                    ]
                    * monthly_returns[
                        str(
                            portfolio_etfs[
                                sector
                            ]
                        )
                    ][
                        month_index
                    ]
                )

            portfolio_return += (
                portfolio_weights[
                    "CASH"
                ]
                * cash_return
            )

            monthly_portfolio_returns.append(
                float(
                    portfolio_return
                )
            )

            allocation_month = 0.0
            selection_month = 0.0
            interaction_month = 0.0

            for sector in sectors:
                portfolio_weight = (
                    portfolio_weights[
                        sector
                    ]
                )
                benchmark_weight = (
                    benchmark_month_weights[
                        sector
                    ]
                )
                benchmark_sector_return = (
                    monthly_returns[
                        sector
                    ][
                        month_index
                    ]
                )
                portfolio_sector_return = (
                    monthly_returns[
                        str(
                            portfolio_etfs[
                                sector
                            ]
                        )
                    ][
                        month_index
                    ]
                )

                allocation = (
                    (
                        portfolio_weight
                        - benchmark_weight
                    )
                    * (
                        benchmark_sector_return
                        - benchmark_return
                    )
                )

                selection = (
                    benchmark_weight
                    * (
                        portfolio_sector_return
                        - benchmark_sector_return
                    )
                )

                interaction = (
                    (
                        portfolio_weight
                        - benchmark_weight
                    )
                    * (
                        portfolio_sector_return
                        - benchmark_sector_return
                    )
                )

                allocation_month += (
                    allocation
                )
                selection_month += (
                    selection
                )
                interaction_month += (
                    interaction
                )

                sector_allocation[
                    sector
                ] += allocation
                sector_selection[
                    sector
                ] += selection
                sector_interaction[
                    sector
                ] += interaction

                monthly_sector_allocation[
                    sector
                ].append(
                    float(
                        allocation
                    )
                )

            cash_allocation = (
                portfolio_weights[
                    "CASH"
                ]
                * (
                    cash_return
                    - benchmark_return
                )
            )

            allocation_month += (
                cash_allocation
            )

            sector_allocation[
                "CASH"
            ] += cash_allocation

            monthly_sector_allocation[
                "CASH"
            ].append(
                float(
                    cash_allocation
                )
            )

            monthly_allocation.append(
                float(
                    allocation_month
                )
            )
            monthly_selection.append(
                float(
                    selection_month
                )
            )
            monthly_interaction.append(
                float(
                    interaction_month
                )
            )

            # Drift portfolio weights to next month's beginning
            # weights. Quarterly rebalance overwrites these weights
            # at the start of Jan/Apr/Jul/Oct.
            end_values: dict[
                str,
                float,
            ] = {}

            total_end_value = 0.0

            for sector in sectors:
                end_value = (
                    portfolio_weights[
                        sector
                    ]
                    * (
                        1.0
                        + monthly_returns[
                            str(
                                portfolio_etfs[
                                    sector
                                ]
                            )
                        ][
                            month_index
                        ]
                    )
                )

                end_values[
                    sector
                ] = float(
                    end_value
                )
                total_end_value += (
                    end_value
                )

            cash_end_value = (
                portfolio_weights[
                    "CASH"
                ]
                * (
                    1.0
                    + cash_return
                )
            )

            end_values[
                "CASH"
            ] = float(
                cash_end_value
            )

            total_end_value += (
                cash_end_value
            )

            if total_end_value <= 0.0:
                raise RuntimeError(
                    "Portfolio drift produced non-positive total value."
                )

            portfolio_weights = {
                key: (
                    value
                    / total_end_value
                )
                for key, value
                in end_values.items()
            }

        if weight_snapshot_march is None:
            raise RuntimeError(
                "March weight snapshot was not produced."
            )

        portfolio_return = (
            _compound(
                monthly_portfolio_returns
            )
        )
        benchmark_return = (
            _compound(
                monthly_benchmark_returns
            )
        )

        total_allocation_effect = float(
            sum(
                monthly_allocation
            )
        )
        total_selection_effect = float(
            sum(
                monthly_selection
            )
        )
        total_interaction_effect = float(
            sum(
                monthly_interaction
            )
        )

        result = {
            "portfolio_return": (
                portfolio_return
            ),
            "benchmark_return": (
                benchmark_return
            ),
            "active_return": (
                portfolio_return
                - benchmark_return
            ),
            "total_allocation_effect": (
                total_allocation_effect
            ),
            "total_selection_effect": (
                total_selection_effect
            ),
            "total_interaction_effect": (
                total_interaction_effect
            ),
            "monthly_allocation": (
                monthly_allocation
            ),
            "monthly_selection": (
                monthly_selection
            ),
            "monthly_interaction": (
                monthly_interaction
            ),
            "q1_allocation": float(
                sum(
                    monthly_allocation[
                        0:3
                    ]
                )
            ),
            "q2_allocation": float(
                sum(
                    monthly_allocation[
                        3:6
                    ]
                )
            ),
            "q3_allocation": float(
                sum(
                    monthly_allocation[
                        6:9
                    ]
                )
            ),
            "q4_allocation": float(
                sum(
                    monthly_allocation[
                        9:12
                    ]
                )
            ),
            "sector_allocation": {
                key: float(value)
                for key, value
                in sector_allocation.items()
            },
            "sector_selection": {
                key: float(value)
                for key, value
                in sector_selection.items()
            },
            "sector_interaction": {
                key: float(value)
                for key, value
                in sector_interaction.items()
            },
            "monthly_sector_allocation": {
                key: [
                    float(value)
                    for value
                    in values
                ]
                for key, values
                in monthly_sector_allocation.items()
            },
            "weight_snapshot_march": {
                key: float(value)
                for key, value
                in weight_snapshot_march.items()
            },
        }

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            out_dir
            / "results.json"
        ).write_text(
            json.dumps(
                result,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
