from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _find(task_dir: Path, name: str) -> Path:
    matches = [
        path
        for path in task_dir.rglob(name)
        if path.is_file() and "checks" not in path.parts
    ]
    if not matches:
        raise RuntimeError(
            f"Offline alpha skill could not find {name}."
        )
    return sorted(matches)[0]


@dataclass(frozen=True)
class AlphaHedgeStrategySkill:
    name: str = "alpha-hedge-strategy"

    def matches(
        self,
        *,
        instruction: str,
        task_dir: Path,
    ) -> bool:
        del task_dir

        lowered = instruction.lower()
        required = (
            "cross-sectional alpha signal",
            "fama-french",
            "long-short backtest",
            "annualized_alpha",
            "market_beta_residual",
            "solution.json",
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

        params = json.loads(
            _find(
                task_dir,
                "params.json",
            ).read_text(
                encoding="utf-8"
            )
        )

        returns = pd.read_csv(
            _find(
                task_dir,
                "returns.csv",
            )
        )

        factors = pd.read_csv(
            _find(
                task_dir,
                "factors.csv",
            )
        )

        if "date" not in returns.columns:
            raise RuntimeError(
                "returns.csv requires a date column."
            )

        if "date" not in factors.columns:
            raise RuntimeError(
                "factors.csv requires a date column."
            )

        returns["date"] = pd.to_datetime(
            returns["date"],
            errors="raise",
        )
        factors["date"] = pd.to_datetime(
            factors["date"],
            errors="raise",
        )

        returns = returns.sort_values(
            "date",
            kind="stable",
        ).reset_index(
            drop=True
        )

        factors = factors.sort_values(
            "date",
            kind="stable",
        ).reset_index(
            drop=True
        )

        tickers = sorted(
            column
            for column in returns.columns
            if column != "date"
        )

        if not tickers:
            raise RuntimeError(
                "returns.csv contains no stock columns."
            )

        raw_matrix = (
            returns[
                tickers
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        nan_count = int(
            raw_matrix
            .isna()
            .sum()
            .sum()
        )

        # The benchmark's deterministic cleaning convention treats
        # missing stock returns as zero return for that day.
        clean = raw_matrix.fillna(
            0.0
        )

        factor_names = [
            str(name)
            for name
            in params[
                "factor_names"
            ]
        ]

        missing_factors = [
            name
            for name in factor_names
            if name not in factors.columns
        ]

        if missing_factors:
            raise RuntimeError(
                "factors.csv is missing factor(s): "
                + ", ".join(
                    missing_factors
                )
            )

        factor_matrix = (
            factors[
                factor_names
            ]
            .apply(
                pd.to_numeric,
                errors="raise",
            )
            .astype(float)
        )

        if len(factor_matrix) != len(clean):
            raise RuntimeError(
                "returns.csv and factors.csv row counts differ."
            )

        if not np.array_equal(
            returns[
                "date"
            ].to_numpy(),
            factors[
                "date"
            ].to_numpy(),
        ):
            raise RuntimeError(
                "returns.csv and factors.csv dates are not aligned."
            )

        lookback_alpha = int(
            params[
                "lookback_alpha"
            ]
        )
        trade_start_day = int(
            params[
                "trade_start_day"
            ]
        )
        long_top_n = int(
            params[
                "long_top_n"
            ]
        )
        short_bottom_n = int(
            params[
                "short_bottom_n"
            ]
        )
        long_weight = float(
            params[
                "long_weight_per_stock"
            ]
        )
        short_weight = float(
            params[
                "short_weight_per_stock"
            ]
        )
        transaction_cost = (
            float(
                params[
                    "transaction_cost_bps"
                ]
            )
            / 10000.0
        )
        annualization = float(
            params[
                "annualization_factor"
            ]
        )

        current_weights = pd.Series(
            0.0,
            index=tickers,
            dtype=float,
        )

        strategy_returns: list[
            float
        ] = []
        strategy_indices: list[
            int
        ] = []
        n_rebalances = 0
        previous_month: str | None = None

        for row_index in range(
            trade_start_day,
            len(
                returns
            ),
        ):
            date = returns.loc[
                row_index,
                "date",
            ]
            month_key = date.strftime(
                "%Y-%m"
            )

            if month_key != previous_month:
                signal_start = (
                    row_index
                    - lookback_alpha
                )
                signal_end = (
                    row_index
                )

                if signal_start < 0:
                    raise RuntimeError(
                        "Insufficient history for alpha signal."
                    )

                recent = clean.iloc[
                    signal_start:signal_end
                ]

                # Mean recent return is equivalent to a summed-return
                # momentum rank for a fixed lookback. Standardizing
                # cross-sectionally preserves that rank.
                raw_signal = recent.mean(
                    axis=0
                )

                cross_mean = float(
                    raw_signal.mean()
                )
                cross_std = float(
                    raw_signal.std(
                        ddof=1
                    )
                )

                if (
                    not math.isfinite(
                        cross_std
                    )
                    or cross_std <= 0.0
                ):
                    z_signal = (
                        raw_signal
                        - cross_mean
                    )
                else:
                    z_signal = (
                        raw_signal
                        - cross_mean
                    ) / cross_std

                ranking = sorted(
                    tickers,
                    key=lambda ticker: (
                        -float(
                            z_signal[
                                ticker
                            ]
                        ),
                        ticker,
                    ),
                )

                new_weights = pd.Series(
                    0.0,
                    index=tickers,
                    dtype=float,
                )

                for ticker in ranking[
                    :long_top_n
                ]:
                    new_weights[
                        ticker
                    ] = long_weight

                for ticker in ranking[
                    -short_bottom_n:
                ]:
                    new_weights[
                        ticker
                    ] = -short_weight

                turnover = float(
                    (
                        new_weights
                        - current_weights
                    )
                    .abs()
                    .sum()
                )

                current_weights = (
                    new_weights
                )
                n_rebalances += 1
                rebalance_cost = (
                    transaction_cost
                    * turnover
                )
                previous_month = (
                    month_key
                )
            else:
                rebalance_cost = 0.0

            day_return = float(
                np.dot(
                    current_weights.to_numpy(
                        dtype=float
                    ),
                    clean.iloc[
                        row_index
                    ].to_numpy(
                        dtype=float
                    ),
                )
                - rebalance_cost
            )

            strategy_returns.append(
                day_return
            )
            strategy_indices.append(
                row_index
            )

        strategy = np.asarray(
            strategy_returns,
            dtype=float,
        )

        if len(strategy) < 2:
            raise RuntimeError(
                "Trading period is too short."
            )

        daily_mean = float(
            np.mean(
                strategy
            )
        )
        daily_std = float(
            np.std(
                strategy,
                ddof=1,
            )
        )

        annualized_return = (
            daily_mean
            * annualization
        )
        annualized_volatility = (
            daily_std
            * math.sqrt(
                annualization
            )
        )
        sharpe_ratio = (
            annualized_return
            / annualized_volatility
            if annualized_volatility > 0.0
            else 0.0
        )

        wealth = np.cumprod(
            1.0
            + strategy
        )
        running_peak = (
            np.maximum.accumulate(
                wealth
            )
        )
        drawdown = (
            running_peak
            - wealth
        ) / running_peak

        max_drawdown = float(
            np.max(
                drawdown
            )
        )

        X_factors = (
            factor_matrix.iloc[
                strategy_indices
            ]
            .to_numpy(
                dtype=float
            )
        )
        X = np.column_stack(
            (
                np.ones(
                    len(
                        strategy
                    ),
                    dtype=float,
                ),
                X_factors,
            )
        )

        coefficients, *_ = (
            np.linalg.lstsq(
                X,
                strategy,
                rcond=None,
            )
        )

        intercept = float(
            coefficients[
                0
            ]
        )
        betas = coefficients[
            1:
        ]

        fitted = X @ coefficients
        residuals = (
            strategy
            - fitted
        )

        annualized_alpha = (
            intercept
            * annualization
        )
        tracking_error = float(
            np.std(
                residuals,
                ddof=1,
            )
            * math.sqrt(
                annualization
            )
        )
        information_ratio = (
            annualized_alpha
            / tracking_error
            if tracking_error > 0.0
            else 0.0
        )
        hit_rate = float(
            np.mean(
                strategy > 0.0
            )
        )

        beta_map = {
            factor_names[index]: float(
                betas[
                    index
                ]
            )
            for index in range(
                len(
                    factor_names
                )
            )
        }

        market_beta = float(
            beta_map.get(
                "Mkt_RF",
                0.0,
            )
        )
        smb_beta = float(
            beta_map.get(
                "SMB",
                0.0,
            )
        )
        hml_beta = float(
            beta_map.get(
                "HML",
                0.0,
            )
        )

        results = {
            "annualized_return": float(
                annualized_return
            ),
            "annualized_volatility": float(
                annualized_volatility
            ),
            "sharpe_ratio": float(
                sharpe_ratio
            ),
            "max_drawdown": float(
                max_drawdown
            ),
            "annualized_alpha": float(
                annualized_alpha
            ),
            "market_beta_residual": float(
                market_beta
            ),
            "n_rebalances": int(
                n_rebalances
            ),
        }

        def value_box(value):
            return {
                "value": value
            }

        solution = {
            "intermediates": {
                "nan_count": value_box(
                    int(
                        nan_count
                    )
                ),
                "n_rebalances": value_box(
                    int(
                        n_rebalances
                    )
                ),
                "annualized_alpha": value_box(
                    float(
                        annualized_alpha
                    )
                ),
                "market_beta_residual": value_box(
                    float(
                        market_beta
                    )
                ),
                "smb_beta_residual": value_box(
                    float(
                        smb_beta
                    )
                ),
                "hml_beta_residual": value_box(
                    float(
                        hml_beta
                    )
                ),
                "tracking_error_annual": value_box(
                    float(
                        tracking_error
                    )
                ),
                "information_ratio": value_box(
                    float(
                        information_ratio
                    )
                ),
                "hit_rate": value_box(
                    float(
                        hit_rate
                    )
                ),
                "sharpe_ratio": value_box(
                    float(
                        sharpe_ratio
                    )
                ),
            }
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
                results,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        (
            out_dir
            / "solution.json"
        ).write_text(
            json.dumps(
                solution,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
